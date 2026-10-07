"""Database tables as SQLAlchemy models. The schema itself is created by Alembic migrations
(backend/migrations/versions), never by Base.metadata.create_all, so Cloud SQL can replay it.

Structured movie fields are SQL columns; flexible extras go in a JSONB `metadata` column.
`metadata` is a reserved attribute name on SQLAlchemy models, so the Python attribute is
`metadata_` while the database column is still called `metadata`.
"""

import uuid
from datetime import datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Numeric,
    SmallInteger,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Movie(Base):
    __tablename__ = "movies"

    # IMDb ID ("tt1392214"): stable across re-ingestion and across databases.
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(SmallInteger)
    director: Mapped[str | None] = mapped_column(Text)  # "Evan Goldberg, Seth Rogen" when several
    rating: Mapped[Decimal | None] = mapped_column(Numeric(3, 1))  # IMDb average, 1.0–10.0
    genres: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    description: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, server_default=text("'{}'::jsonb"))

    reviews: Mapped[list["Review"]] = relationship(back_populates="movie", passive_deletes=True)


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (
        CheckConstraint("review_rating BETWEEN 0 AND 10", name="review_rating_0_to_10"),
    )

    # Rotten Tomatoes review ID, kept so a citation can point back to the original review.
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    movie_id: Mapped[str] = mapped_column(ForeignKey("movies.id", ondelete="CASCADE"), index=True)
    review_text: Mapped[str] = mapped_column(Text)
    review_rating: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))  # normalised to 0–10
    source: Mapped[str] = mapped_column(Text)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, server_default=text("'{}'::jsonb"))

    movie: Mapped[Movie] = relationship(back_populates="reviews")


class RagChunk(Base):
    __tablename__ = "rag_chunks"
    __table_args__ = (
        # metadata->>'chunk_key' ("review:122525:0") identifies a chunk across re-ingestion runs,
        # so ingestion can upsert and keep embeddings of unchanged chunks.
        Index("ux_rag_chunks_chunk_key", text("(metadata->>'chunk_key')"), unique=True),
        # Implements: specs/5.md#AC-003. HNSW: approximate nearest-neighbour graph index; vector_cosine_ops
        # makes it serve ORDER BY embedding <=> query (cosine distance).
        Index(
            "ix_rag_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # Phase 17 keyword search: GIN index over the generated tsvector (migration 0004).
        Index("ix_rag_chunks_search_vector", "search_vector", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    movie_id: Mapped[str] = mapped_column(ForeignKey("movies.id", ondelete="CASCADE"), index=True)
    content: Mapped[str] = mapped_column(Text)
    # Implements: specs/5.md#AC-002. 1536 = openai/text-embedding-3-small. Changing the model means a
    # migration to the new size and re-embedding every chunk.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536))
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, server_default=text("'{}'::jsonb"))
    # Phase 17: computed by PostgreSQL from content (stems + positions) for full-text keyword search.
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english'::regconfig, content)", persisted=True)
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", order_by="Message.id", passive_deletes=True
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (CheckConstraint("role IN ('user', 'assistant')", name="message_role"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
