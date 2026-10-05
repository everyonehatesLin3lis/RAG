"""initial schema: pgvector extension and the five core tables

Revision ID: 0001
Revises:
Create Date: 2026-10-05 22:50:20.042241

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMPTY_JSONB = sa.text("'{}'::jsonb")


def upgrade() -> None:
    # pgvector adds the VECTOR column type and the distance operators (<=> cosine, <-> L2, <#> inner product).
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "movies",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("year", sa.SmallInteger()),
        sa.Column("director", sa.Text()),
        sa.Column("rating", sa.Numeric(3, 1)),
        sa.Column("genres", postgresql.ARRAY(sa.Text()), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("description", sa.Text()),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=EMPTY_JSONB),
    )

    op.create_table(
        "reviews",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=False),
        sa.Column("movie_id", sa.Text(), sa.ForeignKey("movies.id", ondelete="CASCADE"), nullable=False),
        sa.Column("review_text", sa.Text(), nullable=False),
        sa.Column("review_rating", sa.Numeric(4, 2)),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=EMPTY_JSONB),
        sa.CheckConstraint("review_rating BETWEEN 0 AND 10", name="review_rating_0_to_10"),
    )
    op.create_index("ix_reviews_movie_id", "reviews", ["movie_id"])

    op.create_table(
        "rag_chunks",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("movie_id", sa.Text(), sa.ForeignKey("movies.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        # No dimension yet: fixed to vector(N) with an HNSW index once the embedding model is chosen (Phase 5).
        sa.Column("embedding", Vector()),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=EMPTY_JSONB),
    )
    op.create_index("ix_rag_chunks_movie_id", "rag_chunks", ["movie_id"])

    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="message_role"),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])


def downgrade() -> None:
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("rag_chunks")
    op.drop_table("reviews")
    op.drop_table("movies")
    # The vector extension is left installed; dropping it could break other databases' expectations
    # and it is harmless when unused.
