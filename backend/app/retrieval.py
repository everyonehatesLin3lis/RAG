"""Vector search over rag_chunks (Phase 6).

Retrieval = embed the question with the same model as the chunks, then ask PostgreSQL for the K chunks
whose embeddings are closest by cosine distance (`embedding <=> query`, 0 = same direction).

The question itself never reaches SQL: only its embedding does, as a bound parameter, so text like
"'; DROP TABLE movies" is harmless. PostgreSQL chooses how to run the query; at ~10k chunks it does an
exact scan, and it will use the HNSW index on its own as the table grows (specs/6.md, option 1).
"""

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app import embeddings
from app.config import get_settings
from app.errors import AppError
from app.models import RagChunk


@dataclass
class RetrievedChunk:
    id: int
    movie_id: str
    movie_title: str
    year: int | None
    content: str
    distance: float
    metadata: dict = field(default_factory=dict)


# Implements: specs/6.md#AC-002, #AC-006, #AC-007
def search_chunks(session: Session, query_vector: list[float], k: int) -> list[RetrievedChunk]:
    distance = RagChunk.embedding.cosine_distance(query_vector).label("distance")
    statement = (
        select(RagChunk.id, RagChunk.movie_id, RagChunk.content, RagChunk.metadata_, distance)
        .where(RagChunk.embedding.is_not(None))
        .order_by(distance)
        .limit(k)
    )
    try:
        rows = session.execute(statement).all()
    except SQLAlchemyError as exc:
        raise AppError("RAG_RETRIEVAL_FAILED", "Unable to retrieve movie information.", 503) from exc

    return [
        RetrievedChunk(
            id=row.id,
            movie_id=row.movie_id,
            movie_title=row.metadata_.get("movie_title", row.movie_id),
            year=row.metadata_.get("year"),
            content=row.content,
            distance=float(row.distance),
            metadata=row.metadata_,
        )
        for row in rows
    ]


# Implements: specs/6.md#AC-001
def retrieve(question: str, session: Session, k: int | None = None) -> list[RetrievedChunk]:
    query_vector = embeddings.embed_query(question)
    return search_chunks(session, query_vector, k or get_settings().retrieval_top_k)
