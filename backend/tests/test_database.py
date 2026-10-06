"""Phase 3 database checks against the real local PostgreSQL (docker compose up -d, alembic upgrade head).

Each test runs inside a transaction that is rolled back (see conftest.session), so nothing is left behind.
"""

import pytest
from sqlalchemy import inspect, select, text

from app.models import Movie, RagChunk

DIMS = 1536  # rag_chunks.embedding is vector(1536) since Phase 5


def toy_vector(*values: float) -> list[float]:
    """A 1536-dimensional vector whose first few numbers are `values` and the rest zero."""
    return list(values) + [0.0] * (DIMS - len(values))


def test_schema_has_all_tables_and_pgvector(session):
    tables = set(inspect(session.connection()).get_table_names())
    assert {"movies", "reviews", "rag_chunks", "conversations", "messages"} <= tables

    version = session.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")).scalar()
    assert version is not None


def test_vector_insert_retrieve_and_similarity(session):
    session.add(Movie(id="tt_test", title="Test Movie"))
    # Three toy "embeddings": two point almost the same way, one is orthogonal.
    session.add_all(
        [
            RagChunk(movie_id="tt_test", content="tense thriller", embedding=toy_vector(1.0, 0.0, 0.0)),
            RagChunk(movie_id="tt_test", content="slow-burn thriller", embedding=toy_vector(0.9, 0.1, 0.0)),
            RagChunk(movie_id="tt_test", content="romantic comedy", embedding=toy_vector(0.0, 0.0, 1.0)),
        ]
    )
    session.flush()

    # Retrieve: the stored vector comes back unchanged.
    stored = session.scalars(select(RagChunk).where(RagChunk.content == "tense thriller")).one()
    assert list(stored.embedding) == toy_vector(1.0, 0.0, 0.0)

    # Similarity: <=> is cosine distance (0 = same direction, 1 = orthogonal). Nearest first.
    distance = RagChunk.embedding.cosine_distance(toy_vector(1.0, 0.0, 0.0))
    rows = session.execute(
        select(RagChunk.content, distance.label("distance"))
        .where(RagChunk.movie_id == "tt_test")
        .order_by(distance)
    ).all()

    assert [r.content for r in rows] == ["tense thriller", "slow-burn thriller", "romantic comedy"]
    assert rows[0].distance == pytest.approx(0.0)
    assert rows[2].distance == pytest.approx(1.0)


def test_deleting_a_movie_removes_its_chunks(session):
    session.add(Movie(id="tt_test", title="Test Movie"))
    session.add(RagChunk(movie_id="tt_test", content="x", embedding=toy_vector(1.0)))
    session.flush()

    session.execute(text("DELETE FROM movies WHERE id = 'tt_test'"))
    remaining = session.scalar(select(RagChunk.id).where(RagChunk.movie_id == "tt_test"))
    assert remaining is None
