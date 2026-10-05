"""Phase 3 database checks against the real local PostgreSQL (docker compose up -d, alembic upgrade head).

Each test runs inside a transaction that is rolled back, so nothing is left in the database.
Skipped when the database is not reachable, so the rest of the suite still runs without Docker.
"""

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.db import get_engine
from app.errors import AppError
from app.models import Movie, RagChunk


@pytest.fixture
def session():
    try:
        connection = get_engine().connect()
    except (AppError, OperationalError) as exc:
        pytest.skip(f"database not available: {exc}")
    transaction = connection.begin()
    db = Session(bind=connection)
    try:
        yield db
    finally:
        db.close()
        transaction.rollback()
        connection.close()


def test_schema_has_all_tables_and_pgvector(session):
    tables = set(inspect(session.connection()).get_table_names())
    assert {"movies", "reviews", "rag_chunks", "conversations", "messages"} <= tables

    version = session.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")).scalar()
    assert version is not None


def test_vector_insert_retrieve_and_similarity(session):
    session.add(Movie(id="tt_test", title="Test Movie"))
    # Three toy 3-dimensional "embeddings": two point the same way, one is orthogonal.
    session.add_all(
        [
            RagChunk(movie_id="tt_test", content="tense thriller", embedding=[1.0, 0.0, 0.0]),
            RagChunk(movie_id="tt_test", content="slow-burn thriller", embedding=[0.9, 0.1, 0.0]),
            RagChunk(movie_id="tt_test", content="romantic comedy", embedding=[0.0, 0.0, 1.0]),
        ]
    )
    session.flush()

    # Retrieve: the stored vector comes back unchanged.
    stored = session.scalars(select(RagChunk).where(RagChunk.content == "tense thriller")).one()
    assert list(stored.embedding) == [1.0, 0.0, 0.0]

    # Similarity: <=> is cosine distance (0 = same direction, 1 = orthogonal). Nearest first.
    query = [1.0, 0.0, 0.0]
    distance = RagChunk.embedding.cosine_distance(query)
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
    session.add(RagChunk(movie_id="tt_test", content="x", embedding=[1.0, 0.0]))
    session.flush()

    session.execute(text("DELETE FROM movies WHERE id = 'tt_test'"))
    remaining = session.scalar(select(RagChunk.id).where(RagChunk.movie_id == "tt_test"))
    assert remaining is None
