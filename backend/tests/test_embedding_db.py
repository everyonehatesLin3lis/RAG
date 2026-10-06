"""Spec 5 (specs/5.md) database tests: vector(1536) column, HNSW index, pending-chunk selection.

Run against the local PostgreSQL inside a rolled-back transaction; skipped if the database is down.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DataError, StatementError

from app.embedding_job import fetch_pending, save_embeddings
from app.models import Movie, RagChunk

DIMS = 1536


def vector(first: float = 1.0) -> list[float]:
    return [first] + [0.0] * (DIMS - 1)


def add_movie_with_chunks(session, embeddings: list):
    session.add(Movie(id="tt_test", title="Test Movie"))
    chunks = [RagChunk(movie_id="tt_test", content=f"chunk {i}", embedding=e) for i, e in enumerate(embeddings)]
    session.add_all(chunks)
    session.flush()
    return chunks


# --- AC-002: vector(1536), other lengths rejected -------------------------------------------------


def test_ac002_column_is_vector_1536(session):
    column_type = session.execute(
        text(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'rag_chunks'::regclass AND attname = 'embedding'"
        )
    ).scalar()
    assert column_type == "vector(1536)"


def test_ac002_a_1536_vector_is_stored(session):
    [chunk] = add_movie_with_chunks(session, [vector()])
    session.expire(chunk)
    assert len(chunk.embedding) == DIMS


@pytest.mark.parametrize("length", [3, 1535, 1537])
def test_ac002_other_lengths_are_rejected(session, length):
    session.add(Movie(id="tt_test", title="Test Movie"))
    session.add(RagChunk(movie_id="tt_test", content="x", embedding=[0.5] * length))
    with pytest.raises((DataError, StatementError)):
        session.flush()


# --- AC-003: HNSW cosine index, used by nearest-neighbour queries --------------------------------


def test_ac003_hnsw_cosine_index_exists(session):
    definitions = session.execute(
        text("SELECT indexdef FROM pg_indexes WHERE tablename = 'rag_chunks'")
    ).scalars().all()
    assert any("USING hnsw" in d and "vector_cosine_ops" in d for d in definitions)


def test_ac003_nearest_neighbour_query_uses_the_index(session):
    # Small tables make a sequential scan cheaper, so we tell the planner to prefer indexes and check
    # that the HNSW index can serve this query shape at all.
    session.execute(text("SET LOCAL enable_seqscan = off"))
    plan = session.execute(
        text("EXPLAIN SELECT id FROM rag_chunks ORDER BY embedding <=> CAST(:q AS vector) LIMIT 5"),
        {"q": str(vector())},
    ).scalars().all()
    assert any("hnsw" in line.lower() or "ix_rag_chunks_embedding" in line for line in plan), plan


# --- AC-004: only chunks without an embedding are selected; saving fills them -------------------


def test_ac004_only_chunks_without_embedding_are_pending(session):
    embedded, pending = add_movie_with_chunks(session, [vector(), None])

    pending_ids = {chunk_id for chunk_id, _ in fetch_pending(session.connection())}

    assert pending.id in pending_ids
    assert embedded.id not in pending_ids


def test_ac004_save_embeddings_fills_the_chunk(session):
    [chunk] = add_movie_with_chunks(session, [None])

    save_embeddings(session.connection(), [(chunk.id, vector(0.5))])

    session.expire(chunk)
    assert chunk.embedding[0] == pytest.approx(0.5)
    assert chunk.id not in {chunk_id for chunk_id, _ in fetch_pending(session.connection())}


# --- AC-005: --limit selects only a handful ---------------------------------------------------------


def test_ac005_limit_caps_the_pending_selection(session):
    add_movie_with_chunks(session, [None, None, None])
    assert len(fetch_pending(session.connection(), limit=2)) == 2
