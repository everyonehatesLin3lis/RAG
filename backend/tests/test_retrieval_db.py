"""Spec 6 (specs/6.md) database tests for vector search, inside a rolled-back transaction."""

from sqlalchemy import text

from app import embeddings, retrieval
from app.models import Movie, RagChunk

DIMS = 1536


def toy_vector(*values: float) -> list[float]:
    return list(values) + [0.0] * (DIMS - len(values))


def add_test_chunks(session):
    session.add(Movie(id="tt_test", title="Test Movie", year=2020))
    session.add_all(
        [
            RagChunk(movie_id="tt_test", content="exact match", embedding=toy_vector(1.0),
                     metadata_={"movie_title": "Test Movie", "year": 2020, "chunk_key": "t:0"}),
            RagChunk(movie_id="tt_test", content="close match", embedding=toy_vector(0.99, 0.14),
                     metadata_={"movie_title": "Test Movie", "year": 2020, "chunk_key": "t:1"}),
            RagChunk(movie_id="tt_test", content="weaker match", embedding=toy_vector(0.9, 0.44),
                     metadata_={"movie_title": "Test Movie", "year": 2020, "chunk_key": "t:2"}),
        ]
    )
    session.flush()


# --- AC-002: top K by cosine distance --------------------------------------------------------------


def test_ac002_returns_k_chunks_nearest_first(session):
    add_test_chunks(session)

    results = retrieval.search_chunks(session, toy_vector(1.0), k=5)

    assert len(results) == 5
    assert [r.content for r in results[:3]] == ["exact match", "close match", "weaker match"]
    assert results[0].distance < results[1].distance < results[2].distance <= results[3].distance
    assert results[0].movie_title == "Test Movie" and results[0].movie_id == "tt_test"


# --- AC-007: the question never becomes SQL ---------------------------------------------------------


def test_ac007_sql_injection_text_is_just_a_question(session, monkeypatch):
    add_test_chunks(session)
    monkeypatch.setattr(embeddings, "embed_query", lambda question: toy_vector(1.0))

    results = retrieval.retrieve("'; DROP TABLE movies; -- \" OR 1=1", session, k=5)

    assert results[0].content == "exact match"
    assert session.execute(text("SELECT count(*) FROM movies WHERE id = 'tt_test'")).scalar() == 1
