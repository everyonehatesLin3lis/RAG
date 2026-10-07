"""Spec 6 (specs/6.md) database tests for vector search, inside a rolled-back transaction."""

from sqlalchemy import text

from app import embeddings, retrieval
from app.models import Movie, RagChunk, Review

DIMS = 1536


def toy_vector(*values: float) -> list[float]:
    return list(values) + [0.0] * (DIMS - len(values))


def add_test_chunks(session):
    # These tests check the ranking SQL, so they force an exact scan. PostgreSQL may choose the HNSW index for the
    # app's query, and HNSW is approximate: it can miss neighbours of artificial mostly-zero vectors like these
    # (on real embeddings its recall@8 against exact search was 1.000 over 200 queries; see README, Phase 17).
    session.execute(text("SET LOCAL enable_indexscan = off"))
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


# --- Phase 8: review chunks carry the original review's URL -----------------------------------------


def test_review_chunk_gets_its_review_url_and_profile_chunk_none(session):
    session.execute(text("SET LOCAL enable_indexscan = off"))  # exact scan, see add_test_chunks
    session.add(Movie(id="tt_test", title="Test Movie", year=2020))
    session.add(Review(id=999_999_001, movie_id="tt_test", review_text="Great.", source="rotten_tomatoes",
                       metadata_={"url": "https://example.com/r1"}))
    session.add_all(
        [
            RagChunk(movie_id="tt_test", content="review chunk", embedding=toy_vector(1.0),
                     metadata_={"movie_title": "Test Movie", "chunk_key": "t:r", "review_id": 999_999_001}),
            RagChunk(movie_id="tt_test", content="profile chunk", embedding=toy_vector(0.99, 0.14),
                     metadata_={"movie_title": "Test Movie", "chunk_key": "t:p", "review_id": None}),
        ]
    )
    session.flush()

    results = {r.content: r for r in retrieval.search_chunks(session, toy_vector(1.0), k=5)}

    assert results["review chunk"].url == "https://example.com/r1"
    assert results["profile chunk"].url is None


# --- AC-007: the question never becomes SQL ---------------------------------------------------------


def test_ac007_sql_injection_text_is_just_a_question(session, monkeypatch):
    add_test_chunks(session)
    monkeypatch.setattr(embeddings, "embed_query", lambda question: toy_vector(1.0))

    results = retrieval.retrieve("'; DROP TABLE movies; -- \" OR 1=1", session, k=5)

    assert results[0].content == "exact match"
    assert session.execute(text("SELECT count(*) FROM movies WHERE id = 'tt_test'")).scalar() == 1


# --- Phase 17: keyword (full-text) search ------------------------------------------------------------------


def add_text_chunks(session):
    session.add(Movie(id="tt_test", title="Test Movie", year=2020))
    texts = {
        "kw:1": "Movie: Testfilm Quokka\n\nDirector: Zelda Quillfeather\nCast: Hugo Brightwater",
        "kw:2": "Movie: Testfilm Quokka\n\nReview by A (B), 4/5, fresh:\nZelda Quillfeather directs glimmerworts with Hugo.",
        "kw:3": "Movie: Other Testfilm\n\nCast: Brightwater Hugo",  # same words, wrong order for the phrase
    }
    for key, content in texts.items():
        session.add(RagChunk(movie_id="tt_test", content=content,
                             metadata_={"movie_title": "Test Movie", "year": 2020, "chunk_key": key}))
    session.flush()


def test_keyword_search_matches_a_name_as_a_phrase(session):
    add_text_chunks(session)

    found = {r.metadata["chunk_key"] for r in retrieval.keyword_search(session, ["Hugo Brightwater"], k=10)}

    assert "kw:1" in found and "kw:3" not in found  # "Brightwater Hugo" is not the phrase


def test_keyword_search_uses_stems_and_any_keyword(session):
    add_text_chunks(session)

    found = {r.metadata["chunk_key"] for r in retrieval.keyword_search(session, ["glimmerwort", "Quillfeather"], k=10)}

    assert {"kw:1", "kw:2"} <= found  # "glimmerworts" matched the stem "glimmerwort"; either keyword is enough


def test_keyword_search_ranks_more_matches_higher_and_respects_k(session):
    add_text_chunks(session)

    results = retrieval.keyword_search(session, ["Zelda Quillfeather", "glimmerworts", "Testfilm Quokka"], k=1)

    assert len(results) == 1 and results[0].metadata["chunk_key"] == "kw:2"
    assert 0 < results[0].keyword_score <= 1 and results[0].distance is None


def test_keyword_search_with_no_usable_keywords_returns_nothing(session):
    assert retrieval.keyword_search(session, [], k=10) == []
    assert retrieval.keyword_search(session, ["  ", ""], k=10) == []
    assert retrieval.keyword_search(session, ["the"], k=10) == []  # stop words only


def test_keyword_text_is_a_bound_parameter(session):
    add_text_chunks(session)
    assert retrieval.keyword_search(session, ["x'); DROP TABLE movies; --"], k=10) == []
    assert session.execute(text("SELECT count(*) FROM movies WHERE id = 'tt_test'")).scalar() == 1
