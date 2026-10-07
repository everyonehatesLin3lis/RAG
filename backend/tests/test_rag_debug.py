"""RAG visualisation (Phase 12): the debug object. Translation, retrieval and the model are faked: no API calls."""

import pytest

from app import main, query_translation, rag, retrieval, tool_calling
from app.history import Turn
from app.query_translation import QueryFilters, TranslatedQuery
from app.retrieval import RetrievedChunk


def chunk(chunk_id: int, distance: float, doc_type: str = "review") -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id, movie_id="tt1392214", movie_title="Prisoners", year=2013,
        content="Movie: Prisoners\nYear: 2013\n\nReview by A Critic (A Paper), 4/5, fresh:\nTense and bleak.",
        distance=distance, metadata={"doc_type": doc_type, "critic": "A Critic"},
    )


CHUNKS = [chunk(11, 0.21), chunk(12, 0.30, "profile")]
TRANSLATION = TranslatedQuery(
    semantic_query="Prisoners (2013) critical reception",
    keywords=["Prisoners", "Hugh Jackman"],
    filters=QueryFilters(genres=["Thriller"]),
)


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(query_translation, "translate_query", lambda message, history=None: TRANSLATION)
    monkeypatch.setattr(retrieval, "retrieve", lambda query, session, k=None: list(CHUNKS))
    monkeypatch.setattr(tool_calling, "run_with_tools", lambda messages, tool_list: ("Very tense.", []))


def test_debug_records_what_each_step_did(pipeline):
    history = [Turn("user", "hi"), Turn("assistant", "hello")]

    result = rag.answer_question("why do people like prisoners?", session=None, history=history)

    debug = result.debug
    assert debug.original_query == "why do people like prisoners?"
    assert debug.translation is TRANSLATION
    assert debug.history_messages == 2
    assert [c.id for c in debug.vector_results] == [11, 12]
    assert debug.selected_chunk_ids == [11, 12]
    assert set(debug.timings_ms) == {"translation", "embedding_and_search", "generation", "total"}
    assert all(isinstance(ms, int) and ms >= 0 for ms in debug.timings_ms.values())


def test_no_results_still_explains_what_happened(monkeypatch):
    monkeypatch.setattr(query_translation, "translate_query", lambda message, history=None: TRANSLATION)
    monkeypatch.setattr(retrieval, "retrieve", lambda query, session, k=None: [])

    result = rag.answer_question("anything?", session=None)

    assert result.answer == rag.NO_RESULTS_ANSWER
    assert result.debug.vector_results == [] and result.debug.selected_chunk_ids == []
    assert "generation" not in result.debug.timings_ms


def test_debug_is_mapped_for_the_api(pipeline):
    debug = rag.answer_question("why do people like prisoners?", session=None).debug

    out = main.to_debug(debug).model_dump()

    assert out["original_query"] == "why do people like prisoners?"
    assert out["translated_query"] == "Prisoners (2013) critical reception"
    assert out["keywords"] == ["Prisoners", "Hugh Jackman"]
    assert out["filters"] == {"genres": ["Thriller"]}  # empty filters left out
    assert out["translation_origin"] == "model"
    assert out["selected_chunks"] == ["11", "12"]
    first = out["vector_results"][0]
    assert first == {
        "rank": 1, "chunk_id": "11", "movie": "Prisoners", "year": 2013, "doc_type": "review",
        "critic": "A Critic", "distance": 0.21, "similarity": 0.79, "excerpt": "Tense and bleak.",
    }


def test_fallback_translation_is_visible(monkeypatch, pipeline):
    monkeypatch.setattr(
        query_translation, "translate_query", lambda message, history=None: TranslatedQuery.passthrough(message)
    )

    out = main.to_debug(rag.answer_question("films like Zodiac", session=None).debug)

    assert out.translation_origin == "fallback"
    assert out.translated_query == "films like Zodiac"


def test_chat_response_includes_debug(api_without_database, pipeline):
    body = api_without_database.post("/api/chat", json={"message": "why do people like prisoners?"}).json()

    assert body["debug"]["translated_query"] == "Prisoners (2013) critical reception"
    assert [r["chunk_id"] for r in body["debug"]["vector_results"]] == ["11", "12"]
