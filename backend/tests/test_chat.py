"""Chat endpoint tests. The RAG pipeline and the database session are replaced with fakes: no API calls."""

import pytest
from fastapi.testclient import TestClient

from app import rag
from app.db import get_session
from app.errors import AppError
from app.main import app
from app.retrieval import RetrievedChunk

client = TestClient(app)


@pytest.fixture(autouse=True)
def no_database():
    app.dependency_overrides[get_session] = lambda: "fake-session"
    yield
    app.dependency_overrides.clear()


REVIEW_CHUNK = RetrievedChunk(
    id=2281, movie_id="tt1392214", movie_title="Prisoners", year=2013,
    content="Movie: Prisoners\nYear: 2013\n\nReview by A.O. Scott (New York Times), 4/5, fresh:\nGripping and grim.",
    distance=0.31,
    metadata={"doc_type": "review", "review_id": 1822, "source": "rotten_tomatoes",
              "critic": "A.O. Scott", "publication": "New York Times"},
    url="https://example.com/review",
)
PROFILE_CHUNK = RetrievedChunk(
    id=7, movie_id="tt1392214", movie_title="Prisoners", year=2013,
    content="Movie: Prisoners\nYear: 2013\n\nDirector: Denis Villeneuve",
    distance=0.35,
    metadata={"doc_type": "profile", "review_id": None, "source": "tmdb_imdb"},
)


@pytest.fixture
def fake_rag(monkeypatch):
    received = []

    def fake_answer_question(question, session):
        received.append((question, session))
        return rag.RagAnswer(answer="Try Prisoners (2013).", sources=[REVIEW_CHUNK, PROFILE_CHUNK])

    monkeypatch.setattr(rag, "answer_question", fake_answer_question)
    return received


def test_chat_returns_answer_and_sources(fake_rag):
    response = client.post("/api/chat", json={"message": "Recommend me a thriller", "conversation_id": "123"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Try Prisoners (2013)."
    assert fake_rag == [("Recommend me a thriller", "fake-session")]
    # Phase 8: the plan's citation fields, as strings, plus display fields
    assert body["sources"] == [
        {
            "movie": "Prisoners", "year": 2013, "review_id": "1822", "chunk_id": "2281",
            "source": "rotten_tomatoes", "critic": "A.O. Scott", "publication": "New York Times",
            "url": "https://example.com/review", "excerpt": "Gripping and grim.",
        },
        {
            "movie": "Prisoners", "year": 2013, "review_id": None, "chunk_id": "7",
            "source": "tmdb_imdb", "critic": None, "publication": None,
            "url": None, "excerpt": "Director: Denis Villeneuve",
        },
    ]


def test_no_results_means_no_sources(monkeypatch):
    monkeypatch.setattr(rag, "answer_question", lambda question, session: rag.RagAnswer(answer=rag.NO_RESULTS_ANSWER))

    body = client.post("/api/chat", json={"message": "Hi"}).json()

    assert body == {"answer": rag.NO_RESULTS_ANSWER, "sources": [], "tool_calls": []}


def test_chat_works_without_conversation_id(fake_rag):
    response = client.post("/api/chat", json={"message": "Hi"})

    assert response.status_code == 200


@pytest.mark.parametrize("message", ["", "   ", "x" * 2001])
def test_chat_rejects_invalid_message(fake_rag, message):
    response = client.post("/api/chat", json={"message": message})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"
    assert fake_rag == []


def test_chat_rejects_invalid_json(fake_rag):
    response = client.post("/api/chat", content="{not json", headers={"Content-Type": "application/json"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_JSON"


@pytest.mark.parametrize(
    "code, status",
    [("LLM_UNAVAILABLE", 502), ("EMBEDDING_FAILED", 502), ("RAG_RETRIEVAL_FAILED", 503)],
)
def test_chat_reports_pipeline_failures_in_the_standard_shape(monkeypatch, code, status):
    # Implements: specs/6.md#AC-006 (endpoint level)
    def failing(question, session):
        raise AppError(code, "Something failed.", status)

    monkeypatch.setattr(rag, "answer_question", failing)

    response = client.post("/api/chat", json={"message": "Hi"})

    assert response.status_code == status
    assert response.json() == {"error": {"code": code, "message": "Something failed."}}
