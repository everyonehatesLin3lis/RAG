"""Chat endpoint tests. The RAG pipeline and the database session are replaced with fakes: no API calls."""

import pytest
from fastapi.testclient import TestClient

from app import rag
from app.db import get_session
from app.errors import AppError
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def no_database():
    app.dependency_overrides[get_session] = lambda: "fake-session"
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def fake_rag(monkeypatch):
    received = []

    def fake_answer_question(question, session):
        received.append((question, session))
        return "Try Prisoners (2013)."

    monkeypatch.setattr(rag, "answer_question", fake_answer_question)
    return received


def test_chat_returns_answer(fake_rag):
    response = client.post("/api/chat", json={"message": "Recommend me a thriller", "conversation_id": "123"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Try Prisoners (2013)."}
    assert fake_rag == [("Recommend me a thriller", "fake-session")]


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
