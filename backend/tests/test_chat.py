"""Chat endpoint tests. The LLM is replaced with a fake, so these make no API calls."""

import pytest
from fastapi.testclient import TestClient

from app import llm
from app.errors import AppError
from app.main import app

client = TestClient(app)


@pytest.fixture
def fake_llm(monkeypatch):
    received = []

    async def fake_generate_answer(message: str) -> str:
        received.append(message)
        return "Try Prisoners (2013)."

    monkeypatch.setattr(llm, "generate_answer", fake_generate_answer)
    return received


def test_chat_returns_answer(fake_llm):
    response = client.post("/api/chat", json={"message": "Recommend me a thriller", "conversation_id": "123"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Try Prisoners (2013)."}
    assert fake_llm == ["Recommend me a thriller"]


def test_chat_works_without_conversation_id(fake_llm):
    response = client.post("/api/chat", json={"message": "Hi"})

    assert response.status_code == 200


@pytest.mark.parametrize("message", ["", "   ", "x" * 2001])
def test_chat_rejects_invalid_message(fake_llm, message):
    response = client.post("/api/chat", json={"message": message})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"
    assert fake_llm == []


def test_chat_rejects_invalid_json(fake_llm):
    response = client.post("/api/chat", content="{not json", headers={"Content-Type": "application/json"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_JSON"


def test_chat_reports_llm_failure(monkeypatch):
    async def failing_generate_answer(message: str) -> str:
        raise AppError("LLM_UNAVAILABLE", "The language model is unavailable right now.", 502)

    monkeypatch.setattr(llm, "generate_answer", failing_generate_answer)

    response = client.post("/api/chat", json={"message": "Hi"})

    assert response.status_code == 502
    assert response.json() == {
        "error": {"code": "LLM_UNAVAILABLE", "message": "The language model is unavailable right now."}
    }
