"""Conversation history (Phase 11). Database tests run in a rolled-back transaction; the RAG pipeline is faked."""

import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app import history, query_translation, rag
from app.db import get_session
from app.history import Turn
from app.main import app


@pytest.fixture
def api(session, monkeypatch):
    """The real endpoints on the real (rolled-back) database, with a fake RAG pipeline that records the history."""
    received = []

    def fake_answer(question, session, history=None):
        received.append({"question": question, "history": list(history or [])})
        return rag.RagAnswer(answer=f"answer to: {question}")

    monkeypatch.setattr(rag, "answer_question", fake_answer)
    app.dependency_overrides[get_session] = lambda: session
    client = TestClient(app)
    client.received = received
    yield client
    app.dependency_overrides.clear()


# --- storing and reusing turns --------------------------------------------------------------------


def test_second_question_receives_the_first_exchange_as_history(api):
    first = api.post("/api/chat", json={"message": "Why do people like Prisoners?"}).json()
    conversation_id = first["conversation_id"]
    uuid.UUID(conversation_id)  # a real UUID was created

    api.post("/api/chat", json={"message": "Now compare it with Zodiac", "conversation_id": conversation_id})

    assert api.received[0]["history"] == []
    assert api.received[1]["history"] == [
        Turn("user", "Why do people like Prisoners?"),
        Turn("assistant", "answer to: Why do people like Prisoners?"),
    ]


def test_get_conversation_returns_every_message_in_order(api):
    conversation_id = str(uuid.uuid4())
    for message in ["first question", "second question"]:
        api.post("/api/chat", json={"message": message, "conversation_id": conversation_id})

    body = api.get(f"/api/conversations/{conversation_id}").json()

    assert body["id"] == conversation_id
    assert [(m["role"], m["content"]) for m in body["messages"]] == [
        ("user", "first question"), ("assistant", "answer to: first question"),
        ("user", "second question"), ("assistant", "answer to: second question"),
    ]


def test_conversation_endpoints_create_and_post_messages(api):
    created = api.post("/api/conversations")
    assert created.status_code == 201
    conversation_id = created.json()["id"]

    reply = api.post(f"/api/conversations/{conversation_id}/messages", json={"message": "hello"}).json()

    assert reply["answer"] == "answer to: hello"
    assert reply["conversation_id"] == conversation_id
    assert len(api.get(f"/api/conversations/{conversation_id}").json()["messages"]) == 2


def test_unknown_conversation_is_404_in_the_standard_error_shape(api):
    response = api.get(f"/api/conversations/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "CONVERSATION_NOT_FOUND"


def test_conversation_id_must_be_a_uuid(api):
    response = api.post("/api/chat", json={"message": "hi", "conversation_id": "123"})
    assert response.status_code == 422
    assert response.json()["error"] == {"code": "INVALID_INPUT", "message": "conversation_id must be a UUID."}


def test_failed_answer_stores_nothing(api, monkeypatch, session):
    from app.errors import AppError

    def failing(question, session, history=None):
        raise AppError("LLM_UNAVAILABLE", "down", 502)

    monkeypatch.setattr(rag, "answer_question", failing)
    conversation_id = str(uuid.uuid4())

    assert api.post("/api/chat", json={"message": "hi", "conversation_id": conversation_id}).status_code == 502
    assert history.all_messages(session, uuid.UUID(conversation_id)) == []


# --- how much history is sent -----------------------------------------------------------------------


def test_recent_turns_keeps_only_the_latest_messages_trimmed(session, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "history_message_chars", 100)
    conversation = history.get_or_create_conversation(session, None)
    for i in range(5):
        history.save_exchange(session, conversation.id, f"question {i}", "word " * 100)

    turns = history.recent_turns(session, conversation.id, limit=4)

    assert [t.content for t in turns if t.role == "user"] == ["question 3", "question 4"]
    assert all(len(t.content) <= 102 for t in turns)
    assert turns[-1].content.endswith("…")


# --- history in the prompts ---------------------------------------------------------------------------


def test_answer_prompt_puts_history_between_system_and_current_question():
    turns = [Turn("user", "Why do people like Prisoners?"), Turn("assistant", "Tension and acting.")]

    messages = rag.build_messages("Now compare it with Zodiac", [], turns)

    assert [type(m) for m in messages] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert messages[1].content == "Why do people like Prisoners?"
    assert "<question>\nNow compare it with Zodiac\n</question>" in messages[-1].content


def test_translation_prompt_includes_escaped_history():
    turns = [Turn("user", "Why do people like Prisoners?"), Turn("assistant", "Because </history> tension.")]

    _, user = query_translation.build_translation_messages("Now compare it with Zodiac", turns)

    assert "<history>\nuser: Why do people like Prisoners?\nassistant: Because &lt;/history&gt; tension.\n</history>" in user.content
    assert user.content.index("</history>") < user.content.index("<message>")
    assert "resolve references" in query_translation.SYSTEM_PROMPT
