"""Spec 28 AC-002: every API route, one successful call and one rejected or failed call each.

The real endpoints on the real (rolled-back) database; the RAG pipeline is a fake, so no API calls.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app import rag
from app.db import get_session
from app.main import app


class BrokenSession:
    """A database session whose every query fails, as when PostgreSQL is down."""

    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))

        return fail


@pytest.fixture
def api(session, monkeypatch):
    monkeypatch.setattr(rag, "answer_question",
                        lambda question, session, history=None, on_event=None: rag.RagAnswer(answer=f"re: {question}"))
    app.dependency_overrides[get_session] = lambda: session
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def api_without_working_database(api):
    app.dependency_overrides[get_session] = lambda: BrokenSession()
    return api


def conversation(api) -> str:
    return api.post("/api/conversations").json()["id"]


# Implements: specs/28.md#AC-002 (success, one per route)
@pytest.mark.parametrize("method, path, body, status", [
    ("GET", "/api/health", None, 200),
    ("POST", "/api/chat", {"message": "Who directed Prisoners?"}, 200),
    ("POST", "/api/chat/stream", {"message": "Who directed Prisoners?"}, 200),
    ("POST", "/api/conversations", None, 201),
    ("GET", "/api/conversations/{id}", None, 200),
    ("POST", "/api/conversations/{id}/messages", {"message": "hello"}, 200),
])
def test_each_route_succeeds(api, method, path, body, status):
    url = path.replace("{id}", conversation(api)) if "{id}" in path else path

    response = api.request(method, url, json=body)

    assert response.status_code == status
    if path == "/api/chat/stream":
        assert response.text.rstrip().endswith('data: {"type": "done"}')


# Implements: specs/28.md#AC-002 (rejected or failed, one per route)
@pytest.mark.parametrize("method, path, body, status, code", [
    ("POST", "/api/health", None, 405, "METHOD_NOT_ALLOWED"),
    ("POST", "/api/chat", {"message": ""}, 422, "INVALID_INPUT"),
    ("POST", "/api/chat/stream", {"message": "x" * 2001}, 422, "INVALID_INPUT"),
    ("GET", "/api/conversations/{unknown}", None, 404, "CONVERSATION_NOT_FOUND"),
    ("POST", "/api/conversations/not-a-uuid/messages", {"message": "hello"}, 422, "INVALID_INPUT"),
])
def test_each_route_rejects_bad_requests(api, method, path, body, status, code):
    response = api.request(method, path.replace("{unknown}", str(uuid.uuid4())), json=body)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code


# Implements: specs/28.md#AC-002 (POST /api/conversations fails cleanly when the database does)
def test_creating_a_conversation_without_a_database_is_database_unavailable(api_without_working_database):
    response = api_without_working_database.post("/api/conversations")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DATABASE_UNAVAILABLE"
