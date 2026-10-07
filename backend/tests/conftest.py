import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app import embeddings, llm
from app.db import get_engine
from app.errors import AppError


@pytest.fixture(autouse=True)
def no_paid_api_calls(monkeypatch):
    """Tests must never reach OpenRouter. Any LLM or embedding call a test did not replace with a fake fails loudly."""

    def blocked(*args, **kwargs):
        raise RuntimeError("A test tried to call a paid API. Replace it with a fake.")

    monkeypatch.setattr(llm, "get_chat_model", blocked)
    monkeypatch.setattr(llm, "get_structured_model", blocked)
    monkeypatch.setattr(embeddings, "get_embedder", blocked)


@pytest.fixture(autouse=True)
def temporary_request_log(tmp_path, monkeypatch):
    """Tests write the request log to a temporary file, never to logs/requests.jsonl."""
    from app.config import get_settings

    path = tmp_path / "requests.jsonl"
    monkeypatch.setattr(get_settings(), "request_log_path", str(path))
    return path


class FakeSession:
    """Stands in for a database session in API tests that fake the pipeline; only commit() is called."""

    def commit(self):
        pass


@pytest.fixture
def api_without_database(monkeypatch):
    """A TestClient whose endpoints get a FakeSession and an in-memory conversation history."""
    import uuid

    from fastapi.testclient import TestClient

    from app import history
    from app.db import get_session
    from app.main import app

    saved = []
    monkeypatch.setattr(
        history, "get_or_create_conversation",
        lambda session, conversation_id: type("Conv", (), {"id": conversation_id or uuid.uuid4()})(),
    )
    monkeypatch.setattr(history, "recent_turns", lambda session, conversation_id, limit=None: [])
    monkeypatch.setattr(history, "save_exchange", lambda session, cid, q, a: saved.append((cid, q, a)))
    fake = FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    client = TestClient(app)
    client.fake_session, client.saved = fake, saved
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def session():
    """A session on the real local database inside a transaction that is always rolled back.

    Skips the test when the database is not reachable, so the rest of the suite runs without Docker.
    """
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
