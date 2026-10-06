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
