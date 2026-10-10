from pathlib import Path

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app import embeddings, llm
from app.db import get_engine
from app.errors import AppError

# Implements: specs/28.md#AC-001. The plan's five test groups; `pytest -m rag` runs one of them.
GROUPS = ("backend", "rag", "tools", "security", "mcp")

# Every test file and its group. A file listed as None marks each test itself (test_live.py mixes rag and security).
TEST_GROUPS = {
    "test_chat": "backend", "test_health": "backend", "test_history": "backend", "test_streaming": "backend",
    "test_errors": "backend", "test_routes": "backend", "test_groups": "backend", "test_request_log": "backend",
    "test_usage": "backend", "test_evaluation": "backend",
    "test_database": "backend", "test_embedding_db": "backend", "test_embeddings": "backend",
    "test_retrieval_db": "backend", "test_fusion": "backend", "test_documents": "backend",
    "test_query_translation": "backend",
    "test_rag": "rag", "test_rag_debug": "rag", "test_rag_questions": "rag", "test_per_film": "rag",
    "test_tools": "tools", "test_tools_db": "tools", "test_tool_calling": "tools", "test_tool_cases": "tools",
    "test_security": "security", "test_malicious_arguments": "security", "test_rate_limit": "security",
    "test_mcp": "mcp",
    "test_live": None,
}


def pytest_configure(config):
    for group in GROUPS:
        config.addinivalue_line("markers", f"{group}: the plan's {group} tests (Phase 28)")
    config.addinivalue_line("markers", "live: calls real models and costs money; run with `pytest -m live`")


@pytest.hookimpl(tryfirst=True)  # before pytest filters by -m, which needs the markers
def pytest_collection_modifyitems(config, items):
    """Give each test its file's group, unless the test is marked itself. Then leave out live tests unless the -m
    expression names them: `pytest -m rag` replaces the default `-m "not live"`, and must still not spend money."""
    for item in items:
        if any(item.get_closest_marker(g) for g in GROUPS):
            continue
        group = TEST_GROUPS.get(Path(str(item.fspath)).stem)
        if group:
            item.add_marker(group)

    if "live" not in (config.option.markexpr or ""):
        live = [item for item in items if item.get_closest_marker("live")]
        if live:
            config.hook.pytest_deselected(items=live)
            items[:] = [item for item in items if not item.get_closest_marker("live")]


@pytest.fixture(autouse=True)
def no_paid_api_calls(request, monkeypatch):
    """Tests must never reach OpenRouter. Any LLM or embedding call a test did not replace with a fake fails loudly.
    Live tests (Phase 28, `pytest -m live`) are the one exception: calling the real models is their point."""
    if request.node.get_closest_marker("live"):
        return

    def blocked(*args, **kwargs):
        raise RuntimeError("A test tried to call a paid API. Replace it with a fake.")

    monkeypatch.setattr(llm, "get_chat_model", blocked)
    monkeypatch.setattr(llm, "get_structured_model", blocked)
    monkeypatch.setattr(llm, "get_judge_model", blocked)
    monkeypatch.setattr(embeddings, "get_embedder", blocked)


@pytest.fixture(autouse=True)
def temporary_request_log(tmp_path, monkeypatch):
    """Tests write the request log to a temporary file, never to logs/requests.jsonl."""
    from app.config import get_settings

    path = tmp_path / "requests.jsonl"
    monkeypatch.setattr(get_settings(), "request_log_path", str(path))
    return path


@pytest.fixture(scope="session")
def in_process_mcp():
    """One MCP client connected to the MCP server in-process: the same protocol messages, no child process."""
    from app import mcp_server
    from app.mcp_client import McpToolClient

    client = McpToolClient(server=mcp_server.server)
    yield client
    client.close()


@pytest.fixture(autouse=True)
def mcp_without_subprocess(monkeypatch, in_process_mcp):
    """The chat pipeline (TOOL_BACKEND=mcp) uses the in-process server in tests; tests/test_mcp.py also tests the
    real subprocess over stdio."""
    from app import mcp_client

    monkeypatch.setattr(mcp_client, "get_mcp_client", lambda: in_process_mcp)


class FakeSession:
    """Stands in for a database session in API tests that fake the pipeline; only commit() and rollback() are called."""

    def commit(self):
        pass

    def rollback(self):
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
