"""Logging and monitoring (Phase 15). The pipeline is faked; the log goes to a temporary file (conftest)."""

import json

import pytest
from fastapi.testclient import TestClient

from app import rag, request_log
from app.errors import AppError
from app.query_translation import TranslatedQuery
from app.retrieval import RetrievedChunk
from app.tool_calling import ToolCallRecord
from app.usage import ModelUsage

PLAN_FIELDS = {
    "timestamp", "conversation_id", "query", "translated_query", "retrieved_chunks",
    "tools", "tokens", "cost", "latency_ms", "status",
}


def lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def full_answer() -> rag.RagAnswer:
    chunk = RetrievedChunk(1, "tt1", "Zodiac", 2007, "Movie: Zodiac\n\nObsessive.", 0.2, {"doc_type": "review"})
    debug = rag.RagDebug(
        original_query="which is rated higher, zodiac or prisoners?",
        translation=TranslatedQuery(semantic_query="Zodiac vs Prisoners ratings"),
        history_messages=0, vector_results=[chunk], selected_chunk_ids=[1], timings_ms={"total": 5},
    )
    return rag.RagAnswer(
        answer="Prisoners.",
        sources=[chunk],
        tool_calls=[
            ToolCallRecord("compare_movies", {"movie_a": "Zodiac", "movie_b": "Prisoners"}, {"higher_imdb_rating": "Prisoners"}),
            ToolCallRecord("rating_summary", {"movie": "Beauty and the Beast"}, {"error": {"code": "AMBIGUOUS_TITLE", "message": "…"}}),
        ],
        debug=debug,
        usage=[
            ModelUsage("google/gemini-3.1-flash-lite", calls=1, input_tokens=860, output_tokens=60, cost_usd=0.0003),
            ModelUsage("xiaomi/mimo-v2.6-flash", calls=2, input_tokens=4000, output_tokens=200, cost_usd=0.0001),
        ],
    )


def test_a_successful_request_writes_one_line_with_the_plans_fields(api_without_database, monkeypatch, temporary_request_log):
    monkeypatch.setattr(rag, "answer_question", lambda q, s, h=None: full_answer())

    api_without_database.post("/api/chat", json={"message": "which is rated higher, zodiac or prisoners?"})

    [entry] = lines(temporary_request_log)
    assert PLAN_FIELDS <= set(entry)
    assert entry["query"] == "which is rated higher, zodiac or prisoners?"
    assert entry["translated_query"] == "Zodiac vs Prisoners ratings"
    assert entry["retrieved_chunks"] == 1
    assert entry["tools"] == ["compare_movies", "rating_summary"]
    assert entry["tool_errors"] == ["AMBIGUOUS_TITLE"]
    assert entry["tokens"] == 860 + 60 + 4000 + 200
    assert entry["cost"] == pytest.approx(0.0004)
    assert entry["models"]["xiaomi/mimo-v2.6-flash"]["tokens"] == 4200
    assert entry["status"] == "success"
    assert isinstance(entry["latency_ms"], int) and entry["conversation_id"]


def test_no_results_is_its_own_status(api_without_database, monkeypatch, temporary_request_log):
    monkeypatch.setattr(rag, "answer_question", lambda q, s, h=None: rag.RagAnswer(rag.NO_RESULTS_ANSWER))

    api_without_database.post("/api/chat", json={"message": "anything?"})

    assert lines(temporary_request_log)[0]["status"] == "no_results"


def test_a_failed_request_is_logged_with_its_error_code(api_without_database, monkeypatch, temporary_request_log):
    def failing(q, s, h=None):
        raise AppError("LLM_UNAVAILABLE", "down", 502)

    monkeypatch.setattr(rag, "answer_question", failing)

    response = api_without_database.post("/api/chat", json={"message": "hi"})

    assert response.status_code == 502
    [entry] = lines(temporary_request_log)
    assert entry["status"] == "error" and entry["error_code"] == "LLM_UNAVAILABLE"


def test_an_unexpected_crash_is_logged_as_internal_error(api_without_database, monkeypatch, temporary_request_log):
    from app.main import app

    def crashing(q, s, h=None):
        raise RuntimeError("bug")

    monkeypatch.setattr(rag, "answer_question", crashing)
    client = TestClient(app, raise_server_exceptions=False)

    assert client.post("/api/chat", json={"message": "hi"}).status_code == 500
    assert lines(temporary_request_log)[0]["error_code"] == "INTERNAL_ERROR"


def test_invalid_input_never_reaches_the_pipeline_or_the_log(api_without_database, temporary_request_log):
    api_without_database.post("/api/chat", json={"message": ""})
    assert not temporary_request_log.exists()


def test_a_broken_log_file_never_breaks_the_chat(api_without_database, monkeypatch, tmp_path):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "request_log_path", str(tmp_path))  # a directory: writing fails
    monkeypatch.setattr(rag, "answer_question", lambda q, s, h=None: full_answer())

    assert api_without_database.post("/api/chat", json={"message": "hi"}).status_code == 200


def test_the_log_line_holds_no_secrets(api_without_database, monkeypatch, temporary_request_log):
    from app.config import get_settings

    secret = get_settings().openrouter_api_key.get_secret_value() if get_settings().openrouter_api_key else "sk-test"
    monkeypatch.setattr(rag, "answer_question", lambda q, s, h=None: full_answer())

    api_without_database.post("/api/chat", json={"message": "hi"})

    text = temporary_request_log.read_text(encoding="utf-8")
    assert secret not in text and "postgresql" not in text


# --- monitoring ----------------------------------------------------------------------------------------


def test_summary_counts_statuses_latency_cost_and_tools():
    entries = [
        {"status": "success", "latency_ms": 10_000, "cost": 0.0004, "tokens": 5000, "retrieved_chunks": 8,
         "tools": ["compare_movies"], "tool_errors": [], "translation_origin": "model",
         "models": {"m": {"tokens": 5000, "cost": 0.0004}}},
        {"status": "success", "latency_ms": 20_000, "cost": 0.0008, "tokens": 6000, "retrieved_chunks": 8,
         "tools": [], "tool_errors": [], "translation_origin": "fallback",
         "models": {"m": {"tokens": 6000, "cost": 0.0008}}},
        {"status": "error", "latency_ms": 30_000, "error_code": "LLM_TIMEOUT"},
    ]

    s = request_log.summarise(entries)

    assert s["requests"] == 3 and s["by_status"] == {"success": 2, "error": 1}
    assert s["error_rate"] == pytest.approx(0.333, abs=0.001)
    assert s["error_codes"] == {"LLM_TIMEOUT": 1}
    assert s["latency_ms"] == {"median": 20_000, "p95": 30_000, "max": 30_000}
    assert s["cost_usd"]["total"] == pytest.approx(0.0012)
    assert s["cost_usd"]["average_per_answer"] == pytest.approx(0.0006)
    assert s["tool_calls"] == {"compare_movies": 1} and s["answers_using_tools_pct"] == 50.0
    assert s["translation_fallbacks"] == 1
    assert s["models"]["m"]["tokens"] == 11000


def test_damaged_lines_are_skipped(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"status": "success", "latency_ms": 5}\n{not json\n', encoding="utf-8")
    assert len(request_log.read(path)) == 1
