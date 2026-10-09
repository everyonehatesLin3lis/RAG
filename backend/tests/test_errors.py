"""Phase 27: every failure the plan lists, through both chat endpoints, plus the two fallbacks and the time limit.

The plan's list: OpenRouter failure, database failure, embedding failure, no search results, tool error, MCP error,
invalid JSON, invalid user input, timeout. Each gives the standard {"error": {"code", "message"}} (or, for the stream,
an error event), or, where the answer can still be produced, an answer with a warning. Fakes only: no API calls.
"""

import json
import sys
import time

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from mcp.client.stdio import StdioServerParameters
from sqlalchemy.exc import OperationalError

from app import deadline, embeddings, history, llm, mcp_client, query_translation, retrieval, tools
from app.config import get_settings
from app.errors import AppError
from app.mcp_client import McpToolClient
from app.query_translation import TranslatedQuery
from app.retrieval import RetrievedChunk

REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
CHUNK = RetrievedChunk(2281, "tt1392214", "Prisoners", 2013, "Movie: Prisoners\n\nDirector: Denis Villeneuve", 0.2,
                       {"doc_type": "profile"})


class Model:
    """A chat model that answers, or raises `fail` from invoke() and stream()."""

    def __init__(self, fail: Exception | None = None, reply: AIMessage | None = None):
        self.fail, self.reply = fail, reply or AIMessage(content="Denis Villeneuve directed it.")

    def bind_tools(self, *args, **kwargs):
        return self

    def invoke(self, messages):
        if self.fail:
            raise self.fail
        return self.reply

    def stream(self, messages):
        if self.fail:
            raise self.fail
        yield AIMessageChunk(content=self.reply.content, tool_calls=self.reply.tool_calls)


@pytest.fixture
def pipeline(monkeypatch, api_without_database):
    """The real pipeline with fast fakes around it; each test breaks one part."""
    monkeypatch.setattr(query_translation, "translate_query",
                        lambda m, h=None: TranslatedQuery(semantic_query=m, keywords=["Prisoners"]))
    monkeypatch.setattr(embeddings, "embed_query", lambda text: [0.1] * 1536)
    monkeypatch.setattr(retrieval, "search_chunks", lambda s, v, k: [CHUNK])
    monkeypatch.setattr(retrieval, "keyword_search", lambda s, kw, k=None: [CHUNK])
    monkeypatch.setattr(llm, "get_chat_model", lambda: Model())
    monkeypatch.setattr(get_settings(), "tool_backend", "local")
    return api_without_database


def both(client, message="Who directed Prisoners?"):
    """(JSON response, last stream event) for one question."""
    plain = client.post("/api/chat", json={"message": message})
    stream = client.post("/api/chat/stream", json={"message": message})
    events = [json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: ")]
    return plain, events


def assert_error(plain, events, code: str, status: int):
    assert plain.status_code == status
    assert plain.json()["error"]["code"] == code and plain.json()["error"]["message"]
    assert events[-1]["type"] == "error" and events[-1]["error"]["code"] == code


# --- OpenRouter ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("exc, code, status", [
    (openai.APIConnectionError(request=REQUEST), "LLM_UNAVAILABLE", 502),
    (openai.APITimeoutError(request=REQUEST), "LLM_TIMEOUT", 504),
    (openai.RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None), "LLM_RATE_LIMITED", 429),
    (openai.AuthenticationError("bad key", response=httpx.Response(401, request=REQUEST), body=None), "LLM_AUTH_FAILED", 502),
    (openai.APIStatusError("Insufficient credits", response=httpx.Response(402, request=REQUEST), body=None),
     "LLM_OUT_OF_CREDIT", 503),
])
def test_openrouter_failures_have_their_own_codes(pipeline, monkeypatch, exc, code, status):
    monkeypatch.setattr(llm, "get_chat_model", lambda: Model(fail=exc))

    plain, events = both(pipeline)

    assert_error(plain, events, code, status)
    assert pipeline.saved == []  # a failed exchange is not stored


def test_provider_details_never_reach_the_user(pipeline, monkeypatch):
    secret = openai.AuthenticationError("Incorrect API key provided: sk-or-v1-abc123",
                                        response=httpx.Response(401, request=REQUEST), body=None)
    monkeypatch.setattr(llm, "get_chat_model", lambda: Model(fail=secret))

    plain, events = both(pipeline)

    assert "sk-or" not in plain.text and "sk-or" not in json.dumps(events)


# --- database -----------------------------------------------------------------------------------------------


def test_database_failure_is_database_unavailable(pipeline, monkeypatch):
    def down(*args, **kwargs):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    monkeypatch.setattr(history, "recent_turns", down)

    assert_error(*both(pipeline), "DATABASE_UNAVAILABLE", 503)


# --- embedding: keyword search takes over -------------------------------------------------------------------


def embedding_down(monkeypatch):
    def broken(text):
        raise AppError("EMBEDDING_FAILED", "The embedding model is unavailable right now.", 502)

    monkeypatch.setattr(embeddings, "embed_query", broken)


def test_embedding_failure_falls_back_to_keyword_search(pipeline, monkeypatch):
    embedding_down(monkeypatch)

    plain, events = both(pipeline)

    assert plain.status_code == 200
    debug = plain.json()["debug"]
    assert debug["vector_results"] == [] and debug["selected_chunks"] == ["2281"]  # from keyword search
    assert "keyword search only" in debug["warnings"][0] and "EMBEDDING_FAILED" in debug["warnings"][0]
    assert events[-1] == {"type": "done"}


def test_embedding_failure_without_keywords_is_still_an_error(pipeline, monkeypatch):
    embedding_down(monkeypatch)
    monkeypatch.setattr(query_translation, "translate_query", lambda m, h=None: TranslatedQuery.passthrough(m))

    assert_error(*both(pipeline), "EMBEDDING_FAILED", 502)


# --- no search results ----------------------------------------------------------------------------------------


def test_no_search_results_is_an_answer_saying_so(pipeline, monkeypatch):
    monkeypatch.setattr(retrieval, "search_chunks", lambda s, v, k: [])
    monkeypatch.setattr(retrieval, "keyword_search", lambda s, kw, k=None: [])

    plain, events = both(pipeline)

    assert plain.status_code == 200 and plain.json()["sources"] == []
    assert "couldn't find" in plain.json()["answer"]
    assert events[-1] == {"type": "done"}


# --- tools and MCP ----------------------------------------------------------------------------------------------


def test_a_tool_error_goes_to_the_model_not_to_the_user(pipeline, monkeypatch):
    call = AIMessage(content="", tool_calls=[{"name": "rating_summary", "args": {"movie": ""}, "id": "c1"}])
    replies = iter([call, AIMessage(content="Which movie do you mean?")])
    monkeypatch.setattr(llm, "get_chat_model", lambda: Model())
    monkeypatch.setattr(llm, "invoke", lambda model, messages: next(replies))

    plain = pipeline.post("/api/chat", json={"message": "Rate it"})

    assert plain.status_code == 200
    assert plain.json()["tool_calls"][0]["result"]["error"]["code"] == "INVALID_ARGUMENTS"


def test_mcp_server_that_cannot_start_falls_back_to_local_tools(pipeline, monkeypatch):
    broken = McpToolClient(server=StdioServerParameters(command=sys.executable, args=["-c", "raise SystemExit(1)"]),
                           call_timeout_s=5)
    monkeypatch.setattr(get_settings(), "tool_backend", "mcp")
    monkeypatch.setattr(mcp_client, "get_mcp_client", lambda: broken)

    plain = pipeline.post("/api/chat", json={"message": "Who directed Prisoners?"})

    assert plain.status_code == 200
    assert plain.json()["debug"]["tool_backend"] == "local"
    assert "MCP tool server was unavailable" in plain.json()["debug"]["warnings"][0]


def test_mcp_call_that_fails_mid_answer_runs_in_process(pipeline, monkeypatch, in_process_mcp):
    call = AIMessage(content="", tool_calls=[{"name": "filter_movies", "args": {"rating_min": 42}, "id": "c1"}])
    replies = iter([call, AIMessage(content="Done.")])
    monkeypatch.setattr(get_settings(), "tool_backend", "mcp")
    monkeypatch.setattr(in_process_mcp, "call_tool",
                        lambda name, args: {"error": {"code": "MCP_UNAVAILABLE", "message": "gone"}})
    monkeypatch.setattr(llm, "invoke", lambda model, messages: next(replies))
    ran_locally = []
    monkeypatch.setattr(tools, "run_tool", lambda s, name, args: ran_locally.append(name) or {"error": {"code": "INVALID_ARGUMENTS"}})

    body = pipeline.post("/api/chat", json={"message": "Films rated above 42?"}).json()

    assert ran_locally == ["filter_movies"]
    assert body["tool_calls"][0]["result"]["error"]["code"] == "INVALID_ARGUMENTS"  # the local result, not MCP_UNAVAILABLE
    assert body["debug"]["tool_backend"] == "mcp, then local" and body["debug"]["warnings"]


# --- invalid requests ----------------------------------------------------------------------------------------------


def test_invalid_json_and_invalid_input(pipeline):
    bad_json = pipeline.post("/api/chat", content="{not json", headers={"Content-Type": "application/json"})
    blank = pipeline.post("/api/chat/stream", json={"message": "   "})
    too_long = pipeline.post("/api/chat", json={"message": "x" * 2001})
    bad_id = pipeline.post("/api/chat", json={"message": "hi", "conversation_id": "not-a-uuid"})

    assert (bad_json.status_code, bad_json.json()["error"]["code"]) == (400, "INVALID_JSON")
    assert (blank.status_code, blank.json()["error"]["code"]) == (422, "INVALID_INPUT")
    assert too_long.json()["error"]["code"] == "INVALID_INPUT"
    assert bad_id.json()["error"]["message"] == "conversation_id must be a UUID."


def test_unknown_route_and_wrong_method_use_the_error_shape(pipeline):
    assert pipeline.get("/api/nope").json() == {"error": {"code": "NOT_FOUND", "message": "There is no such endpoint."}}
    assert pipeline.get("/api/chat").json()["error"]["code"] == "METHOD_NOT_ALLOWED"


def test_an_unexpected_bug_is_internal_error_json_with_cors_headers(pipeline, monkeypatch):
    def bug(*args, **kwargs):
        raise KeyError("oops")

    monkeypatch.setattr(retrieval, "search_chunks", bug)
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/chat", json={"message": "hi"}, headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "INTERNAL_ERROR", "message": "Something went wrong on the server."}}
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"  # the page can read it
    assert "oops" not in response.text


# --- timeout -------------------------------------------------------------------------------------------------------


def test_deadline_check_raises_request_timeout_only_past_the_limit():
    deadline.check()  # no limit set: nothing happens
    with deadline.limit(60):
        deadline.check()
    with deadline.limit(0.01):
        time.sleep(0.02)
        with pytest.raises(AppError) as exc:
            deadline.check()
    assert exc.value.code == "REQUEST_TIMEOUT" and exc.value.status_code == 504


def test_a_question_past_its_time_limit_ends_with_request_timeout(pipeline, monkeypatch):
    monkeypatch.setattr(get_settings(), "chat_timeout_s", 0.05)

    def slow_search(session, vector, k):
        time.sleep(0.1)
        return [CHUNK]

    monkeypatch.setattr(retrieval, "search_chunks", slow_search)

    assert_error(*both(pipeline), "REQUEST_TIMEOUT", 504)
    assert pipeline.saved == []


def test_a_reply_still_streaming_past_the_limit_is_cut_off(monkeypatch):
    class Slow:
        def stream(self, messages):
            yield AIMessageChunk(content="Half ")
            time.sleep(0.05)
            yield AIMessageChunk(content="more")

    seen = []
    with deadline.limit(0.02), pytest.raises(AppError) as exc:
        llm.stream(Slow(), [], seen.append)
    assert exc.value.code == "REQUEST_TIMEOUT" and seen == ["Half "]
