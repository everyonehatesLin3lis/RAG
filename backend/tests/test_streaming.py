"""Phase 25: streaming over SSE. Fake models and a fake pipeline: no API calls, no database."""

import json
import threading

import anyio
import openai
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app import llm, rag, request_log, streaming
from app.errors import AppError, ClientDisconnected
from app.retrieval import RetrievedChunk
from app.tool_calling import ToolCallRecord, run_with_tools
from app.usage import track_usage

CONVERSATION_ID = "5f0c1a52-8d2b-4c1e-9a77-0b6f4f5e2c11"
CHUNK = RetrievedChunk(2281, "tt1392214", "Prisoners", 2013, "Movie: Prisoners\n\nReview by A (B), 4/5:\nGrim.", 0.3,
                       {"doc_type": "review", "review_id": 1822, "source": "rotten_tomatoes"})


def parse(body: str) -> list[dict]:
    """SSE text -> the list of JSON events."""
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


# --- keeping OpenRouter's cost when streaming -------------------------------------------------------------


def test_streamed_usage_chunk_keeps_openrouter_cost_and_model():
    model = llm.OpenRouterChat(model="xiaomi/mimo-v2.6-flash", api_key="sk-test", base_url="http://127.0.0.1:1")
    usage_chunk = {
        "model": "xiaomi/mimo-v2.6-flash",
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cost": 0.00042},
    }

    generation = model._convert_chunk_to_generation_chunk(usage_chunk, AIMessageChunk, {})

    assert generation.message.response_metadata["token_usage"]["cost"] == 0.00042
    assert generation.message.response_metadata["usage_model"] == "xiaomi/mimo-v2.6-flash"
    assert generation.message.usage_metadata["input_tokens"] == 100


def test_tracker_uses_the_usage_model_not_the_doubled_merged_name():
    message = AIMessage(content="hi", usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                        response_metadata={"model_name": "x/yx/y", "usage_model": "x/y", "token_usage": {"cost": 0.001}})
    with track_usage() as tracker:
        from app.usage import _current

        _current.get().on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))

    assert list(tracker.models) == ["x/y"]
    assert tracker.models["x/y"].cost_usd == 0.001 and tracker.models["x/y"].cost_source == "reported"


# --- llm.stream -------------------------------------------------------------------------------------------


class ScriptedStream:
    """A model whose stream() replays scripted rounds of chunks; also usable as llm.tool_model's result."""

    def __init__(self, rounds: list[list[AIMessageChunk]]):
        self.rounds = list(rounds)
        self.tool_choices = []

    def factory(self, tool_list, tool_choice=None):
        self.tool_choices.append(tool_choice)
        return self

    def stream(self, messages):
        yield from self.rounds.pop(0)


def test_stream_passes_text_on_and_returns_the_whole_reply():
    pieces = []
    model = ScriptedStream([[AIMessageChunk(content="Pris"), AIMessageChunk(content="oners."),
                             AIMessageChunk(content="", usage_metadata={"input_tokens": 5, "output_tokens": 2,
                                                                        "total_tokens": 7})]])

    reply = llm.stream(model, [], pieces.append)

    assert pieces == ["Pris", "oners."]
    assert reply.text == "Prisoners." and reply.usage_metadata["total_tokens"] == 7


def test_stream_maps_provider_errors():
    class Broken:
        def stream(self, messages):
            yield AIMessageChunk(content="Half")
            raise openai.APITimeoutError(request=None)

    with pytest.raises(AppError) as exc:
        llm.stream(Broken(), [], lambda text: None)
    assert exc.value.code == "LLM_TIMEOUT"


# --- the tool loop, streaming ------------------------------------------------------------------------------


def tool_call_chunk(name: str, args: dict, call_id: str = "c1") -> AIMessageChunk:
    return AIMessageChunk(content="", tool_call_chunks=[{"name": name, "args": json.dumps(args), "id": call_id, "index": 0}])


def test_tool_loop_streams_tokens_tool_progress_and_results(monkeypatch, in_process_mcp):
    model = ScriptedStream([
        [AIMessageChunk(content="Let me check."), tool_call_chunk("filter_movies", {"rating_min": 42})],
        [AIMessageChunk(content="Prisoners "), AIMessageChunk(content="is rated higher.")],
    ])
    monkeypatch.setattr(llm, "tool_model", model.factory)
    events = []

    answer, records = run_with_tools([HumanMessage(content="q")], in_process_mcp.langchain_tools(), on_event=events.append)

    assert [e["type"] for e in events] == ["token", "status", "tool_call", "status", "token", "token", "token"]
    assert events[1]["message"] == "Calling filter_movies"
    assert events[2]["data"]["result"]["error"]["code"] == "INVALID_ARGUMENTS"  # rejected by the MCP server
    assert events[4]["content"] == "\n\n"  # the earlier sentence and the answer stay separate paragraphs
    assert answer == "Let me check.\n\nPrisoners is rated higher."
    assert records[0].tool == "filter_movies"


def test_without_on_event_the_loop_does_not_stream(monkeypatch):
    class InvokeOnly:
        def factory(self, tool_list, tool_choice=None):
            return self

        def invoke(self, messages):
            return AIMessage(content="Plain answer.")

        def stream(self, messages):
            raise AssertionError("/api/chat must not stream")

    monkeypatch.setattr(llm, "tool_model", InvokeOnly().factory)
    assert run_with_tools([HumanMessage(content="q")], [])[0] == "Plain answer."


# --- the endpoint ------------------------------------------------------------------------------------------


def streaming_pipeline(monkeypatch, fail_after_token: AppError | None = None):
    def fake_answer_question(question, session, history=None, on_event=None):
        on_event({"type": "status", "stage": "search", "message": "Searching reviews and movie data"})
        on_event({"type": "tool_call", "data": {"tool": "rating_summary", "arguments": {"movie": "Prisoners"},
                                                "result": {"average_critic_rating": 7.2}}})
        on_event({"type": "token", "content": "Critics "})
        if fail_after_token:
            raise fail_after_token
        on_event({"type": "token", "content": "like it."})
        return rag.RagAnswer(answer="Critics like it.", sources=[CHUNK], tool_calls=[
            ToolCallRecord("rating_summary", {"movie": "Prisoners"}, {"average_critic_rating": 7.2})])

    monkeypatch.setattr(rag, "answer_question", fake_answer_question)


def test_stream_sends_progress_tokens_then_the_plans_sources_and_done(monkeypatch, api_without_database):
    streaming_pipeline(monkeypatch)

    response = api_without_database.post("/api/chat/stream", json={"message": "Why?", "conversation_id": CONVERSATION_ID})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse(response.text)
    assert [e["type"] for e in events] == ["status", "tool_call", "token", "token", "sources", "metadata", "done"]
    assert "".join(e["content"] for e in events if e["type"] == "token") == "Critics like it."
    assert events[4]["data"][0]["chunk_id"] == "2281"
    meta = events[5]["data"]
    assert meta["answer"] == "Critics like it." and meta["conversation_id"] == CONVERSATION_ID
    assert meta["tool_calls"][0]["tool"] == "rating_summary"
    assert [(str(c), q, a) for c, q, a in api_without_database.saved] == [(CONVERSATION_ID, "Why?", "Critics like it.")]  # stored once complete


def test_failure_after_the_stream_started_is_an_error_event_and_nothing_is_stored(monkeypatch, api_without_database,
                                                                                   temporary_request_log):
    streaming_pipeline(monkeypatch, fail_after_token=AppError("LLM_UNAVAILABLE", "The language model is unavailable.", 502))

    events = parse(api_without_database.post("/api/chat/stream", json={"message": "Why?"}).text)

    assert events[-1] == {"type": "error", "error": {"code": "LLM_UNAVAILABLE", "message": "The language model is unavailable."}}
    assert "done" not in [e["type"] for e in events]
    assert api_without_database.saved == []
    assert request_log.read(temporary_request_log)[-1]["error_code"] == "LLM_UNAVAILABLE"


def test_invalid_input_is_rejected_before_streaming(api_without_database):
    response = api_without_database.post("/api/chat/stream", json={"message": ""})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"


# --- the browser going away --------------------------------------------------------------------------------


def test_closing_the_stream_stops_the_pipeline_at_its_next_event():
    first_sent, outcome = threading.Event(), {}

    def run(emit):
        emit({"type": "token", "content": "Half"})
        first_sent.wait(5)
        try:
            emit({"type": "token", "content": " more"})
            outcome["stopped"] = False
        except ClientDisconnected:
            outcome["stopped"] = True
            raise

    async def read_one_then_leave():
        events = streaming.chat_events(run)
        first = await events.__anext__()
        await events.aclose()  # what Starlette does when the browser disconnects
        return first

    async def main():
        with anyio.fail_after(10):
            async with anyio.create_task_group() as tg:
                result = {}

                async def consume():
                    result["first"] = await read_one_then_leave()

                tg.start_soon(consume)
                await anyio.sleep(0.2)
                first_sent.set()
        return result["first"]

    first = anyio.run(main)

    assert json.loads(first[6:]) == {"type": "token", "content": "Half"}
    assert outcome == {"stopped": True}


def test_a_disconnected_answer_is_logged_and_not_stored(monkeypatch, api_without_database, temporary_request_log):
    from app import main

    def gone(question, session, history=None, on_event=None):
        raise ClientDisconnected()

    monkeypatch.setattr(rag, "answer_question", gone)

    with pytest.raises(ClientDisconnected):
        main.answer_in_conversation(api_without_database.fake_session, None, "Why?", on_event=lambda e: None)

    assert api_without_database.saved == []
    assert request_log.read(temporary_request_log)[-1]["status"] == "disconnected"


def test_sse_format_is_one_data_line_and_a_blank_line():
    assert streaming.sse({"type": "done"}) == 'data: {"type": "done"}\n\n'
