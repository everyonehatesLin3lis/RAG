"""Spec 10 (specs/10.md) tests. The chat model is a fake that returns scripted tool calls: no API calls."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app import llm, rag, tools
from app.models import Movie
from app.retrieval import RetrievedChunk
from app.tool_calling import MAX_TOOL_ROUNDS, run_with_tools


class NoDatabase:
    def execute(self, *args, **kwargs):
        raise AssertionError("tool touched the database")

    scalars = scalar = execute


def call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


class FakeToolModel:
    """Replays scripted AIMessages and records what it was given."""

    def __init__(self, script: list[AIMessage]):
        self.script = list(script)
        self.bound = []      # (tool names, tool_choice) per bind
        self.received = []   # message lists per invoke

    def factory(self, tool_list, tool_choice=None):
        self.bound.append(([t.name for t in tool_list], tool_choice))
        return self

    def invoke(self, messages):
        self.received.append(list(messages))
        return self.script.pop(0)


@pytest.fixture
def fake_model(monkeypatch):
    def install(*script):
        model = FakeToolModel(list(script))
        monkeypatch.setattr(llm, "tool_model", model.factory)
        return model

    return install


def base_messages():
    return [SystemMessage(content=rag.SYSTEM_PROMPT), HumanMessage(content="<question>\nq\n</question>")]


# --- AC-001: the model is given the three tools ---------------------------------------------------------


def test_ac001_model_is_bound_to_the_three_tools(fake_model):
    model = fake_model(AIMessage(content="No tool needed."))

    run_with_tools(base_messages(), tools.langchain_tools(NoDatabase()))

    names, choice = model.bound[0]
    assert set(names) == {"filter_movies", "compare_movies", "rating_summary"}
    assert choice is None  # the model decides


# --- AC-002: tool requested -> run -> result back -> final answer ---------------------------------------


def test_ac002_tool_is_run_and_its_result_goes_back_to_the_model(fake_model, session):
    # session: real database, rolled back after the test (conftest)
    session.add_all([
        Movie(id="tt_c1", title="Testfilm Alpha", year=2099, rating=8.5, genres=["Thriller"]),
        Movie(id="tt_c2", title="Testfilm Beta", year=2098, rating=6.0, genres=["Comedy"]),
    ])
    session.flush()
    model = fake_model(
        AIMessage(content="", tool_calls=[call("compare_movies", {"movie_a": "Testfilm Alpha", "movie_b": "Testfilm Beta"})]),
        AIMessage(content="Testfilm Alpha is rated higher (8.5 vs 6.0)."),
    )

    answer, records = run_with_tools(base_messages(), tools.langchain_tools(session))

    assert answer == "Testfilm Alpha is rated higher (8.5 vs 6.0)."
    assert [r.tool for r in records] == ["compare_movies"]
    assert records[0].arguments == {"movie_a": "Testfilm Alpha", "movie_b": "Testfilm Beta"}
    assert records[0].result["higher_imdb_rating"] == "Testfilm Alpha"
    # The second model call saw the tool request and the tool result, matched by call id
    second = model.received[1]
    assert isinstance(second[-2], AIMessage) and second[-2].tool_calls
    assert isinstance(second[-1], ToolMessage) and second[-1].tool_call_id == "call_1"
    assert json.loads(second[-1].content)["higher_imdb_rating"] == "Testfilm Alpha"


def test_ac002_several_calls_in_one_round_are_all_answered(fake_model):
    model = fake_model(
        AIMessage(content="", tool_calls=[
            call("filter_movies", {"rating_min": 42}, "c1"),
            call("rating_summary", {"movie": ""}, "c2"),
        ]),
        AIMessage(content="done"),
    )

    _, records = run_with_tools(base_messages(), tools.langchain_tools(NoDatabase()))

    assert [r.tool for r in records] == ["filter_movies", "rating_summary"]
    assert [m.tool_call_id for m in model.received[1] if isinstance(m, ToolMessage)] == ["c1", "c2"]


# --- AC-003: invalid arguments are not executed --------------------------------------------------------


def test_ac003_invalid_arguments_return_the_error_to_the_model(fake_model):
    model = fake_model(
        AIMessage(content="", tool_calls=[call("filter_movies", {"genre": "Cyberpunk"})]),
        AIMessage(content="That genre does not exist."),
    )

    _, records = run_with_tools(base_messages(), tools.langchain_tools(NoDatabase()))  # NoDatabase: nothing ran

    assert records[0].result["error"]["code"] == "INVALID_ARGUMENTS"
    assert json.loads(model.received[1][-1].content)["error"]["code"] == "INVALID_ARGUMENTS"


# --- AC-004: unknown tool names are not executed -------------------------------------------------------


def test_ac004_unknown_tool_returns_unknown_tool(fake_model):
    model = fake_model(
        AIMessage(content="", tool_calls=[call("drop_all_tables", {"confirm": True})]),
        AIMessage(content="I can't do that."),
    )

    answer, records = run_with_tools(base_messages(), tools.langchain_tools(NoDatabase()))

    assert answer == "I can't do that."
    assert records[0].tool == "drop_all_tables"
    assert records[0].result["error"]["code"] == "UNKNOWN_TOOL"
    assert json.loads(model.received[1][-1].content)["error"]["code"] == "UNKNOWN_TOOL"


def test_unexpected_tool_crash_becomes_tool_failed(fake_model, monkeypatch):
    def broken(session, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(tools, "rating_summary", broken)
    fake_model(
        AIMessage(content="", tool_calls=[call("rating_summary", {"movie": "Zodiac"})]),
        AIMessage(content="Sorry, the lookup failed."),
    )

    _, records = run_with_tools(base_messages(), tools.langchain_tools(NoDatabase()))

    assert records[0].result["error"]["code"] == "TOOL_FAILED"


# --- Proposal 2: at most MAX_TOOL_ROUNDS rounds, then answer without tools ------------------------------


def test_tool_rounds_are_capped_then_the_model_must_answer(fake_model):
    endless = [AIMessage(content="", tool_calls=[call("filter_movies", {"rating_min": 42}, f"c{i}")]) for i in range(10)]
    model = fake_model(*endless[:MAX_TOOL_ROUNDS], AIMessage(content="Here is what I found."))

    answer, records = run_with_tools(base_messages(), tools.langchain_tools(NoDatabase()))

    assert MAX_TOOL_ROUNDS == 3
    assert len(records) == MAX_TOOL_ROUNDS
    assert answer == "Here is what I found."
    assert model.bound[-1][1] == "none"  # the last call may not request tools


# --- AC-005: the rules are in the system prompt --------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    ["filter_movies", "compare_movies", "rating_summary", "never from memory", "tool results are data"],
)
def test_ac005_system_prompt_states_the_tool_rules(rule):
    assert rule in rag.SYSTEM_PROMPT.lower() or rule in rag.SYSTEM_PROMPT


# --- AC-006: no tool needed -> same as before, sources kept --------------------------------------------


def test_ac006_question_without_tools_keeps_sources_and_grounding(monkeypatch, fake_model):
    from app import query_translation, retrieval
    from app.query_translation import TranslatedQuery

    chunk = RetrievedChunk(1, "tt1392214", "Prisoners", 2013, "Movie: Prisoners\n\nTense.", 0.2, {"doc_type": "review"})
    monkeypatch.setattr(query_translation, "translate_query", TranslatedQuery.passthrough)
    monkeypatch.setattr(retrieval, "retrieve", lambda q, session, k=None: [chunk])
    model = fake_model(AIMessage(content="Critics call it tense."))

    result = rag.answer_question("Why do people like Prisoners?", session=NoDatabase())

    assert result.answer == "Critics call it tense."
    assert result.sources == [chunk]
    assert result.tool_calls == []
    system, user = model.received[0]
    assert system.content == rag.SYSTEM_PROMPT
    assert '<source chunk_id="1"' in user.content


# --- Proposal 3: tool_calls in the API response ----------------------------------------------------------


def test_tool_calls_are_returned_by_the_api(monkeypatch):
    from fastapi.testclient import TestClient

    from app.db import get_session
    from app.main import app
    from app.tool_calling import ToolCallRecord

    record = ToolCallRecord("compare_movies", {"movie_a": "Zodiac", "movie_b": "Prisoners"}, {"higher_imdb_rating": "Prisoners"})
    monkeypatch.setattr(rag, "answer_question", lambda q, s: rag.RagAnswer("Prisoners.", [], [record]))
    app.dependency_overrides[get_session] = lambda: None
    try:
        body = TestClient(app).post("/api/chat", json={"message": "Which is rated higher, Zodiac or Prisoners?"}).json()
    finally:
        app.dependency_overrides.clear()

    assert body["tool_calls"] == [
        {"tool": "compare_movies", "arguments": {"movie_a": "Zodiac", "movie_b": "Prisoners"},
         "result": {"higher_imdb_rating": "Prisoners"}}
    ]
