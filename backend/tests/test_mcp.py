"""Phase 24: the MCP server on its own, over a real stdio subprocess, and connected to the LangChain tool loop.

The server is tested the way the plan asks (24.2): MCP client -> MCP tool -> database result, without the chatbot.
Database tests skip when the local database is down (the `session` fixture), like the rest of the suite.
"""

import json
import sys

import anyio
import pytest
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from app import llm, mcp_client, mcp_server, rag, tools
from app.config import get_settings
from app.errors import AppError
from app.mcp_client import McpToolClient
from app.tool_calling import execute, run_with_tools

TOOL_NAMES = {"filter_movies", "compare_movies", "rating_summary", "get_movie_metadata"}


def over_mcp(use_client):
    """Run `await use_client(client)` against the server in-process, with the official MCP client."""

    async def main():
        async with Client(mcp_server.server) as client:
            return await use_client(client)

    return anyio.run(main)


# --- the server on its own --------------------------------------------------------------------------------


def test_discovery_lists_the_four_tools_with_their_pydantic_schemas():
    listed = over_mcp(lambda c: c.list_tools()).tools

    assert {t.name for t in listed} == TOOL_NAMES
    for t in listed:
        _, schema = tools.TOOLS[t.name]
        assert t.input_schema == schema.model_json_schema()
        assert t.description == tools.DESCRIPTIONS[t.name]


def test_server_validates_arguments_itself():
    # Any MCP client can call the server, so it cannot rely on the client having validated anything.
    result = over_mcp(lambda c: c.call_tool("filter_movies", {"rating_min": 42}))

    assert result.is_error
    assert result.structured_content["error"]["code"] == "INVALID_ARGUMENTS"
    assert json.loads(result.content[0].text) == result.structured_content


def test_unknown_tool_is_a_structured_error():
    result = over_mcp(lambda c: c.call_tool("drop_tables", {}))

    assert result.is_error
    assert result.structured_content["error"]["code"] == "UNKNOWN_TOOL"


def test_each_tool_returns_the_same_result_as_the_function(session):
    calls = {
        "filter_movies": {"genre": "war", "year_min": 2010},
        "compare_movies": {"movie_a": "Zodiac", "movie_b": "Prisoners"},
        "rating_summary": {"movie": "Prisoners"},
        "get_movie_metadata": {"movie": "Prisoners"},
    }

    async def call_all(client):
        return {name: await client.call_tool(name, args) for name, args in calls.items()}

    results = over_mcp(call_all)

    for name, args in calls.items():
        assert not results[name].is_error
        assert results[name].structured_content == tools.TOOLS[name][0](session, **args)
    assert results["get_movie_metadata"].structured_content["director"] == "Denis Villeneuve"


def test_title_errors_come_back_through_mcp(session):
    result = over_mcp(lambda c: c.call_tool("rating_summary", {"movie": "Beauty and the Beast"}))

    assert result.is_error
    assert result.structured_content["error"]["code"] == "AMBIGUOUS_TITLE"
    assert len(result.structured_content["error"]["candidates"]) == 2


# --- the real thing: a child process over stdio -------------------------------------------------------------


def test_stdio_subprocess_client_tool_database_result(session):
    client = McpToolClient()  # starts `python -m app.mcp_server`
    try:
        assert {t.name for t in client.langchain_tools()} == TOOL_NAMES
        result = client.call_tool("get_movie_metadata", {"movie": "Zodiac"})
        assert result["title"] == "Zodiac" and result["director"] == "David Fincher"
    finally:
        client.close()


def test_the_server_process_gets_no_api_keys(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@127.0.0.1:5433/db")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-should-not-pass")

    params = mcp_client.server_parameters()

    assert params.env == {"DATABASE_URL": "postgresql+psycopg://u:p@127.0.0.1:5433/db"}
    assert params.args == ["-m", "app.mcp_server"]


def test_a_server_that_cannot_start_is_mcp_unavailable():
    broken = McpToolClient(server=StdioServerParameters(command=sys.executable, args=["-c", "raise SystemExit(1)"]),
                           call_timeout_s=5)

    with pytest.raises(AppError) as exc:
        broken.connect()
    assert exc.value.code == "MCP_UNAVAILABLE"
    # A tool call never raises: the model gets an error it can read and answers without the tool.
    assert broken.call_tool("rating_summary", {"movie": "Zodiac"})["error"]["code"] == "MCP_UNAVAILABLE"


# --- connected to LangChain -------------------------------------------------------------------------------


def test_the_model_sees_identical_tool_definitions_through_mcp_and_locally(in_process_mcp):
    remote = {t.name: convert_to_openai_tool(t) for t in in_process_mcp.langchain_tools()}
    local = {t.name: convert_to_openai_tool(t) for t in tools.langchain_tools(None)}

    assert remote == local


def test_tool_loop_runs_mcp_tools_and_returns_their_errors(in_process_mcp):
    by_name = {t.name: t for t in in_process_mcp.langchain_tools()}

    result = execute({"name": "filter_movies", "args": {"genre": "Cyberpunk"}, "id": "c1"}, by_name)

    assert result["error"]["code"] == "INVALID_ARGUMENTS"


def test_tool_loop_end_to_end_through_mcp(monkeypatch, in_process_mcp, session):
    replies = iter([
        AIMessage(content="", tool_calls=[{"name": "get_movie_metadata", "args": {"movie": "Prisoners"}, "id": "c1"}]),
        AIMessage(content="Denis Villeneuve directed Prisoners."),
    ])
    seen = []
    monkeypatch.setattr(llm, "tool_model", lambda tool_list, tool_choice=None: "model")
    monkeypatch.setattr(llm, "invoke", lambda model, messages: seen.append(list(messages)) or next(replies))

    answer, records = run_with_tools([], in_process_mcp.langchain_tools())

    assert answer == "Denis Villeneuve directed Prisoners."
    assert records[0].tool == "get_movie_metadata" and records[0].result["director"] == "Denis Villeneuve"
    assert json.loads(seen[1][-1].content)["director"] == "Denis Villeneuve"  # the ToolMessage the model received


@pytest.mark.parametrize("backend", ["mcp", "local"])
def test_pipeline_uses_the_configured_tool_backend(monkeypatch, backend):
    from app import embeddings, query_translation, retrieval, tool_calling
    from app.query_translation import TranslatedQuery
    from app.retrieval import RetrievedChunk

    given = []
    monkeypatch.setattr(get_settings(), "tool_backend", backend)
    monkeypatch.setattr(query_translation, "translate_query", lambda m, h=None: TranslatedQuery.passthrough(m))
    monkeypatch.setattr(embeddings, "embed_query", lambda text: [0.1] * 1536)
    monkeypatch.setattr(retrieval, "search_chunks", lambda s, v, k: [
        RetrievedChunk(1, "tt1", "Prisoners", 2013, "Movie: Prisoners\n\nTense.", 0.2, {})])
    monkeypatch.setattr(retrieval, "keyword_search", lambda s, kw, k=None: [])
    monkeypatch.setattr(tool_calling, "run_with_tools", lambda m, tool_list: given.append(tool_list) or ("ok", []))

    result = rag.answer_question("Who directed Prisoners?", session=None)

    assert {t.name for t in given[0]} == TOOL_NAMES
    assert result.debug.tool_backend == backend
    # MCP tools carry the server's JSON schema (a dict); local ones the Pydantic class.
    assert isinstance(given[0][0].args_schema, dict) == (backend == "mcp")


# Implements: specs/28.md#AC-010 (every structured error code comes back through MCP marked as an error)
def test_movie_not_found_comes_back_through_mcp_with_suggestions(session):
    result = over_mcp(lambda c: c.call_tool("get_movie_metadata", {"movie": "Dark Knight"}))

    assert result.is_error
    assert result.structured_content["error"]["code"] == "MOVIE_NOT_FOUND"
    assert {"title": "The Dark Knight", "year": 2008} in result.structured_content["error"]["suggestions"]
