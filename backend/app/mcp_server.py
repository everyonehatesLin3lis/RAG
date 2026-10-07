"""The MCP server (Phase 24): the movie tools behind the Model Context Protocol.

Tool calling (Phase 10) is the model asking *our code* to run a function. MCP is a standard way to put those
functions in a separate program, a server, so that any MCP client can discover and call them: our backend,
Claude Desktop, the MCP Inspector. The protocol is JSON-RPC: the client asks `tools/list` (names, descriptions,
input schemas) and then `tools/call` (a name and arguments); the server runs the tool and returns the result.

Here the transport is stdio: the client starts this file as a child process and they exchange JSON messages over
its stdin and stdout. So nothing may ever be printed to stdout here; logs go to stderr.

The server advertises exactly the Pydantic schemas in app/tools.py and runs exactly the same functions, so the
model sees the same tools as before Phase 24. The server is the trust boundary: it validates every call itself
(any client could send anything) and returns our structured errors (INVALID_ARGUMENTS, MOVIE_NOT_FOUND, ...) as
the result, marked `is_error`, rather than crashing.

Run on its own (from backend/):  python -m app.mcp_server
"""

import json

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from sqlalchemy.orm import Session

from app.db import get_engine
from app.tools import DESCRIPTIONS, TOOLS, run_tool

SERVER_NAME = "movie-research-copilot"


async def list_tools(ctx, params) -> types.ListToolsResult:
    """Tool discovery: name, description and JSON input schema of every tool."""
    return types.ListToolsResult(tools=[
        types.Tool(name=name, description=DESCRIPTIONS[name], input_schema=schema.model_json_schema())
        for name, (_, schema) in TOOLS.items()
    ])


def call_tool_sync(name: str, arguments: dict) -> dict:
    """One database session per call, read-only (nothing is committed)."""
    with Session(get_engine()) as session:
        return run_tool(session, name, arguments)


async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
    # The tools use blocking database calls; run them in a worker thread so the server keeps reading messages.
    result = await anyio.to_thread.run_sync(call_tool_sync, params.name, dict(params.arguments or {}))
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))],
        structured_content=result,
        is_error="error" in result,
    )


server = Server(
    SERVER_NAME,
    version="1.0",
    instructions="Movie database tools: filter, compare, rating summaries and metadata for the movies in the dataset.",
    on_list_tools=list_tools,
    on_call_tool=call_tool,
)


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(main)
