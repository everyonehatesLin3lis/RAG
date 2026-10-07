"""The MCP client (Phase 24): connects the chat pipeline to the MCP server and turns its tools into LangChain tools.

Before Phase 24:  LangChain -> local Python tool -> PostgreSQL
Now:              LangChain -> MCP client (this file) -> MCP server (app/mcp_server.py, a child process) -> PostgreSQL

The tool loop in app/tool_calling.py does not change. It still gets LangChain tools; only what happens inside them
differs: instead of calling a Python function, each tool sends a `tools/call` message to the server and waits.

Three details:
1. Discovery. The client does not import the tools. It asks the server (`tools/list`) and builds one LangChain tool
   per entry from the server's name, description and JSON schema. A tool added to the server appears here by itself.
2. One server for the whole backend. Starting it costs about a second (a Python process importing SQLAlchemy), so it
   is started once and reused. If it dies, the next call starts a new one.
3. Sync on the outside, async inside. The MCP SDK is async; our pipeline is plain synchronous code. The connection
   lives on an asyncio event loop in a background thread, and each call is handed to that loop and waited for.

The server only gets DATABASE_URL (when set) on top of the SDK's minimal environment: no API keys.
"""

import asyncio
import json
import logging
import os
import sys
import threading
from concurrent.futures import Future
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from app.config import get_settings
from app.errors import AppError

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[1]
START_TIMEOUT_S = 30


def server_parameters() -> StdioServerParameters:
    """How to start the server: this same Python, `python -m app.mcp_server`, from backend/."""
    env = {"DATABASE_URL": os.environ["DATABASE_URL"]} if "DATABASE_URL" in os.environ else {}
    return StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"], cwd=str(BACKEND_DIR), env=env)


def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


class McpToolClient:
    """One connection to the MCP server, usable from synchronous code.

    `server` is anything mcp.Client accepts: StdioServerParameters (a subprocess, the normal case) or, in tests,
    the server object itself (in-process, same protocol messages, no subprocess).
    """

    def __init__(self, server: Any = None, call_timeout_s: float | None = None):
        self._server = server if server is not None else server_parameters()
        self._call_timeout_s = call_timeout_s or get_settings().mcp_call_timeout_s
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, name="mcp-client", daemon=True).start()
        self._lock = threading.Lock()
        self._client: Client | None = None
        self._connection: Future | None = None  # the task that keeps the connection open
        self._stop: asyncio.Event | None = None
        self.tools: list = []  # the server's tool list, from discovery

    # --- connection ---------------------------------------------------------------------------------------

    def _connected(self) -> bool:
        return self._connection is not None and not self._connection.done() and self._client is not None

    def connect(self) -> None:
        """Start the server and discover its tools, unless already connected. Raises AppError(MCP_UNAVAILABLE)."""
        with self._lock:
            if self._connected():
                return
            ready: Future = Future()

            async def hold_open() -> None:
                # The connection is opened and closed inside this one task (the SDK requires that); calls from
                # other tasks use it in between. It stays open until close() sets the stop event.
                self._stop = asyncio.Event()
                try:
                    async with Client(self._server, read_timeout_seconds=self._call_timeout_s) as client:
                        listed = await client.list_tools()
                        self._client = client
                        ready.set_result(listed.tools)
                        await self._stop.wait()
                except BaseException as exc:
                    if not ready.done():
                        ready.set_exception(exc)
                    raise
                finally:
                    self._client = None

            self._connection = asyncio.run_coroutine_threadsafe(hold_open(), self._loop)
            try:
                self.tools = ready.result(START_TIMEOUT_S)
            except Exception as exc:
                self._connection.cancel()
                logger.warning("MCP server could not be started: %r", exc)
                raise AppError("MCP_UNAVAILABLE", "The movie tools are unavailable right now.", 503) from exc
            logger.info("MCP server connected, tools: %s", ", ".join(t.name for t in self.tools))

    def close(self) -> None:
        """Close the connection; the server process exits when its stdin closes."""
        with self._lock:
            if self._connection is not None and not self._connection.done() and self._stop is not None:
                self._loop.call_soon_threadsafe(self._stop.set)
                try:
                    self._connection.result(5)
                except Exception:
                    self._connection.cancel()
            self._connection, self._client = None, None

    # --- calls ----------------------------------------------------------------------------------------------

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Call one tool on the server. Never raises: a broken connection becomes an MCP_UNAVAILABLE result."""
        try:
            self.connect()
            future = asyncio.run_coroutine_threadsafe(self._client.call_tool(name, arguments), self._loop)
            result = future.result(self._call_timeout_s + 5)
        except Exception as exc:
            logger.warning("MCP call %s failed: %r", name, exc)
            self.close()  # the next call starts a fresh server
            return _error("MCP_UNAVAILABLE", f"The {name} tool could not be reached. Answer without it.")

        # Our server returns the result dict as structured content and as JSON text; use whichever is there.
        if isinstance(result.structured_content, dict):
            return result.structured_content
        try:
            return json.loads(result.content[0].text)
        except (IndexError, AttributeError, ValueError):
            return _error("TOOL_FAILED", f"The {name} tool returned an unreadable result.")

    def langchain_tools(self) -> list[StructuredTool]:
        """One LangChain tool per tool the server lists. The JSON schema is passed as is: the model sees the server's
        schema, and the server (not LangChain) validates the arguments and returns INVALID_ARGUMENTS when they are wrong."""
        self.connect()

        def make(name: str):
            return lambda **kwargs: json.dumps(self.call_tool(name, kwargs), ensure_ascii=False)

        return [
            StructuredTool(name=t.name, description=t.description or "", args_schema=t.input_schema, func=make(t.name))
            for t in self.tools
        ]


@lru_cache
def get_mcp_client() -> McpToolClient:
    """The backend's single MCP client (and so its single server process)."""
    return McpToolClient()
