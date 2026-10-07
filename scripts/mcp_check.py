"""Phase 24.2: test the MCP server on its own, without the chatbot. No LLM, no API calls, no cost.

Starts the server as a child process over stdio (exactly as the backend does), lists its tools (discovery), then
calls each one with real arguments plus three bad calls, and prints what came back.

    python scripts/mcp_check.py            # from the repo root, backend venv active, database running

The same server also works with any other MCP client, e.g. the MCP Inspector:
    npx @modelcontextprotocol/inspector <repo>/backend/.venv/Scripts/python.exe -m app.mcp_server   (cwd: backend/)
"""

import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.mcp_client import McpToolClient  # noqa: E402

CALLS = [
    ("filter_movies", {"genre": "thriller", "year_min": 2010, "rating_min": 7.5, "limit": 3}),
    ("compare_movies", {"movie_a": "Zodiac", "movie_b": "Prisoners"}),
    ("rating_summary", {"movie": "Hereditary"}),
    ("get_movie_metadata", {"movie": "Dune (2021)"}),
    # Error paths: every one must come back as a structured error, not a crash.
    ("rating_summary", {"movie": "Beauty and the Beast"}),        # AMBIGUOUS_TITLE
    ("compare_movies", {"movie_a": "Zodiac", "movie_b": "Zodak"}),  # MOVIE_NOT_FOUND
    ("filter_movies", {"rating_min": 42}),                          # INVALID_ARGUMENTS
    ("drop_tables", {}),                                            # UNKNOWN_TOOL
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # cast names like Chalamet's
    client = McpToolClient()
    started = perf_counter()
    client.connect()
    print(f"server started and tools discovered in {perf_counter() - started:.2f}s:")
    for tool in client.tools:
        print(f"  {tool.name}({', '.join(tool.input_schema.get('properties', {}))})")

    for name, arguments in CALLS:
        started = perf_counter()
        result = client.call_tool(name, arguments)
        ms = (perf_counter() - started) * 1000
        status = result["error"]["code"] if "error" in result else "ok"
        print(f"\n{name}({json.dumps(arguments)})  {status}  {ms:.0f} ms")
        print("  " + json.dumps(result, ensure_ascii=False)[:300])
    client.close()


if __name__ == "__main__":
    main()
