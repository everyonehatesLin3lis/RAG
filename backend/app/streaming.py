"""Streaming over Server-Sent Events (Phase 25): the answer appears on the page while it is being written.

SSE is a plain HTTP response that stays open; the server writes one event at a time, each as a `data: <json>` line
followed by a blank line, and the browser reads them as they arrive. Events, in order:

    {"type": "status", "stage": "search", "message": "Searching reviews and movie data"}   progress before text
    {"type": "tool_call", "data": {"tool", "arguments", "result"}}                         as each tool finishes
    {"type": "token", "content": "Prisoners"}                                              each piece of answer text
    {"type": "sources", "data": [...]}                                                     the plan's event
    {"type": "metadata", "data": {"answer", "conversation_id", "tool_calls", "debug", "usage"}}
    {"type": "done"}                                                                       the plan's event
or, if something fails after the stream has started (the HTTP status is already 200 by then):
    {"type": "error", "error": {"code": "LLM_UNAVAILABLE", "message": "..."}}

The pipeline is ordinary blocking code, so it runs in a worker thread and hands each event to this async generator
through a queue. When the browser goes away, the generator is closed; the worker's next event then raises
ClientDisconnected, which stops the pipeline (no more tokens paid for) and stores nothing.
"""

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Callable

import anyio
from sqlalchemy.exc import SQLAlchemyError

from app.errors import AppError, ClientDisconnected
from app.schemas import ChatResponse

Emit = Callable[[dict], None]

_END = object()


def sse(event: dict) -> str:
    """One Server-Sent Event: a data line with the JSON, then a blank line."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def error_event(code: str, message: str) -> dict:
    return {"type": "error", "error": {"code": code, "message": message}}


def final_events(response: ChatResponse) -> list[dict]:
    """After the last token: the sources (the plan's event), everything else /api/chat returns, then done."""
    data = response.model_dump()
    return [
        {"type": "sources", "data": data["sources"]},
        {"type": "metadata", "data": {k: data[k] for k in ("answer", "conversation_id", "tool_calls", "debug", "usage")}},
        {"type": "done"},
    ]


async def chat_events(run: Callable[[Emit], ChatResponse]) -> AsyncIterator[str]:
    """Run `run(emit)` in a worker thread and yield every event it emits, then the final ones, as SSE."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    closed = threading.Event()  # the browser went away
    finished = threading.Event()  # the worker has stopped

    def put(event) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def emit(event: dict) -> None:
        if closed.is_set():
            raise ClientDisconnected()
        put(event)

    def work() -> None:
        try:
            for event in final_events(run(emit)):
                put(event)
        except ClientDisconnected:
            pass
        except AppError as exc:
            put(error_event(exc.code, exc.message))
        except SQLAlchemyError:
            put(error_event("DATABASE_UNAVAILABLE", "The database is unavailable right now."))
        except Exception:
            put(error_event("INTERNAL_ERROR", "Something went wrong while answering."))
        finally:
            finished.set()
            put(_END)

    threading.Thread(target=work, name="chat-stream", daemon=True).start()
    try:
        while (event := await queue.get()) is not _END:
            yield sse(event)
    finally:
        closed.set()
        # The request's database session is closed when this response ends, so wait until the worker no longer
        # uses it (it stops at its next event). Shielded: this also runs when the response is being cancelled.
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(finished.wait, 60)
