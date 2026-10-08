"""An overall time limit for answering one question (Phase 27).

Each model call already has its own limit (30 s, one retry), but one question can make several: translation, up to
three tool rounds and a final answer. So the whole pipeline gets a deadline (CHAT_TIMEOUT_S, 90 s; the slowest
measured answer took 63 s). It is checked between steps: before every model call, while a reply streams in, before
every tool and between the retrieval steps. Past it, the request ends with REQUEST_TIMEOUT. A call already waiting
on the provider cannot be interrupted, so the true worst case is the deadline plus that call's own limit.

The deadline lives in a context variable, like the usage tracker (app/usage.py): set once per question, visible to
every function that question runs, without passing it through every signature.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

from app.errors import AppError

_deadline: ContextVar[float | None] = ContextVar("movie_copilot_deadline", default=None)


@contextmanager
def limit(seconds: float) -> Iterator[None]:
    token = _deadline.set(perf_counter() + seconds)
    try:
        yield
    finally:
        _deadline.reset(token)


def check() -> None:
    """Raise REQUEST_TIMEOUT if the current question is past its deadline (no deadline set: nothing happens)."""
    deadline = _deadline.get()
    if deadline is not None and perf_counter() > deadline:
        raise AppError("REQUEST_TIMEOUT", "Answering took too long. Please try again.", 504)
