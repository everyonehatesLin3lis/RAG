"""Rate limits for the public deployment: every question costs OpenRouter credit, so nobody may ask too many.

Two limits, both off by default (local development, tests and evaluation runs never meet them):

- RATE_LIMIT_PER_MINUTE: questions per client IP in any 60 seconds. Stops one visitor, or one script, from
  asking hundreds of questions in a row.
- RATE_LIMIT_PER_DAY: questions from everyone together per UTC day. This one does not depend on telling clients
  apart, so it still caps the bill if the IP check is fooled.

The counts live in memory, per backend instance: simple, no extra service, but they reset when Cloud Run starts a
new instance, and with max 2 instances the real caps are up to twice the setting. That is fine for a demo; the hard
ceiling is the credit limit on the OpenRouter key itself (docs/deployment.md).

Which IP? Behind Cloud Run every request comes from Google's front end, so the client address is in the
X-Forwarded-For header. A client can send that header itself, with any addresses it likes; the proxy then appends
the address it really saw. So we take the rightmost entry, the one the client cannot write.
"""

import threading
import time
from collections import deque
from datetime import UTC, datetime

from fastapi import Request

from app.config import get_settings
from app.errors import AppError

WINDOW_S = 60

_lock = threading.Lock()
_recent: dict[str, deque] = {}  # client IP -> times of its questions in the last minute
_day = {"date": "", "count": 0}


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def check(request: Request) -> None:
    """FastAPI dependency on the endpoints that ask the model. Raises RATE_LIMITED (429) before any work starts,
    so a streamed answer is refused with a normal JSON error, not halfway through the stream."""
    settings = get_settings()
    per_minute, per_day = settings.rate_limit_per_minute, settings.rate_limit_per_day
    if not per_minute and not per_day:
        return
    allow(client_ip(request), per_minute, per_day, now=time.monotonic(), today=datetime.now(UTC).date().isoformat())


def allow(ip: str, per_minute: int, per_day: int, now: float, today: str) -> None:
    """Count one question, or raise if it would go over a limit. A refused question is not counted."""
    with _lock:
        if _day["date"] != today:
            _day.update(date=today, count=0)
        if per_day and _day["count"] >= per_day:
            raise AppError("RATE_LIMITED", "The demo has reached its question limit for today. Please try again tomorrow.", 429)

        times = _recent.setdefault(ip, deque())
        while times and now - times[0] >= WINDOW_S:
            times.popleft()
        if per_minute and len(times) >= per_minute:
            raise AppError("RATE_LIMITED", "Too many questions in a short time. Please wait a minute and try again.", 429)

        times.append(now)
        _day["count"] += 1
        if len(_recent) > 10_000:  # forget clients that have gone quiet, so memory stays small
            for key in [k for k, t in _recent.items() if not t or now - t[-1] >= WINDOW_S]:
                del _recent[key]


def reset() -> None:
    """Tests start from empty counts."""
    with _lock:
        _recent.clear()
        _day.update(date="", count=0)
