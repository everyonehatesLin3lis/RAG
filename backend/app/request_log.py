"""Logging and monitoring (Phase 15): one JSON line per chat request, and a summary over the log.

JSON Lines (one JSON object per line) is easy to append to, easy to read back line by line, and every line can be
parsed on its own, so a crash mid-write damages at most one line. The fields are the plan's:

    timestamp, conversation_id, query, translated_query, retrieved_chunks, tools, tokens, cost, latency_ms, status

plus: error_code (on failure), tool_errors, translation_origin, and tokens/cost per model.

What is never logged: API keys, database URLs, the retrieved texts themselves, or the answer. The question is
logged because the plan asks for it; on a shared deployment that would need a privacy notice or redaction.
"""

import json
import statistics
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings

_lock = threading.Lock()


def log_path() -> Path:
    return Path(get_settings().request_log_path)


def write(entry: dict) -> None:
    """Append one line. A logging failure must never break the chat, so errors here are swallowed."""
    entry = {"timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"), **entry}
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with _lock, path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def success_entry(conversation_id: str, query: str, result, latency_ms: int) -> dict:
    """Build the log line for a request that produced an answer (result is a rag.RagAnswer)."""
    debug = result.debug
    usage = result.usage
    return {
        "conversation_id": conversation_id,
        "query": query,
        "translated_query": debug.translation.semantic_query if debug else None,
        "translation_origin": debug.translation.origin if debug else None,
        "retrieved_chunks": len(result.sources),
        "tools": [c.tool for c in result.tool_calls],
        "tool_errors": [c.result["error"]["code"] for c in result.tool_calls if isinstance(c.result.get("error"), dict)],
        "tokens": sum(m.total_tokens for m in usage),
        "cost": round(sum(m.cost_usd or 0.0 for m in usage), 8),
        "models": {m.model: {"tokens": m.total_tokens, "cost": m.cost_usd} for m in usage},
        "latency_ms": latency_ms,
        # Phase 25: how long the user waited for the first text of a streamed answer (absent for /api/chat)
        "first_token_ms": debug.timings_ms.get("first_token") if debug else None,
        "status": "success" if result.sources else "no_results",
    }


def error_entry(conversation_id: str | None, query: str, code: str, latency_ms: int) -> dict:
    return {
        "conversation_id": conversation_id,
        "query": query,
        "latency_ms": latency_ms,
        "status": "error",
        "error_code": code,
    }


def disconnected_entry(conversation_id: str | None, query: str, latency_ms: int) -> dict:
    """Phase 25: the user closed a streamed answer before it finished. Not an error of ours, so its own status."""
    return {"conversation_id": conversation_id, "query": query, "latency_ms": latency_ms, "status": "disconnected"}


# --- monitoring: summarise the log ------------------------------------------------------------------


def read(path: Path | None = None) -> list[dict]:
    path = path or log_path()
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a damaged line is skipped, not fatal
    return entries


def _percentile(values: list[int], p: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


def summarise(entries: list[dict]) -> dict:
    """Requests, errors, latency, cost, tools and models over a list of log lines."""
    answered = [e for e in entries if e.get("status") in ("success", "no_results")]
    latencies = [e["latency_ms"] for e in entries if isinstance(e.get("latency_ms"), int)]
    first_tokens = [e["first_token_ms"] for e in answered if isinstance(e.get("first_token_ms"), int)]
    costs = [e.get("cost") or 0.0 for e in answered]
    models: Counter = Counter()
    model_cost: Counter = Counter()
    for e in answered:
        for name, m in (e.get("models") or {}).items():
            models[name] += m.get("tokens") or 0
            model_cost[name] += m.get("cost") or 0.0
    return {
        "requests": len(entries),
        "by_status": dict(Counter(e.get("status") for e in entries)),
        "error_codes": dict(Counter(e["error_code"] for e in entries if e.get("error_code"))),
        "error_rate": round(sum(e.get("status") == "error" for e in entries) / len(entries), 3) if entries else 0.0,
        "latency_ms": {
            "median": int(statistics.median(latencies)) if latencies else None,
            "p95": _percentile(latencies, 95),
            "max": max(latencies) if latencies else None,
        },
        "first_token_ms": {  # streamed answers only
            "streamed": len(first_tokens),
            "median": int(statistics.median(first_tokens)) if first_tokens else None,
            "p95": _percentile(first_tokens, 95),
        },
        "cost_usd": {
            "total": round(sum(costs), 6),
            "average_per_answer": round(sum(costs) / len(costs), 6) if costs else None,
        },
        "tokens": {
            "total": sum(e.get("tokens") or 0 for e in answered),
            "average_per_answer": round(sum(e.get("tokens") or 0 for e in answered) / len(answered)) if answered else None,
        },
        "retrieved_chunks_avg": round(statistics.mean(e.get("retrieved_chunks", 0) for e in answered), 1) if answered else None,
        "tool_calls": dict(Counter(t for e in answered for t in e.get("tools", []))),
        "tool_errors": dict(Counter(t for e in answered for t in e.get("tool_errors", []))),
        "answers_using_tools_pct": round(100 * sum(bool(e.get("tools")) for e in answered) / len(answered), 1) if answered else None,
        "translation_fallbacks": sum(e.get("translation_origin") == "fallback" for e in answered),
        "models": {name: {"tokens": models[name], "cost_usd": round(model_cost[name], 6)} for name in models},
    }
