"""Token usage and cost per request (Phase 14).

One question makes several model calls: query translation (Gemini), one or more answer rounds (MiMo, more
when tools are used), and one embedding of the search query. We add them all up per model.

How it is collected: a LangChain callback handler sees the result of every chat-model call made while
`track_usage()` is active, without passing anything through the pipeline. It reads:
- token counts from LangChain's usage_metadata (input, output, and the reasoning part of the output);
- the cost OpenRouter reports for the call (`token_usage.cost`), so LLM cost is real, not computed from a price list.

The embedding call goes through LangChain's embeddings client, which does not pass usage on, so it is
estimated (about 4 characters per token, times the list price) and marked "estimated".

Why cost grows with context: every token sent is billed. A RAG answer sends the system rules, recent history,
8 retrieved chunks and the question, so input tokens (about 1,000+) far outnumber the question itself; each tool
round sends all of that again plus the tool result.
"""

import math
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.tracers.context import register_configure_hook

EMBEDDING_PRICE_PER_MILLION_USD = 0.02  # openai/text-embedding-3-small on OpenRouter
CHARS_PER_TOKEN = 4  # rough rule of thumb for English text


@dataclass
class ModelUsage:
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0  # input the provider served from its prompt cache (billed at a discount)
    cost_usd: float | None = 0.0
    cost_source: str = "reported"  # "reported" by OpenRouter, "estimated", or "not reported"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class UsageTracker(BaseCallbackHandler):
    """Collects usage for one request. LangChain calls on_llm_end after every chat-model call."""

    def __init__(self) -> None:
        self.models: dict[str, ModelUsage] = {}
        self._lock = threading.Lock()

    def _entry(self, model: str) -> ModelUsage:
        return self.models.setdefault(model, ModelUsage(model=model))

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                if message is None:
                    continue
                meta = message.response_metadata or {}
                usage = message.usage_metadata or {}
                cost = (meta.get("token_usage") or {}).get("cost")
                with self._lock:
                    entry = self._entry(meta.get("model_name") or "unknown")
                    entry.calls += 1
                    entry.input_tokens += usage.get("input_tokens", 0)
                    entry.output_tokens += usage.get("output_tokens", 0)
                    entry.reasoning_tokens += (usage.get("output_token_details") or {}).get("reasoning") or 0
                    entry.cached_input_tokens += (usage.get("input_token_details") or {}).get("cache_read") or 0
                    if cost is None or entry.cost_usd is None:
                        entry.cost_usd, entry.cost_source = None, "not reported"
                    else:
                        entry.cost_usd += float(cost)

    def record_embedding(self, model: str, text: str) -> None:
        tokens = math.ceil(len(text) / CHARS_PER_TOKEN)
        with self._lock:
            entry = self._entry(model)
            entry.calls += 1
            entry.input_tokens += tokens
            entry.cost_usd = (entry.cost_usd or 0.0) + tokens * EMBEDDING_PRICE_PER_MILLION_USD / 1_000_000
            entry.cost_source = "estimated"


# The same mechanism LangChain uses for its own get_usage_metadata_callback(): while the context variable holds
# a tracker, every LangChain run in this context gets it as a callback automatically.
_current: ContextVar[UsageTracker | None] = ContextVar("movie_copilot_usage", default=None)
register_configure_hook(_current, inheritable=True)


@contextmanager
def track_usage() -> Iterator[UsageTracker]:
    tracker = UsageTracker()
    token = _current.set(tracker)
    try:
        yield tracker
    finally:
        _current.reset(token)


def record_embedding(model: str, text: str) -> None:
    """Called by the embedder; does nothing when no request is being tracked (e.g. the batch embedding job)."""
    tracker = _current.get()
    if tracker is not None:
        tracker.record_embedding(model, text)
