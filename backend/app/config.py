"""Application settings, read from environment variables (or backend/.env).

Secrets live only here and are never returned by an endpoint or written to logs.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/.env, wherever the server is started from
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # LLM (Phase 1). The model name is configuration so cost tracking can read it.
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str | None = None

    # Database (Phase 3)
    database_url: str | None = None

    # Embeddings (Phase 5). Implements: specs/5.md#AC-001, #AC-007
    # One model for documents and queries; the dimension must match rag_chunks.embedding (vector(1536)).
    # Served through OpenRouter, so the key falls back to OPENROUTER_API_KEY when EMBEDDING_API_KEY is empty.
    embedding_api_key: SecretStr | None = None
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Retrieval (Phase 6). Implements: specs/6.md#AC-002. The plan asks for the top 5–10 chunks.
    retrieval_top_k: int = Field(default=8, ge=5, le=10)
    # Keyword search (Phase 17): how many full-text matches to fetch ("top 10 each" in the plan's hybrid example).
    keyword_top_k: int = Field(default=10, ge=1, le=50)
    # Hybrid search (Phase 18). Implements: specs/18.md#AC-004. "vector" skips fusion, for comparison.
    retrieval_strategy: Literal["hybrid", "vector"] = "hybrid"
    hybrid_candidates: int = Field(default=10, ge=1, le=50)  # taken from each search before fusing
    rrf_k: int = Field(default=60, ge=1, le=1000)

    # Conversation history (Phase 11): how much of the past is sent with each new question.
    history_max_messages: int = Field(default=6, ge=0, le=20)  # 6 messages = the last 3 question/answer pairs
    history_message_chars: int = Field(default=1500, ge=100, le=10000)

    # Query translation (Phase 7): an extra LLM call that rewrites the question before retrieval.
    query_translation_enabled: bool = True
    # Model for the rewrite step; empty = the chat model (OPENROUTER_MODEL). A small fast model can do this job.
    query_translation_model: str | None = None
    # MiMo is a reasoning model; reasoning makes richer rewrites but is much slower (see README findings).
    query_translation_reasoning: bool = False

    # Web
    cors_origins: list[str] = ["http://localhost:3000"]

    # Evaluation (Phase 20): the LLM judge, from a different vendor than the answer (MiMo) and translation (Gemini) models.
    evaluation_judge_model: str = "anthropic/claude-haiku-4.5"

    # Logging (Phase 15): one JSON line per chat request. Relative paths are relative to the repo root.
    request_log_path: str = str(Path(__file__).resolve().parents[2] / "logs" / "requests.jsonl")


@lru_cache
def get_settings() -> Settings:
    return Settings()
