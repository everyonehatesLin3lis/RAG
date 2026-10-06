"""Application settings, read from environment variables (or backend/.env).

Secrets live only here and are never returned by an endpoint or written to logs.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
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

    # Web
    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
