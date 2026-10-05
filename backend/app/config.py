"""Application settings, read from environment variables (or backend/.env).

Secrets live only here and are never returned by an endpoint or written to logs.
"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM (Phase 1). The model name is configuration so cost tracking can read it.
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str | None = None

    # Database (Phase 3)
    database_url: str | None = None

    # Embeddings (Phase 5)
    embedding_api_key: SecretStr | None = None

    # Web
    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
