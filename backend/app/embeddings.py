"""The only place that calls the embedding API, for document chunks (Phase 5) and user queries (Phase 6).

An embedding is a list of numbers (here 1,536) that places a text in a "meaning space": texts about
similar things get vectors pointing in similar directions, so cosine distance between vectors works as
a measure of how related two texts are. Documents and queries must use the same model, otherwise their
vectors live in different spaces and the distances mean nothing.

LangChain's OpenAIEmbeddings talks to OpenRouter because OpenRouter speaks the OpenAI API. We turn off
check_embedding_ctx_length: with it on, LangChain tokenises texts locally with OpenAI's tokenizer and
sends token IDs, which OpenRouter does not accept; our chunks are far below the model's 8k-token limit anyway.
"""

from functools import lru_cache

import openai
from langchain_openai import OpenAIEmbeddings

from app.config import Settings, get_settings
from app.errors import AppError


# Implements: specs/5.md#AC-007
def embedding_api_key(settings: Settings) -> str:
    for key in (settings.embedding_api_key, settings.openrouter_api_key):
        if key is not None and key.get_secret_value():
            return key.get_secret_value()
    raise AppError("EMBEDDING_NOT_CONFIGURED", "The embedding model is not configured.", 500)


# Implements: specs/5.md#AC-001
def build_embedder(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=embedding_api_key(settings),  # stored by LangChain as a SecretStr, hidden in repr
        base_url=settings.openrouter_base_url,
        check_embedding_ctx_length=False,
        timeout=60,
        max_retries=2,
    )


@lru_cache
def get_embedder() -> OpenAIEmbeddings:
    return build_embedder(get_settings())


# Implements: specs/5.md#AC-006
def _check(vectors: list[list[float]], expected_count: int) -> list[list[float]]:
    dims = get_settings().embedding_dimensions
    if len(vectors) != expected_count:
        raise AppError(
            "EMBEDDING_FAILED", f"Expected {expected_count} embeddings, got {len(vectors)}.", 502
        )
    bad = [len(v) for v in vectors if len(v) != dims]
    if bad:
        raise AppError("EMBEDDING_FAILED", f"Expected {dims} numbers per embedding, got {bad[0]}.", 502)
    return vectors


def _call(fn, *args):
    try:
        return fn(*args)
    except openai.APITimeoutError as exc:
        raise AppError("EMBEDDING_TIMEOUT", "The embedding model took too long to respond.", 504) from exc
    except openai.APIError as exc:
        raise AppError("EMBEDDING_FAILED", "The embedding model is unavailable right now.", 502) from exc


# Implements: specs/5.md#AC-001, #AC-006
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed document chunks (one API call for the whole list)."""
    return _check(_call(get_embedder().embed_documents, texts), len(texts))


# Implements: specs/5.md#AC-001, #AC-006
def embed_query(text: str) -> list[float]:
    """Embed a user question with the same model as the documents."""
    return _check([_call(get_embedder().embed_query, text)], 1)[0]
