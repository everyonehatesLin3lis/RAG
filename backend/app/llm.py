"""LLM access through LangChain.

OpenRouter exposes an OpenAI-compatible API, so LangChain's ChatOpenAI client works with it:
we only point base_url at OpenRouter and pass an OpenRouter key and model name.
The prompt itself is built elsewhere (app/rag.py); this module only sends messages and handles failures.
"""

from functools import lru_cache

import openai
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import get_settings
from app.errors import AppError


@lru_cache
def get_chat_model() -> ChatOpenAI:
    settings = get_settings()
    if settings.openrouter_api_key is None or not settings.openrouter_model:
        raise AppError("LLM_NOT_CONFIGURED", "The language model is not configured.", 500)

    return ChatOpenAI(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        timeout=30,
        max_retries=1,
        max_tokens=1024,
        default_headers={"X-Title": "Movie Research Copilot"},
    )


@lru_cache
def get_structured_model(reasoning: bool) -> ChatOpenAI:
    """Model for short structured output (query translation): QUERY_TRANSLATION_MODEL, or the chat model when unset.
    Deterministic, small token budget, and (by default) without reasoning, which OpenRouter lets us switch off."""
    settings = get_settings()
    model = settings.query_translation_model or settings.openrouter_model
    if settings.openrouter_api_key is None or not model:
        raise AppError("LLM_NOT_CONFIGURED", "The language model is not configured.", 500)

    return ChatOpenAI(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0,
        timeout=30,
        max_retries=1,
        max_tokens=1024 if reasoning else 300,
        extra_body=None if reasoning else {"reasoning": {"enabled": False}},
        default_headers={"X-Title": "Movie Research Copilot"},
    )


def _invoke(runnable, messages):
    try:
        return runnable.invoke(messages)
    except openai.APITimeoutError as exc:
        raise AppError("LLM_TIMEOUT", "The language model took too long to respond.", 504) from exc
    except openai.APIError as exc:
        raise AppError("LLM_UNAVAILABLE", "The language model is unavailable right now.", 502) from exc


def complete(messages: list[BaseMessage]) -> str:
    return _invoke(get_chat_model(), messages).text


def structured(messages: list[BaseMessage], schema: type[BaseModel], reasoning: bool = False) -> BaseModel | None:
    """Ask for output matching a Pydantic schema (JSON schema mode). Returns the parsed object, or None
    if the model gave nothing usable. Raises ValueError if the JSON does not validate against the schema."""
    model = get_structured_model(reasoning).with_structured_output(schema, method="json_schema")
    return _invoke(model, messages)
