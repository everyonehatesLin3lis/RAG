"""LLM access through LangChain.

OpenRouter exposes an OpenAI-compatible API, so LangChain's ChatOpenAI client works with it:
we only point base_url at OpenRouter and pass an OpenRouter key and model name.
The prompt itself is built elsewhere (app/rag.py); this module only sends messages and handles failures.
"""

from collections.abc import Callable
from functools import lru_cache

import openai
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import get_settings
from app.errors import AppError


class OpenRouterChat(ChatOpenAI):
    """ChatOpenAI that keeps OpenRouter's usage block when streaming (Phase 25).

    Without streaming, LangChain puts the provider's whole usage dict, including OpenRouter's `cost`, into
    response_metadata["token_usage"], which app/usage.py reads. When streaming, the usage arrives in the last chunk
    and LangChain keeps only the token counts, so the cost would be lost. This puts the raw block back, plus the model
    that chunk names (OpenRouter repeats finish_reason and model on that chunk, so LangChain's merged model_name comes
    out doubled: "x/yx/y").
    """

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        generation = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        if generation is not None and chunk.get("usage"):
            generation.message.response_metadata["token_usage"] = chunk["usage"]
            generation.message.response_metadata["usage_model"] = chunk.get("model")
        return generation


@lru_cache
def get_chat_model() -> ChatOpenAI:
    settings = get_settings()
    if settings.openrouter_api_key is None or not settings.openrouter_model:
        raise AppError("LLM_NOT_CONFIGURED", "The language model is not configured.", 500)

    return OpenRouterChat(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        timeout=30,
        max_retries=1,
        max_tokens=1024,
        stream_usage=True,  # ask for the usage chunk at the end of a stream
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


def invoke(runnable, messages):
    """Call a model, turning provider failures into our standard errors."""
    try:
        return runnable.invoke(messages)
    except openai.APITimeoutError as exc:
        raise AppError("LLM_TIMEOUT", "The language model took too long to respond.", 504) from exc
    except openai.APIError as exc:
        raise AppError("LLM_UNAVAILABLE", "The language model is unavailable right now.", 502) from exc


def stream(runnable, messages, on_text: Callable[[str], None]) -> AIMessageChunk:
    """Phase 25: call a model with streaming. Each piece of text goes to on_text as it arrives; the return value is
    the whole reply (text, tool calls, usage), the same thing invoke() would have returned."""
    reply: AIMessageChunk | None = None
    try:
        for chunk in runnable.stream(messages):
            if chunk.text:
                on_text(chunk.text)
            reply = chunk if reply is None else reply + chunk
    except openai.APITimeoutError as exc:
        raise AppError("LLM_TIMEOUT", "The language model took too long to respond.", 504) from exc
    except openai.APIError as exc:
        raise AppError("LLM_UNAVAILABLE", "The language model is unavailable right now.", 502) from exc
    return reply if reply is not None else AIMessageChunk(content="")


def tool_model(tools: list[BaseTool], tool_choice: str | None = None):
    """The chat model with the tools' schemas attached (Phase 10). LangChain's bind_tools sends each tool's
    name, description and JSON schema with every request; tool_choice="none" forbids calling them."""
    return get_chat_model().bind_tools(tools, tool_choice=tool_choice)


@lru_cache
def get_judge_model() -> ChatOpenAI:
    """Evaluation judge (Phase 20): EVALUATION_JUDGE_MODEL, from a different vendor than the system's own models,
    deterministic, with room for a list of claims."""
    settings = get_settings()
    if settings.openrouter_api_key is None:
        raise AppError("LLM_NOT_CONFIGURED", "The language model is not configured.", 500)
    return ChatOpenAI(
        model=settings.evaluation_judge_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0,
        timeout=90,
        max_retries=2,
        max_tokens=2500,
        default_headers={"X-Title": "Movie Research Copilot (evaluation)"},
    )


def judge(messages: list[BaseMessage], schema: type[BaseModel]) -> BaseModel | None:
    """Structured judgement from the judge model. Raises ValueError if the output does not fit the schema."""
    return invoke(get_judge_model().with_structured_output(schema, method="json_schema"), messages)


def structured(messages: list[BaseMessage], schema: type[BaseModel], reasoning: bool = False) -> BaseModel | None:
    """Ask for output matching a Pydantic schema (JSON schema mode). Returns the parsed object, or None
    if the model gave nothing usable. Raises ValueError if the JSON does not validate against the schema."""
    model = get_structured_model(reasoning).with_structured_output(schema, method="json_schema")
    return invoke(model, messages)
