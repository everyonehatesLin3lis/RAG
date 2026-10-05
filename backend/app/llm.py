"""LLM access through LangChain.

OpenRouter exposes an OpenAI-compatible API, so LangChain's ChatOpenAI client works with it:
we only point base_url at OpenRouter and pass an OpenRouter key and model name.
"""

from functools import lru_cache

import openai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.errors import AppError

SYSTEM_PROMPT = (
    "You are Movie Research Copilot, an assistant that answers questions about movies. "
    "Stay on the topic of movies, film reviews and recommendations; politely decline anything else. "
    "Be concise. If you are not sure about a fact, say so instead of guessing."
)


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


async def generate_answer(message: str) -> str:
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=message)]
    try:
        response = await get_chat_model().ainvoke(messages)
    except openai.APITimeoutError as exc:
        raise AppError("LLM_TIMEOUT", "The language model took too long to respond.", 504) from exc
    except openai.APIError as exc:
        raise AppError("LLM_UNAVAILABLE", "The language model is unavailable right now.", 502) from exc

    return response.text
