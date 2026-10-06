"""Pydantic models for every request and response body. Add fields as phases land; never rename them."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=2000)
    # Phase 11: the browser keeps one UUID per conversation. Omitted -> a new conversation is started.
    conversation_id: UUID | None = None


class MessageRequest(BaseModel):
    """Body of POST /api/conversations/{id}/messages (the id is in the path)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    """One retrieved chunk the answer was based on (Phase 8). `movie`, `review_id` and `chunk_id` are the
    plan's citation fields; the rest lets the UI show who said it and link to the original review."""

    movie: str
    year: int | None = None
    review_id: str | None = None  # None for a movie-profile chunk
    chunk_id: str
    source: str  # "rotten_tomatoes" (critic review) or "tmdb_imdb" (movie profile)
    critic: str | None = None
    publication: str | None = None
    url: str | None = None
    excerpt: str


class ToolCall(BaseModel):
    """One tool call the model made (Phase 10), shown on the page in Phase 13."""

    tool: str
    arguments: dict
    result: dict


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    tool_calls: list[ToolCall] = []
    conversation_id: str | None = None  # Phase 11


class StoredMessage(BaseModel):
    role: str
    content: str
    created_at: datetime


class ConversationOut(BaseModel):
    """GET /api/conversations/{id} and POST /api/conversations."""

    id: str
    created_at: datetime
    messages: list[StoredMessage] = []


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
