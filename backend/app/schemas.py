"""Pydantic models for every request and response body. Add fields as phases land; never rename them."""

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=2000)
    # Accepted now so the frontend contract is stable; used from Phase 11 (conversation history).
    conversation_id: str | None = Field(default=None, max_length=100)


class ChatResponse(BaseModel):
    answer: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
