"""Pydantic models for every request and response body. Add fields as phases land; never rename them."""

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=2000)
    # Accepted now so the frontend contract is stable; used from Phase 11 (conversation history).
    conversation_id: str | None = Field(default=None, max_length=100)


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


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
