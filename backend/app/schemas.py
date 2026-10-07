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


class VectorResult(BaseModel):
    """One chunk returned by vector search, for the RAG process panel (Phase 12)."""

    rank: int
    chunk_id: str
    movie: str
    year: int | None = None
    doc_type: str | None = None  # "review" or "profile"
    critic: str | None = None
    distance: float  # cosine distance: 0 = same direction as the question's embedding
    similarity: float  # 1 - distance, easier to read: higher = closer
    excerpt: str


class KeywordResult(BaseModel):
    """One chunk returned by full-text keyword search (Phase 17)."""

    rank: int
    chunk_id: str
    movie: str
    year: int | None = None
    doc_type: str | None = None
    critic: str | None = None
    score: float  # PostgreSQL ts_rank_cd, normalised to 0..1: more and closer matches = higher
    excerpt: str


class FusedResult(BaseModel):
    """One chunk of the hybrid ranking (Phase 18): where each search put it, and its Reciprocal Rank Fusion score."""

    rank: int
    chunk_id: str
    movie: str
    year: int | None = None
    doc_type: str | None = None
    found_by: list[str]  # "vector", "keyword" or both
    vector_rank: int | None = None
    keyword_rank: int | None = None
    fused_score: float  # sum of 1 / (k + rank) over the lists the chunk appears in
    selected: bool  # sent to the model


class RagDebugOut(BaseModel):
    """What retrieval did for this answer (Phase 12). The first four fields are the plan's; the rest add detail."""

    original_query: str
    translated_query: str  # the semantic_query that was embedded
    vector_results: list[VectorResult] = []
    selected_chunks: list[str] = []  # chunk ids sent to the model
    keywords: list[str] = []
    filters: dict = {}
    translation_origin: str  # "model", "fallback" (translation failed) or "disabled"
    history_messages: int = 0
    timings_ms: dict[str, int] = {}
    keyword_results: list[KeywordResult] = []  # Phase 17
    strategy: str = "vector"  # Phase 18: "hybrid" or "vector"
    fused_results: list[FusedResult] = []  # Phase 18: the hybrid ranking (empty for "vector")
    tool_backend: str = "local"  # Phase 24: "mcp" (tools through the MCP server) or "local" (in-process)


class ModelUsageOut(BaseModel):
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0  # part of output_tokens: the model's hidden "thinking", billed as output
    cached_input_tokens: int = 0  # part of input_tokens served from the provider's prompt cache, billed at a discount
    total_tokens: int
    cost_usd: float | None = None
    cost_source: str  # "reported" (by OpenRouter), "estimated" (embedding), "not reported"


class Usage(BaseModel):
    """Phase 14. The first five fields are the plan's; by_model breaks them down per model."""

    model: str  # the answer model
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    by_model: list[ModelUsageOut] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    tool_calls: list[ToolCall] = []
    conversation_id: str | None = None  # Phase 11
    debug: RagDebugOut | None = None  # Phase 12
    usage: Usage | None = None  # Phase 14


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
