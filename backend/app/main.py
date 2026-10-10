"""FastAPI entrypoint. Run with: uvicorn app.main:app --reload --port 8000"""

import logging
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import UUID

from fastapi import Depends, FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app import history, mcp_client, rag, rate_limit, request_log, streaming
from app.config import get_settings
from app.db import get_session
from app.errors import AppError, ClientDisconnected, register_error_handlers
from app.models import Conversation
from app.retrieval import RetrievedChunk
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationOut,
    FusedResult,
    KeywordResult,
    MessageRequest,
    RagDebugOut,
    Source,
    StoredMessage,
    ModelUsageOut,
    ToolCall,
    Usage,
    VectorResult,
)
from app.usage import ModelUsage

settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Phase 24: start the MCP server with the backend, so the first question does not wait for it, and stop it on
    shutdown. If it cannot start now, the backend still runs and the first tool use tries again."""
    if settings.tool_backend == "mcp":
        try:
            await run_in_threadpool(mcp_client.get_mcp_client().connect)
        except AppError:
            logger.warning("MCP server not available at startup; will retry on first use")
    yield
    if settings.tool_backend == "mcp":
        await run_in_threadpool(mcp_client.get_mcp_client().close)


app = FastAPI(title="Movie Research Copilot", lifespan=lifespan)
register_error_handlers(app)

# The Next.js dev server runs on another port, so the browser needs CORS to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class HealthResponse(BaseModel):
    status: str


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


# A plain `def` (not async): embedding, database and LLM calls are blocking, and FastAPI runs sync
# endpoints in a worker thread so they don't stall other requests.
# The endpoints that ask the model are rate limited when deployed (app/rate_limit.py); off locally.
@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit.check)])
def chat(request: ChatRequest, session: Session = Depends(get_session)) -> ChatResponse:
    return answer_in_conversation(session, request.conversation_id, request.message)


# Phase 25: the same answer as /api/chat, sent as Server-Sent Events while it is produced (app/streaming.py).
# Invalid input is still rejected with the normal JSON error before the stream starts.
@app.post("/api/chat/stream", dependencies=[Depends(rate_limit.check)])
async def chat_stream(request: ChatRequest, session: Session = Depends(get_session)) -> StreamingResponse:
    def run(emit):
        return answer_in_conversation(session, request.conversation_id, request.message, on_event=emit)

    return StreamingResponse(
        streaming.chat_events(run),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # no proxy may hold the events back
    )


# Phase 11 conversation endpoints (the plan's proposed API). /api/chat stays the main entry point.
@app.post("/api/conversations", response_model=ConversationOut, status_code=201)
def create_conversation(session: Session = Depends(get_session)) -> ConversationOut:
    conversation = history.get_or_create_conversation(session, None)
    session.commit()
    return ConversationOut(id=str(conversation.id), created_at=conversation.created_at)


@app.get("/api/conversations/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: UUID, session: Session = Depends(get_session)) -> ConversationOut:
    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        raise AppError("CONVERSATION_NOT_FOUND", "No conversation with that id.", 404)
    return ConversationOut(
        id=str(conversation.id),
        created_at=conversation.created_at,
        messages=[
            StoredMessage(role=m.role, content=m.content, created_at=m.created_at)
            for m in history.all_messages(session, conversation.id)
        ],
    )


@app.post(
    "/api/conversations/{conversation_id}/messages", response_model=ChatResponse, dependencies=[Depends(rate_limit.check)]
)
def post_message(conversation_id: UUID, request: MessageRequest, session: Session = Depends(get_session)) -> ChatResponse:
    return answer_in_conversation(session, conversation_id, request.message)


def answer_in_conversation(
    session: Session, conversation_id: UUID | None, message: str, on_event=None
) -> ChatResponse:
    """Load recent history, answer with it, then store the new question and answer together.
    Phase 15: every request, answered or failed, writes one line to the request log.
    Phase 25: on_event receives the streaming events; a closed stream stores nothing."""
    started = perf_counter()
    logged_id = str(conversation_id) if conversation_id else None
    try:
        conversation = history.get_or_create_conversation(session, conversation_id)
        logged_id = str(conversation.id)
        past = history.recent_turns(session, conversation.id)
        if on_event is None:
            result = rag.answer_question(message, session, past)
        else:
            result = rag.answer_question(message, session, past, on_event=on_event)
        history.save_exchange(session, conversation.id, message, result.answer)
        session.commit()
    except ClientDisconnected:
        session.rollback()  # nothing of this exchange is kept
        request_log.write(request_log.disconnected_entry(logged_id, message, _ms_since(started)))
        raise
    except AppError as exc:
        request_log.write(request_log.error_entry(logged_id, message, exc.code, _ms_since(started)))
        raise
    except SQLAlchemyError:
        request_log.write(request_log.error_entry(logged_id, message, "DATABASE_UNAVAILABLE", _ms_since(started)))
        raise
    except Exception:
        request_log.write(request_log.error_entry(logged_id, message, "INTERNAL_ERROR", _ms_since(started)))
        raise
    request_log.write(request_log.success_entry(logged_id, message, result, _ms_since(started)))
    return ChatResponse(
        answer=result.answer,
        sources=[to_source(chunk) for chunk in result.sources],
        tool_calls=[ToolCall(tool=c.tool, arguments=c.arguments, result=c.result) for c in result.tool_calls],
        conversation_id=str(conversation.id),
        debug=to_debug(result.debug) if result.debug else None,
        usage=to_usage(result.usage),
    )


def _ms_since(start: float) -> int:
    return round((perf_counter() - start) * 1000)


def to_usage(models: list[ModelUsage]) -> Usage:
    by_model = [
        ModelUsageOut(
            model=m.model, calls=m.calls, input_tokens=m.input_tokens, output_tokens=m.output_tokens,
            reasoning_tokens=m.reasoning_tokens, cached_input_tokens=m.cached_input_tokens, total_tokens=m.total_tokens,
            cost_usd=round(m.cost_usd, 8) if m.cost_usd is not None else None, cost_source=m.cost_source,
        )
        for m in models
    ]
    return Usage(
        model=settings.openrouter_model or "unknown",
        input_tokens=sum(m.input_tokens for m in by_model),
        output_tokens=sum(m.output_tokens for m in by_model),
        total_tokens=sum(m.total_tokens for m in by_model),
        # Mostly OpenRouter's reported cost; the embedding part is an estimate, hence the plan's name.
        estimated_cost_usd=round(sum(m.cost_usd or 0.0 for m in by_model), 8),
        by_model=by_model,
    )


def to_debug(debug: rag.RagDebug) -> RagDebugOut:
    filters = {k: v for k, v in debug.translation.filters.model_dump().items() if v not in (None, [])}
    selected = set(debug.selected_chunk_ids)
    return RagDebugOut(
        original_query=debug.original_query,
        translated_query=debug.translation.semantic_query,
        vector_results=[
            VectorResult(
                rank=rank,
                chunk_id=str(chunk.id),
                movie=chunk.movie_title,
                year=chunk.year,
                doc_type=chunk.metadata.get("doc_type"),
                critic=chunk.metadata.get("critic"),
                distance=round(chunk.distance, 4),
                similarity=round(1 - chunk.distance, 4),
                excerpt=rag.excerpt(chunk),
            )
            for rank, chunk in enumerate(debug.vector_results, start=1)
        ],
        selected_chunks=[str(i) for i in debug.selected_chunk_ids],
        keywords=debug.translation.keywords,
        filters=filters,
        translation_origin=debug.translation.origin,
        history_messages=debug.history_messages,
        timings_ms=debug.timings_ms,
        keyword_results=[
            KeywordResult(
                rank=rank,
                chunk_id=str(chunk.id),
                movie=chunk.movie_title,
                year=chunk.year,
                doc_type=chunk.metadata.get("doc_type"),
                critic=chunk.metadata.get("critic"),
                score=round(chunk.keyword_score or 0.0, 4),
                excerpt=rag.excerpt(chunk),
            )
            for rank, chunk in enumerate(debug.keyword_results, start=1)
        ],
        strategy=debug.strategy,
        fused_results=[
            FusedResult(
                rank=rank,
                chunk_id=str(f.chunk.id),
                movie=f.chunk.movie_title,
                year=f.chunk.year,
                doc_type=f.chunk.metadata.get("doc_type"),
                found_by=list(f.ranks),
                vector_rank=f.ranks.get("vector"),
                keyword_rank=f.ranks.get("keyword"),
                fused_score=round(f.score, 6),
                selected=f.chunk.id in selected,
            )
            for rank, f in enumerate(debug.fused, start=1)
        ],
        tool_backend=debug.tool_backend,
        warnings=debug.warnings,
        per_film=debug.per_film,
    )


def to_source(chunk: RetrievedChunk) -> Source:
    review_id = chunk.metadata.get("review_id")
    return Source(
        movie=chunk.movie_title,
        year=chunk.year,
        review_id=str(review_id) if review_id is not None else None,
        chunk_id=str(chunk.id),
        source=chunk.metadata.get("source", "unknown"),
        critic=chunk.metadata.get("critic"),
        publication=chunk.metadata.get("publication"),
        url=chunk.url,
        excerpt=rag.excerpt(chunk),
    )
