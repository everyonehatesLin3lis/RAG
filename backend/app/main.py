"""FastAPI entrypoint. Run with: uvicorn app.main:app --reload --port 8000"""

from uuid import UUID

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import history, rag
from app.config import get_settings
from app.db import get_session
from app.errors import AppError, register_error_handlers
from app.models import Conversation
from app.retrieval import RetrievedChunk
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationOut,
    MessageRequest,
    Source,
    StoredMessage,
    ToolCall,
)

settings = get_settings()

app = FastAPI(title="Movie Research Copilot")
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
@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, session: Session = Depends(get_session)) -> ChatResponse:
    return answer_in_conversation(session, request.conversation_id, request.message)


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


@app.post("/api/conversations/{conversation_id}/messages", response_model=ChatResponse)
def post_message(conversation_id: UUID, request: MessageRequest, session: Session = Depends(get_session)) -> ChatResponse:
    return answer_in_conversation(session, conversation_id, request.message)


def answer_in_conversation(session: Session, conversation_id: UUID | None, message: str) -> ChatResponse:
    """Load recent history, answer with it, then store the new question and answer together."""
    conversation = history.get_or_create_conversation(session, conversation_id)
    past = history.recent_turns(session, conversation.id)
    result = rag.answer_question(message, session, past)
    history.save_exchange(session, conversation.id, message, result.answer)
    session.commit()
    return ChatResponse(
        answer=result.answer,
        sources=[to_source(chunk) for chunk in result.sources],
        tool_calls=[ToolCall(tool=c.tool, arguments=c.arguments, result=c.result) for c in result.tool_calls],
        conversation_id=str(conversation.id),
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
