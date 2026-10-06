"""FastAPI entrypoint. Run with: uvicorn app.main:app --reload --port 8000"""

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import rag
from app.config import get_settings
from app.db import get_session
from app.errors import register_error_handlers
from app.retrieval import RetrievedChunk
from app.schemas import ChatRequest, ChatResponse, Source, ToolCall

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
    result = rag.answer_question(request.message, session)
    return ChatResponse(
        answer=result.answer,
        sources=[to_source(chunk) for chunk in result.sources],
        tool_calls=[ToolCall(tool=c.tool, arguments=c.arguments, result=c.result) for c in result.tool_calls],
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
