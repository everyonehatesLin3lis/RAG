"""FastAPI entrypoint. Run with: uvicorn app.main:app --reload --port 8000"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import llm
from app.config import get_settings
from app.errors import register_error_handlers
from app.schemas import ChatRequest, ChatResponse

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


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    answer = await llm.generate_answer(request.message)
    return ChatResponse(answer=answer)
