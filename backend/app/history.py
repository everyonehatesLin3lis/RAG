"""Conversation history (Phase 11): store each question and answer, and feed recent turns back in.

A chat model has no memory between requests; every call starts blank. "Now compare it with Zodiac" only makes
sense if the earlier messages are sent along. So each exchange is saved in PostgreSQL, and the most recent
messages are added to the next request:

- to query translation, so "it" can be resolved into a stand-alone search ("compare Prisoners with Zodiac");
- to the answer model, so the reply follows on from the conversation.

Only the last HISTORY_MAX_MESSAGES messages are sent, each cut to HISTORY_MESSAGE_CHARS characters. Every
message sent costs input tokens on every later request, so history is a trade-off between memory and cost.
Earlier sources and tool results are not resent; each question retrieves fresh evidence.
"""

import uuid
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Conversation, Message


@dataclass(frozen=True)
class Turn:
    role: str  # "user" or "assistant"
    content: str


def get_or_create_conversation(session: Session, conversation_id: uuid.UUID | None) -> Conversation:
    """Find the conversation, or start it. The id comes from the browser, so an unknown id simply starts a new one."""
    if conversation_id is not None:
        existing = session.get(Conversation, conversation_id)
        if existing is not None:
            return existing
    conversation = Conversation(id=conversation_id) if conversation_id else Conversation()
    session.add(conversation)
    session.flush()  # assigns the id when the database generates it
    return conversation


def recent_turns(session: Session, conversation_id: uuid.UUID, limit: int | None = None) -> list[Turn]:
    """The last `limit` messages, oldest first, each trimmed to the configured length."""
    settings = get_settings()
    limit = settings.history_max_messages if limit is None else limit
    if limit <= 0:
        return []
    rows = session.scalars(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id.desc()).limit(limit)
    ).all()
    return [Turn(m.role, _trim(m.content, settings.history_message_chars)) for m in reversed(rows)]


def save_exchange(session: Session, conversation_id: uuid.UUID, question: str, answer: str) -> None:
    """Store the question and the answer together, so history never holds a question without its answer."""
    session.add_all([
        Message(conversation_id=conversation_id, role="user", content=question),
        Message(conversation_id=conversation_id, role="assistant", content=answer),
    ])
    session.flush()


def all_messages(session: Session, conversation_id: uuid.UUID) -> list[Message]:
    return list(
        session.scalars(select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)).all()
    )


def as_langchain_messages(turns: list[Turn]) -> list[BaseMessage]:
    return [HumanMessage(content=t.content) if t.role == "user" else AIMessage(content=t.content) for t in turns]


def _trim(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " …"
