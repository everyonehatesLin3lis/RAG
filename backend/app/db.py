"""Database connection. One engine per process; one session per request."""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.errors import AppError


@lru_cache
def get_engine() -> Engine:
    url = get_settings().database_url
    if not url:
        raise AppError("DATABASE_NOT_CONFIGURED", "The database is not configured.", 500)
    # pool_pre_ping drops dead connections (e.g. after the container restarts) instead of failing a request.
    # connect_timeout makes an unreachable database fail fast instead of hanging the request.
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 5})


def get_session() -> Iterator[Session]:
    """FastAPI dependency: `session: Session = Depends(get_session)`."""
    session = sessionmaker(bind=get_engine())()
    try:
        yield session
    finally:
        session.close()
