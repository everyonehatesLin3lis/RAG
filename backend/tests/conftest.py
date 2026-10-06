import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.db import get_engine
from app.errors import AppError


@pytest.fixture
def session():
    """A session on the real local database inside a transaction that is always rolled back.

    Skips the test when the database is not reachable, so the rest of the suite runs without Docker.
    """
    try:
        connection = get_engine().connect()
    except (AppError, OperationalError) as exc:
        pytest.skip(f"database not available: {exc}")
    transaction = connection.begin()
    db = Session(bind=connection)
    try:
        yield db
    finally:
        db.close()
        transaction.rollback()
        connection.close()
