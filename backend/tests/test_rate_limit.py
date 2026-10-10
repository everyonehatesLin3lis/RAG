"""Rate limits for the public deployment (app/rate_limit.py). No database, no models: the pipeline is faked."""

import pytest

from app import rag, rate_limit
from app.config import get_settings
from app.errors import AppError


@pytest.fixture(autouse=True)
def empty_counts():
    rate_limit.reset()
    yield
    rate_limit.reset()


@pytest.fixture
def limited_api(api_without_database, monkeypatch):
    monkeypatch.setattr(rag, "answer_question",
                        lambda question, session, history=None, on_event=None: rag.RagAnswer(answer="ok"))
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 2)
    monkeypatch.setattr(get_settings(), "rate_limit_per_day", 100)
    return api_without_database


def ask(ip: str, now: float, per_minute=2, per_day=0, today="2026-10-10") -> None:
    rate_limit.allow(ip, per_minute, per_day, now=now, today=today)


def test_per_minute_limit_counts_each_ip_separately():
    ask("1.1.1.1", 0)
    ask("1.1.1.1", 1)
    with pytest.raises(AppError) as refused:
        ask("1.1.1.1", 2)
    assert (refused.value.code, refused.value.status_code) == ("RATE_LIMITED", 429)
    ask("2.2.2.2", 2)  # another visitor is not affected


def test_per_minute_limit_frees_up_after_a_minute():
    ask("1.1.1.1", 0)
    ask("1.1.1.1", 30)
    with pytest.raises(AppError):
        ask("1.1.1.1", 59)
    ask("1.1.1.1", 60)  # the question at 0 s has left the window


def test_daily_limit_covers_everyone_and_resets_the_next_day():
    ask("1.1.1.1", 0, per_minute=0, per_day=2)
    ask("2.2.2.2", 1, per_minute=0, per_day=2)
    with pytest.raises(AppError) as refused:
        ask("3.3.3.3", 2, per_minute=0, per_day=2)
    assert refused.value.code == "RATE_LIMITED"
    ask("3.3.3.3", 3, per_minute=0, per_day=2, today="2026-10-11")


def test_a_refused_question_is_not_counted():
    ask("1.1.1.1", 0, per_minute=1)
    for second in range(1, 5):
        with pytest.raises(AppError):
            ask("1.1.1.1", second, per_minute=1)
    ask("1.1.1.1", 60, per_minute=1)  # only the first question counted, and it has expired


def test_chat_endpoints_answer_429_in_the_api_error_shape(limited_api):
    body = {"message": "Who directed Zodiac?"}
    assert limited_api.post("/api/chat", json=body).status_code == 200
    assert limited_api.post("/api/chat/stream", json=body).status_code == 200
    refused = limited_api.post("/api/chat", json=body)
    assert refused.status_code == 429
    assert refused.json()["error"]["code"] == "RATE_LIMITED"
    assert limited_api.post("/api/chat/stream", json=body).status_code == 429  # JSON error, not a started stream
    assert limited_api.get("/api/health").status_code == 200  # only the endpoints that ask the model are limited


def test_client_ip_is_the_rightmost_forwarded_address(limited_api):
    """A client can put any address at the front of X-Forwarded-For; the proxy appends the real one at the end."""
    body = {"message": "Who directed Zodiac?"}
    for fake in ("9.9.9.1", "9.9.9.2"):
        limited_api.post("/api/chat", json=body, headers={"X-Forwarded-For": f"{fake}, 5.5.5.5"})
    refused = limited_api.post("/api/chat", json=body, headers={"X-Forwarded-For": "9.9.9.3, 5.5.5.5"})
    assert refused.status_code == 429  # a new fake address in front did not give a fresh allowance
    other = limited_api.post("/api/chat", json=body, headers={"X-Forwarded-For": "6.6.6.6"})
    assert other.status_code == 200


def test_no_limits_by_default():
    settings = get_settings()
    assert (settings.rate_limit_per_minute, settings.rate_limit_per_day) == (0, 0)
