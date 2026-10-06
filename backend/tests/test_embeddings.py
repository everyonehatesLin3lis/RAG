"""Spec 5 (specs/5.md) unit tests. A fake embedder stands in for the API, so these cost nothing."""

import pytest

from app import embeddings
from app.config import Settings
from app.embedding_job import api_calls_needed, embed_in_batches, plan_message
from app.errors import AppError

DIMS = 1536


class FakeEmbedder:
    """Records calls and returns vectors of a chosen size."""

    def __init__(self, dims: int = DIMS, drop_one: bool = False):
        self.dims, self.drop_one, self.calls = dims, drop_one, []

    def embed_documents(self, texts):
        self.calls.append(("documents", list(texts)))
        vectors = [[0.1] * self.dims for _ in texts]
        return vectors[:-1] if self.drop_one else vectors

    def embed_query(self, text):
        self.calls.append(("query", text))
        return [0.1] * self.dims


def settings(**overrides) -> Settings:
    base = dict(openrouter_api_key="or-key-123", embedding_api_key=None)
    return Settings(_env_file=None, **{**base, **overrides})


@pytest.fixture
def fake(monkeypatch):
    embedder = FakeEmbedder()
    monkeypatch.setattr(embeddings, "get_embedder", lambda: embedder)
    return embedder


# --- AC-001: one configured model, one module for documents and queries -----------------------


def test_ac001_model_and_dimension_are_configured():
    s = settings()
    assert s.embedding_model == "openai/text-embedding-3-small"
    assert s.embedding_dimensions == 1536


def test_ac001_embedder_uses_the_configured_model():
    embedder = embeddings.build_embedder(settings(embedding_model="openai/some-other-model"))
    assert embedder.model == "openai/some-other-model"


def test_ac001_documents_and_queries_go_through_the_same_embedder(fake):
    embeddings.embed_texts(["a review"])
    embeddings.embed_query("a question")
    assert fake.calls == [("documents", ["a review"]), ("query", "a question")]


# --- AC-004: embed in batches, every pending text sent once -------------------------------------


def test_ac004_every_chunk_is_embedded_and_saved_once():
    chunks = [(i, f"text {i}") for i in range(1, 251)]
    sent, saved = [], []

    def embed(texts):
        sent.extend(texts)
        return [[0.0] * DIMS for _ in texts]

    count = embed_in_batches(chunks, embed, saved.extend, batch_size=100)

    assert count == 250
    assert sent == [text for _, text in chunks]
    assert [chunk_id for chunk_id, _ in saved] == [chunk_id for chunk_id, _ in chunks]


# --- AC-005: say how many calls before calling ---------------------------------------------------


@pytest.mark.parametrize("chunks, calls", [(0, 0), (5, 1), (100, 1), (101, 2), (10540, 106)])
def test_ac005_api_call_estimate(chunks, calls):
    assert api_calls_needed(chunks, batch_size=100) == calls


def test_ac005_plan_message_states_chunks_and_calls():
    assert plan_message(10540, batch_size=100) == "10540 chunks to embed in 106 API calls (batches of 100)"


# --- AC-006: validate every response -------------------------------------------------------------


def test_ac006_wrong_dimension_is_rejected(monkeypatch):
    monkeypatch.setattr(embeddings, "get_embedder", lambda: FakeEmbedder(dims=768))
    with pytest.raises(AppError) as exc:
        embeddings.embed_texts(["a review"])
    assert exc.value.code == "EMBEDDING_FAILED"


def test_ac006_missing_vector_is_rejected(monkeypatch):
    monkeypatch.setattr(embeddings, "get_embedder", lambda: FakeEmbedder(drop_one=True))
    with pytest.raises(AppError) as exc:
        embeddings.embed_texts(["one", "two"])
    assert exc.value.code == "EMBEDDING_FAILED"


def test_ac006_wrong_dimension_query_is_rejected(monkeypatch):
    monkeypatch.setattr(embeddings, "get_embedder", lambda: FakeEmbedder(dims=3))
    with pytest.raises(AppError):
        embeddings.embed_query("a question")


def test_ac006_a_failing_batch_stores_nothing_from_that_batch():
    chunks = [(i, f"text {i}") for i in range(1, 201)]
    saved = []

    def embed(texts):
        if "text 150" in texts:
            raise AppError("EMBEDDING_FAILED", "bad response", 502)
        return [[0.0] * DIMS for _ in texts]

    with pytest.raises(AppError):
        embed_in_batches(chunks, embed, saved.extend, batch_size=100)

    assert [chunk_id for chunk_id, _ in saved] == list(range(1, 101))  # first batch only


# --- AC-007: key from the environment, fallback, never shown -------------------------------------


def test_ac007_embedding_key_is_used_when_set():
    assert embeddings.embedding_api_key(settings(embedding_api_key="emb-key-456")) == "emb-key-456"


def test_ac007_falls_back_to_openrouter_key():
    assert embeddings.embedding_api_key(settings(embedding_api_key="")) == "or-key-123"


def test_ac007_no_key_is_a_configuration_error():
    with pytest.raises(AppError) as exc:
        embeddings.embedding_api_key(settings(openrouter_api_key=None))
    assert exc.value.code == "EMBEDDING_NOT_CONFIGURED"


def test_ac007_key_is_not_visible_in_the_embedder_repr():
    embedder = embeddings.build_embedder(settings(embedding_api_key="emb-secret-789"))
    assert "emb-secret-789" not in repr(embedder)
    assert "emb-secret-789" not in str(embedder)
