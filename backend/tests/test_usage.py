"""Token usage and cost (Phase 14). A fake chat model reports known tokens and costs: no API calls."""

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app import main, query_translation, rag, retrieval, tool_calling, usage
from app.query_translation import TranslatedQuery
from app.retrieval import RetrievedChunk
from app.usage import ModelUsage, track_usage


class FakeChat(BaseChatModel):
    """Answers "ok" and reports usage the way OpenRouter does through LangChain."""

    model_name: str = "fake/answer-model"
    input_tokens: int = 1000
    output_tokens: int = 200
    reasoning_tokens: int = 50
    cost: float | None = 0.0004

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
                "output_token_details": {"reasoning": self.reasoning_tokens},
            },
            response_metadata={
                "model_name": self.model_name,
                "token_usage": {} if self.cost is None else {"cost": self.cost},
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_every_call_inside_track_usage_is_counted_per_model():
    answer, translator = FakeChat(), FakeChat(model_name="fake/translator", input_tokens=300, output_tokens=40,
                                              reasoning_tokens=0, cost=0.0001)
    with track_usage() as tracker:
        answer.invoke("q1")
        answer.invoke("q2")  # a second answer round, e.g. after a tool call
        translator.invoke("q")

    a, t = tracker.models["fake/answer-model"], tracker.models["fake/translator"]
    assert (a.calls, a.input_tokens, a.output_tokens, a.reasoning_tokens) == (2, 2000, 400, 100)
    assert a.cost_usd == pytest.approx(0.0008) and a.cost_source == "reported"
    assert (t.calls, t.total_tokens, t.cost_usd) == (1, 340, pytest.approx(0.0001))


def test_calls_outside_tracking_are_not_counted():
    with track_usage() as tracker:
        pass
    FakeChat().invoke("not tracked")
    usage.record_embedding("emb", "not tracked")
    assert tracker.models == {}


def test_missing_cost_is_marked_not_reported_instead_of_zero():
    with track_usage() as tracker:
        FakeChat(cost=None).invoke("q")

    entry = tracker.models["fake/answer-model"]
    assert entry.cost_usd is None and entry.cost_source == "not reported"


def test_embedding_is_an_estimate_from_characters():
    with track_usage() as tracker:
        usage.record_embedding("openai/text-embedding-3-small", "x" * 400)  # ~100 tokens

    entry = tracker.models["openai/text-embedding-3-small"]
    assert entry.input_tokens == 100
    assert entry.cost_usd == pytest.approx(100 * 0.02 / 1_000_000)
    assert entry.cost_source == "estimated"


def test_answer_question_reports_usage_from_inside_the_pipeline(monkeypatch):
    chunk = RetrievedChunk(1, "tt1", "Prisoners", 2013, "Movie: Prisoners\n\nTense.", 0.2, {"doc_type": "review"})
    monkeypatch.setattr(query_translation, "translate_query", lambda m, history=None: TranslatedQuery.passthrough(m))

    def fake_retrieve(query, session, k=None):
        usage.record_embedding("openai/text-embedding-3-small", query)
        return [chunk]

    def fake_tool_loop(messages, tool_list):
        FakeChat().invoke(messages)  # the model call happens deep inside the pipeline
        return "Very tense.", []

    monkeypatch.setattr(retrieval, "retrieve", fake_retrieve)
    monkeypatch.setattr(tool_calling, "run_with_tools", fake_tool_loop)

    result = rag.answer_question("Why do people like Prisoners?", session=None)

    by_model = {m.model: m for m in result.usage}
    assert by_model["fake/answer-model"].input_tokens == 1000
    assert by_model["openai/text-embedding-3-small"].cost_source == "estimated"


def test_usage_totals_for_the_api():
    out = main.to_usage([
        ModelUsage("fake/answer-model", calls=2, input_tokens=2000, output_tokens=400, reasoning_tokens=100, cost_usd=0.0008),
        ModelUsage("fake/translator", calls=1, input_tokens=300, output_tokens=40, cost_usd=0.0001),
        ModelUsage("emb", calls=1, input_tokens=10, cost_usd=None, cost_source="not reported"),
    ])

    assert (out.input_tokens, out.output_tokens, out.total_tokens) == (2310, 440, 2750)
    assert out.estimated_cost_usd == pytest.approx(0.0009)  # unreported cost counts as 0 in the total
    assert [m.model for m in out.by_model] == ["fake/answer-model", "fake/translator", "emb"]
    assert out.by_model[0].reasoning_tokens == 100


def test_chat_response_includes_usage(api_without_database, monkeypatch):
    monkeypatch.setattr(
        rag, "answer_question",
        lambda q, s, h=None: rag.RagAnswer("ok", usage=[ModelUsage("fake/answer-model", calls=1, input_tokens=100, output_tokens=20, cost_usd=0.00005)]),
    )

    body = api_without_database.post("/api/chat", json={"message": "hi"}).json()

    assert body["usage"]["total_tokens"] == 120
    assert body["usage"]["estimated_cost_usd"] == pytest.approx(0.00005)
    assert body["usage"]["by_model"][0]["cost_source"] == "reported"
