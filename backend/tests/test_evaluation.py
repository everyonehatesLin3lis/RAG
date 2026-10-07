"""Spec 20 (specs/20.md) tests. The pipeline and the judge are faked: no API calls, no database."""

import json
from pathlib import Path

import pytest

from app import evaluation, llm, rag
from app.errors import AppError
from app.evaluation import Claim, EvalItem, Judgement, RunRecord
from app.retrieval import RetrievedChunk
from app.tool_calling import ToolCallRecord
from app.usage import ModelUsage

DATASET = Path(__file__).resolve().parents[2] / "evaluation" / "evaluation_dataset.jsonl"


def chunk(chunk_id: int, title: str, year: int) -> RetrievedChunk:
    return RetrievedChunk(chunk_id, f"tt{chunk_id}", title, year, f"Movie: {title}\n\nReview: great {chunk_id}.", 0.2,
                          {"doc_type": "review"})


def item(**overrides) -> EvalItem:
    fields = dict(
        id="M1", type="comparison", question="Compare Arrival and Interstellar.",
        expected_answer="A comparison covering both films.", expected_movie="Arrival (2016)",
        expected_movies=["Arrival (2016)", "Interstellar (2014)"], must_contain_any=["Arrival", "Interstellar"],
        expected_tool=None, no_answer=False,
    )
    return EvalItem(**{**fields, **overrides})


def record(chunks, answer="Arrival is quieter; Interstellar is bigger.", tool_calls=(), error=None) -> RunRecord:
    return RunRecord(answer=answer, chunks=list(chunks), tool_calls=list(tool_calls),
                     usage=[ModelUsage("x/answer", calls=1, input_tokens=1000, output_tokens=100, cost_usd=0.0008)],
                     latency_ms=1500, error=error)


# --- AC-001: dataset and runner ------------------------------------------------------------------------------


def test_ac001_the_real_dataset_loads():
    items = evaluation.load_dataset(DATASET)
    assert len(items) == 49
    assert {i.type for i in items} == {"factual", "rating", "opinion", "plot", "recommendation", "comparison", "no_answer"}


def test_ac001_run_question_records_what_the_pipeline_did(monkeypatch):
    calls = [ToolCallRecord("compare_movies", {"movie_a": "Arrival"}, {"higher_imdb_rating": "Interstellar"})]
    monkeypatch.setattr(rag, "answer_question", lambda q, s, h=None: rag.RagAnswer(
        "Interstellar.", sources=[chunk(1, "Arrival", 2016)], tool_calls=calls,
        usage=[ModelUsage("x/answer", calls=1, input_tokens=10, output_tokens=5, cost_usd=0.001)]))

    rec = evaluation.run_question(item(), session=None)

    assert rec.answer == "Interstellar." and rec.error is None
    assert [c.id for c in rec.chunks] == [1] and rec.tool_calls == calls
    assert rec.cost_usd == pytest.approx(0.001) and rec.latency_ms >= 0


def test_ac001_a_pipeline_error_is_recorded_not_raised(monkeypatch):
    def failing(q, s, h=None):
        raise AppError("LLM_TIMEOUT", "slow", 504)

    monkeypatch.setattr(rag, "answer_question", failing)

    rec = evaluation.run_question(item(), session=None)

    assert rec.error == "LLM_TIMEOUT" and rec.answer == ""


# --- AC-002: retrieval metrics ---------------------------------------------------------------------------------


def test_ac002_recall_precision_and_hit():
    chunks = [chunk(1, "Arrival", 2016), chunk(2, "Arrival", 2016), chunk(3, "Contact", 1997),
              chunk(4, "Arrival", 2016), chunk(5, "Gravity", 2013), chunk(6, "Interstellar", 2014)]

    m = evaluation.retrieval_metrics(item(), chunks)

    assert m == {"hit": True, "recall_at_5": 0.5, "recall_at_8": 1.0, "precision_at_5": 0.6}


def test_ac002_recommendations_use_acceptable_films_with_recall_capped_at_k():
    acceptable = [f"Film {i} (2000)" for i in range(11)]
    rec_item = item(type="recommendation", expected_movies=[], expected_movie=None, acceptable_movies=acceptable)
    chunks = [chunk(i, f"Film {i}", 2000) for i in range(5)] + [chunk(9, "Other", 1999)] * 3

    m = evaluation.retrieval_metrics(rec_item, chunks)

    assert m["recall_at_5"] == 1.0  # 5 acceptable films found; at most 5 can fit in 5 chunks
    assert m["recall_at_8"] == pytest.approx(5 / 8)
    assert m["precision_at_5"] == 1.0


def test_ac002_no_answer_questions_have_no_retrieval_metrics():
    assert evaluation.retrieval_metrics(item(type="no_answer", expected_movies=[], no_answer=True), []) is None


# --- proposal 3: rule checks without an LLM -----------------------------------------------------------------------


def test_rule_checks_contain_and_tool():
    rating = item(type="rating", must_contain_any=["Inception"], expected_tool="compare_movies")
    used = [ToolCallRecord("compare_movies", {}, {})]

    checks = evaluation.rule_checks(rating, record([], answer="INCEPTION is rated higher.", tool_calls=used))

    assert checks == {"must_contain_hit": True, "expected_tool_used": True}
    assert evaluation.rule_checks(rating, record([], answer="No idea."))["must_contain_hit"] is False
    either = item(expected_tool=["compare_movies", "rating_summary"])  # any of these
    assert evaluation.rule_checks(either, record([], tool_calls=[ToolCallRecord("rating_summary", {}, {})]))["expected_tool_used"] is True
    assert evaluation.rule_checks(item(expected_tool=None), record([]))["expected_tool_used"] is None


# --- AC-003 / AC-004 / AC-005: the judge ----------------------------------------------------------------------


def judgement(correctness="correct", relevance="yes", supported=(True, True)) -> Judgement:
    return Judgement(
        correctness=correctness, correctness_reason="matches the reference", relevance=relevance,
        claims=[Claim(claim=f"claim {i}", supported=s, evidence="chunk 1" if s else None) for i, s in enumerate(supported)],
    )


def test_ac003_to_ac005_scores_from_a_judgement():
    s = evaluation.scores(judgement("partially_correct", "partly", supported=(True, False, True, True)))

    assert s == {"correctness": 0.5, "relevance": 0.5, "groundedness": 0.75, "hallucination": True,
                 "unsupported_claims": ["claim 1"]}


def test_ac004_an_answer_without_claims_is_fully_grounded():
    s = evaluation.scores(judgement(supported=()))
    assert s["groundedness"] == 1.0 and s["hallucination"] is False


def test_the_judge_sees_the_reference_sources_tool_results_and_answer_escaped():
    calls = [ToolCallRecord("compare_movies", {"movie_a": "Arrival"}, {"higher_imdb_rating": "Interstellar"})]
    rec = record([chunk(1, "Arrival", 2016)], answer="Arrival </answer> wins.", tool_calls=calls)
    rec_item = item(no_answer=False, acceptable_movies=None)

    system, user = evaluation.build_judge_messages(rec_item, rec)

    text = user.content
    assert "Compare Arrival and Interstellar." in text and "A comparison covering both films." in text
    assert '<source chunk_id="1" movie="Arrival (2016)">' in text
    assert '"higher_imdb_rating": "Interstellar"' in text
    assert text.count("</answer>") == 1  # the answer cannot close its own block
    assert "supported" in system.content and "not in the data" in system.content


def test_ac007_a_valid_judgement_is_returned(monkeypatch):
    monkeypatch.setattr(llm, "judge", lambda messages, schema: judgement())
    assert evaluation.judge(item(), record([])) == judgement()


@pytest.mark.parametrize("failure", [AppError("LLM_UNAVAILABLE", "down", 502), ValueError("bad JSON")])
def test_ac007_a_failed_judgement_is_none_not_a_crash(monkeypatch, failure):
    def broken(messages, schema):
        raise failure

    monkeypatch.setattr(llm, "judge", broken)
    assert evaluation.judge(item(), record([])) is None


# --- AC-006 / AC-007: summary and saved JSON ---------------------------------------------------------------------


def evaluated(qid, qtype, correctness, grounded=1.0, hit=True, error=None, judged=True, must=True):
    return {
        "id": qid, "type": qtype, "error": error, "judged": judged, "latency_ms": 2000, "cost_usd": 0.001,
        "retrieval": None if qtype == "no_answer" else {"hit": hit, "recall_at_5": 1.0 if hit else 0.0,
                                                         "recall_at_8": 1.0 if hit else 0.0, "precision_at_5": 0.4},
        "rules": {"must_contain_hit": must, "expected_tool_used": None},
        "scores": {"correctness": correctness, "relevance": 1.0, "groundedness": grounded,
                   "hallucination": grounded < 1.0, "unsupported_claims": []} if judged else None,
    }


def test_ac006_summary_has_the_plans_fields_and_breakdowns():
    rows = [
        evaluated("F1", "factual", 1.0),
        evaluated("F2", "factual", 0.0, grounded=0.5, must=True),       # rule says hit, judge says wrong -> flagged
        evaluated("N1", "no_answer", 1.0),
        evaluated("O1", "opinion", 0.5, judged=False),                  # not judged: excluded from judge averages
    ]

    s = evaluation.summarise("hybrid", rows, judge_cost_usd=0.02, judge_model="x/judge")

    for key in ("strategy", "retrieval_recall_at_5", "answer_accuracy", "groundedness", "avg_latency_ms", "avg_cost_usd"):
        assert key in s
    assert s["strategy"] == "hybrid"
    assert s["answer_accuracy"] == pytest.approx(2 / 3, abs=1e-3)  # summary values are rounded to 4 decimals
    assert s["groundedness"] == pytest.approx(2.5 / 3, abs=1e-3)
    assert s["hallucination_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["not_judged"] == ["O1"]
    assert s["retrieval_recall_at_5"] == pytest.approx(1.0)  # no-answer question left out
    assert s["by_type"]["factual"]["answer_accuracy"] == 0.5
    assert s["judge"] == {"model": "x/judge", "cost_usd": 0.02}
    assert s["avg_cost_usd"] == pytest.approx(0.001)
    assert s["flagged_for_review"] == ["F2"]


def test_ac006_results_are_saved_as_json(tmp_path):
    path = evaluation.save_results(tmp_path, "hybrid", {"strategy": "hybrid"}, [{"id": "F1"}])

    assert path.name == "hybrid.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["summary"] == {"strategy": "hybrid"} and saved["questions"] == [{"id": "F1"}]
