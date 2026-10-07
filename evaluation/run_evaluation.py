"""Phase 20: run the evaluation dataset through the real system and save the results for one strategy.

Costs money: about $0.04 for the system's answers and about $0.25 for the judge per full run of 49 questions
(estimates), and 15-25 minutes, because each answer takes 15-30 s. Try a few first:

    python evaluation/run_evaluation.py --limit 3            # trial
    python evaluation/run_evaluation.py                      # all 49, default strategy (hybrid)
    python evaluation/run_evaluation.py --strategy vector    # Phase 21 comparison
    python evaluation/run_evaluation.py --only F1,M3         # selected questions

Run from the repo root with the backend venv active. Results: evaluation/results/<strategy>.json (rewritten after
every question, so an interrupted run keeps what it finished).
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy.orm import Session  # noqa: E402

from app import evaluation  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import get_engine  # noqa: E402
from app.usage import track_usage  # noqa: E402

DATASET = ROOT / "evaluation" / "evaluation_dataset.jsonl"
RESULTS_DIR = ROOT / "evaluation" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy", choices=["hybrid", "vector"], default=None)
    parser.add_argument("--limit", type=int, help="only the first N questions (trial)")
    parser.add_argument("--only", help="comma-separated question ids")
    args = parser.parse_args()

    settings = get_settings()
    if args.strategy:
        settings.retrieval_strategy = args.strategy  # the pipeline reads the strategy from settings
    strategy = settings.retrieval_strategy

    items = evaluation.load_dataset(DATASET)
    if args.only:
        wanted = set(args.only.split(","))
        items = [i for i in items if i.id in wanted]
    if args.limit:
        items = items[: args.limit]
    print(f"strategy={strategy}  questions={len(items)}  answer model={settings.openrouter_model}  "
          f"judge={settings.evaluation_judge_model}", flush=True)

    started = datetime.now(UTC).isoformat(timespec="seconds")
    rows, judge_cost = [], 0.0
    for n, item in enumerate(items, start=1):
        with Session(get_engine()) as session:
            rec = evaluation.run_question(item, session)
        with track_usage() as judge_usage:  # counted separately from the system's own cost
            judgement = evaluation.judge(item, rec) if not rec.error else None
        judge_cost += sum(m.cost_usd or 0.0 for m in judge_usage.models.values())

        row = {
            "id": item.id, "type": item.type, "question": item.question, "answer": rec.answer, "error": rec.error,
            "latency_ms": rec.latency_ms, "cost_usd": round(rec.cost_usd, 6), "tokens": rec.tokens,
            "chunks": [{"chunk_id": c.id, "movie": f"{c.movie_title} ({c.year})"} for c in rec.chunks],
            "tools": [{"tool": c.tool, "arguments": c.arguments} for c in rec.tool_calls],
            "retrieval": evaluation.retrieval_metrics(item, rec.chunks),
            "rules": evaluation.rule_checks(item, rec),
            "judged": judgement is not None,
            "judgement": judgement.model_dump() if judgement else None,
            "scores": evaluation.scores(judgement) if judgement else None,
        }
        rows.append(row)

        s = row["scores"]
        verdict = rec.error or (f"correct={s['correctness']} grounded={s['groundedness']} relevant={s['relevance']}"
                                if s else "NOT JUDGED")
        hit = row["retrieval"]["hit"] if row["retrieval"] else "-"
        print(f"[{n:2}/{len(items)}] {item.id:3} {item.type:14} {rec.latency_ms / 1000:5.1f}s hit={hit!s:5} {verdict}",
              flush=True)

        summary = evaluation.summarise(strategy, rows, round(judge_cost, 6), settings.evaluation_judge_model)
        summary["started_at"] = started
        path = evaluation.save_results(RESULTS_DIR, strategy, summary, rows)

    keys = ["retrieval_hit_rate", "retrieval_recall_at_5", "retrieval_precision_at_5", "answer_accuracy", "groundedness",
            "hallucination_rate", "relevance", "rule_must_contain_rate", "expected_tool_used_rate", "avg_latency_ms",
            "avg_cost_usd"]
    print("\n" + "\n".join(f"{k:26} {summary[k]}" for k in keys))
    print(f"{'judge cost':26} ${summary['judge']['cost_usd']}")
    print(f"{'errors':26} {summary['errors'] or 'none'}")
    print(f"{'not judged':26} {summary['not_judged'] or 'none'}")
    print(f"{'flagged for review':26} {summary['flagged_for_review'] or 'none'}")
    print(f"saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
