"""Phase 21: compare the vector-only and hybrid evaluation runs (evaluation/results/vector.json, hybrid.json).

Reports, side by side with the difference: retrieval (hit rate, Recall@5/@8, Precision@5), answer quality (accuracy,
groundedness, hallucination rate, relevance), latency (average and median) and tokens and cost per answer, then the
same by question type, and every question whose outcome differs between the two runs.

No API calls: it only reads the two result files. Run from the repo root: python evaluation/compare_strategies.py
Writes evaluation/results/comparison.json.
"""

import json
from pathlib import Path
from statistics import median

RESULTS = Path(__file__).resolve().parent / "results"
STRATEGIES = ("vector", "hybrid")

METRICS = [  # (key in summary, label, higher is better)
    ("retrieval_hit_rate", "Correct source reached the model", True),
    ("retrieval_recall_at_5", "Recall@5", True),
    ("retrieval_recall_at_8", "Recall@8", True),
    ("retrieval_precision_at_5", "Precision@5", True),
    ("answer_accuracy", "Answer accuracy", True),
    ("groundedness", "Groundedness", True),
    ("hallucination_rate", "Hallucination rate", False),
    ("relevance", "Relevance", True),
    ("expected_tool_used_rate", "Expected tool used", True),
]


def load(strategy: str) -> dict:
    return json.loads((RESULTS / f"{strategy}.json").read_text(encoding="utf-8"))


def per_answer(questions: list[dict]) -> dict:
    return {
        "avg_latency_s": round(sum(q["latency_ms"] for q in questions) / len(questions) / 1000, 1),
        "median_latency_s": round(median(q["latency_ms"] for q in questions) / 1000, 1),
        "avg_tokens": round(sum(q["tokens"] for q in questions) / len(questions)),
        "avg_cost_usd": round(sum(q["cost_usd"] for q in questions) / len(questions), 6),
    }


def outcome(q: dict) -> tuple:
    s, r = q["scores"] or {}, q["retrieval"] or {}
    return s.get("correctness"), r.get("hit"), r.get("recall_at_8"), s.get("hallucination")


def repeat_runs(runs: dict) -> dict | None:
    """If repeat runs exist (<strategy>-r2.json, -r3.json …, made with run_evaluation.py --tag), combine them with the
    main run for the questions they cover: how much do answers vary between runs with the same strategy?"""
    extra = {s: sorted(RESULTS.glob(f"{s}-r*.json")) for s in STRATEGIES}
    if not all(extra.values()):
        return None
    all_runs = {s: [runs[s]] + [json.loads(p.read_text(encoding="utf-8")) for p in extra[s]] for s in STRATEGIES}
    ids = sorted({q["id"] for run in all_runs["vector"][1:] for q in run["questions"]})
    per_question, totals = {}, {}
    for s in STRATEGIES:
        rows = [{q["id"]: q for q in run["questions"]} for run in all_runs[s]]
        for i in ids:
            per_question.setdefault(i, {})[s] = {
                "correctness": [r[i]["scores"]["correctness"] for r in rows],
                "recall_at_8": [r[i]["retrieval"]["recall_at_8"] if r[i]["retrieval"] else None for r in rows],
                "hallucination": [r[i]["scores"]["hallucination"] for r in rows],
            }
        cells = [per_question[i][s] for i in ids]
        totals[s] = {
            "runs": len(rows),
            "mean_correctness": round(sum(sum(c["correctness"]) for c in cells) / (len(ids) * len(rows)), 3),
            "hallucinations": f"{sum(sum(c['hallucination']) for c in cells)}/{len(ids) * len(rows)}",
            "mean_recall_at_8": round(sum(sum(x for x in c["recall_at_8"] if x is not None) for c in cells)
                                      / (len(ids) * len(rows)), 3),
        }
    print(f"\nRepeat runs on {len(ids)} questions ({', '.join(ids)}):")
    for s in STRATEGIES:
        print(f"  {s:7} {totals[s]}")
    return {"questions": ids, "totals": totals, "per_question": per_question}


def main() -> None:
    runs = {s: load(s) for s in STRATEGIES}
    summaries = {s: runs[s]["summary"] for s in STRATEGIES}
    answers = {s: per_answer(runs[s]["questions"]) for s in STRATEGIES}

    print(f"{'':34} {'vector':>8} {'hybrid':>8} {'change':>8}")
    overall = {}
    for key, label, higher_better in METRICS:
        v, h = summaries["vector"][key], summaries["hybrid"][key]
        delta = None if v is None or h is None else round(h - v, 4)
        overall[key] = {"vector": v, "hybrid": h, "change": delta}
        better = "" if not delta else ("  better" if (delta > 0) == higher_better else "  worse")
        print(f"{label:34} {v!s:>8} {h!s:>8} {delta if delta is not None else '-':>8}{better}")
    for key in ("avg_latency_s", "median_latency_s", "avg_tokens", "avg_cost_usd"):
        v, h = answers["vector"][key], answers["hybrid"][key]
        overall[key] = {"vector": v, "hybrid": h, "change": round(h - v, 6)}
        print(f"{key:34} {v!s:>8} {h!s:>8} {round(h - v, 6)!s:>8}")

    print("\nBy question type (answer accuracy / Recall@5):")
    by_type = {}
    for qtype in summaries["hybrid"]["by_type"]:
        v, h = summaries["vector"]["by_type"][qtype], summaries["hybrid"]["by_type"][qtype]
        by_type[qtype] = {s: {"answer_accuracy": x["answer_accuracy"], "recall_at_5": x["retrieval_recall_at_5"],
                              "groundedness": x["groundedness"]} for s, x in (("vector", v), ("hybrid", h))}
        print(f"  {qtype:15} n={h['questions']:2}  accuracy {v['answer_accuracy']} -> {h['answer_accuracy']}   "
              f"recall@5 {v['retrieval_recall_at_5']} -> {h['retrieval_recall_at_5']}")

    vq = {q["id"]: q for q in runs["vector"]["questions"]}
    changed = []
    print("\nQuestions whose outcome differs (correctness, hit, recall@8, hallucination):")
    for q in runs["hybrid"]["questions"]:
        a, b = outcome(vq[q["id"]]), outcome(q)
        if a != b:
            changed.append({"id": q["id"], "type": q["type"], "question": q["question"],
                            "vector": dict(zip(("correctness", "hit", "recall_at_8", "hallucination"), a)),
                            "hybrid": dict(zip(("correctness", "hit", "recall_at_8", "hallucination"), b))})
            print(f"  {q['id']:3} {q['type']:14} vector {a} -> hybrid {b} | {q['question'][:60]}")
    if not changed:
        print("  none")

    repeats = repeat_runs(runs)

    plan_fields = {s: {k: summaries[s][k] for k in ("strategy", "retrieval_recall_at_5", "answer_accuracy", "groundedness",
                                                     "avg_latency_ms", "avg_cost_usd")} for s in STRATEGIES}
    out = RESULTS / "comparison.json"
    out.write_text(json.dumps({"plan_fields": plan_fields, "overall": overall, "by_type": by_type,
                               "changed_questions": changed, "repeats": repeats,
                               "judge_cost_usd": {s: summaries[s]["judge"]["cost_usd"] for s in STRATEGIES}},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {out.relative_to(RESULTS.parent.parent)}")


if __name__ == "__main__":
    main()
