"""Phase 29: combine the final evaluation runs (evaluation/results/final/<strategy>-r1.json, -r2, -r3) into one report.

Each strategy was run three times because answers vary between identical runs (Phase 21: single comparison answers
moved by 0.5). For every metric this reports the mean of the runs and the range (lowest, highest), per strategy and
per question type, then the plan's fields, and which questions were unstable or wrong in some run.

No API calls: it only reads the result files. Run from the repo root: python evaluation/final_report.py
Writes evaluation/results/final/summary.json.
"""

import json
from pathlib import Path
from statistics import mean

FINAL = Path(__file__).resolve().parent / "results" / "final"
STRATEGIES = ("vector", "hybrid")
METRICS = [  # (key, label, higher is better)
    ("retrieval_hit_rate", "Correct source reached the model", True),
    ("retrieval_recall_at_5", "Recall@5", True),
    ("retrieval_recall_at_8", "Recall@8", True),
    ("retrieval_precision_at_5", "Precision@5", True),
    ("answer_accuracy", "Answer accuracy", True),
    ("groundedness", "Groundedness", True),
    ("hallucination_rate", "Hallucination rate", False),
    ("relevance", "Relevance", True),
    ("expected_tool_used_rate", "Expected tool used", True),
    ("avg_latency_ms", "Average latency (ms)", False),
    ("avg_cost_usd", "Average cost per answer ($)", False),
]
PLAN_FIELDS = ("retrieval_recall_at_5", "answer_accuracy", "groundedness", "avg_latency_ms", "avg_cost_usd")


def spread(values: list) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "min": None, "max": None}
    average = mean(values)
    rounded = round(average) if all(isinstance(v, int) for v in values) else round(average, 4)  # ms stay whole
    return {"mean": rounded, "min": min(values), "max": max(values)}


def load(strategy: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(FINAL.glob(f"{strategy}-r[0-9].json"))]


def per_question_all(run_list: list[dict], key) -> dict:
    values = {}
    for run in run_list:
        for q in run["questions"]:
            values.setdefault(q["id"], []).append(key(q))
    return {i: mean(v) for i, v in values.items()}


def per_film_comparison(runs: dict) -> dict | None:
    """Phase 29 follow-up: the questions naming two films were re-run (3x per strategy) after per-film retrieval was
    added (results/final/perfilm/). Compare them with the same questions in the full runs, and work out the overall
    answer accuracy with their new scores in place of the old ones (the other 36 questions take the same path as before,
    so their measured scores still apply)."""
    folder = FINAL / "perfilm"
    after = {s: [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob(f"{s}-r[0-9].json"))]
             for s in STRATEGIES}
    if not all(after.values()):
        return None
    result = {}
    print("\nPer-film retrieval, the re-run questions (mean of runs: before -> after):")
    for s in STRATEGIES:
        ids = [q["id"] for q in after[s][0]["questions"]]

        def per_question(run_list, key):
            values = {}
            for run in run_list:
                for q in run["questions"]:
                    if q["id"] in ids:
                        values.setdefault(q["id"], []).append(key(q))
            return {i: mean(v) for i, v in values.items()}

        correct = lambda q: (q["scores"] or {}).get("correctness", 0.0)  # noqa: E731
        recall = lambda q: (q["retrieval"] or {}).get("recall_at_8") or 0.0  # noqa: E731
        claim = lambda q: float(bool((q["scores"] or {}).get("hallucination")))  # noqa: E731
        before_c, after_c = per_question(runs[s], correct), per_question(after[s], correct)
        before_r, after_r = per_question(runs[s], recall), per_question(after[s], recall)
        before_h, after_h = per_question(runs[s], claim), per_question(after[s], claim)

        # every question's mean correctness over the 3 full runs; then the same with the re-run questions replaced
        everything = per_question_all(runs[s], correct)
        projected = mean({**everything, **after_c}.values())
        result[s] = {
            "questions": ids,
            "correctness_before": round(mean(before_c.values()), 4), "correctness_after": round(mean(after_c.values()), 4),
            "recall_at_8_before": round(mean(before_r.values()), 4), "recall_at_8_after": round(mean(after_r.values()), 4),
            "answers_with_unsupported_claim_before": round(mean(before_h.values()), 4),
            "answers_with_unsupported_claim_after": round(mean(after_h.values()), 4),
            "answer_accuracy_full_runs": round(mean(everything.values()), 4),
            "answer_accuracy_with_per_film": round(projected, 4),
            "by_question": {i: {"before": round(before_c[i], 2), "after": round(after_c[i], 2)} for i in ids},
        }
        r = result[s]
        print(f"  {s:7} correctness {r['correctness_before']} -> {r['correctness_after']}   recall@8 "
              f"{r['recall_at_8_before']} -> {r['recall_at_8_after']}   overall accuracy {r['answer_accuracy_full_runs']}"
              f" -> {r['answer_accuracy_with_per_film']} (with the re-run scores)")
        for i in ids:
            if before_c[i] != after_c[i]:
                print(f"      {i:3} {before_c[i]:.2f} -> {after_c[i]:.2f}")
    return result


def main() -> None:
    runs = {s: load(s) for s in STRATEGIES}
    report = {"runs": {s: len(r) for s, r in runs.items()}, "overall": {}, "by_type": {}, "plan_fields": {},
              "questions": {}, "costs": {}}

    print(f"{'':34} {'vector (mean, range)':>24} {'hybrid (mean, range)':>24}")
    for key, label, _ in METRICS:
        row = {s: spread([r["summary"][key] for r in runs[s]]) for s in STRATEGIES}
        report["overall"][key] = row
        cells = [f"{row[s]['mean']} ({row[s]['min']}–{row[s]['max']})" for s in STRATEGIES]
        print(f"{label:34} {cells[0]:>24} {cells[1]:>24}")

    for s in STRATEGIES:
        report["plan_fields"][s] = {"strategy": s, **{k: report["overall"][k][s]["mean"] for k in PLAN_FIELDS}}
        report["costs"][s] = {
            "system_usd": round(sum(r["summary"]["avg_cost_usd"] * r["summary"]["questions"] for r in runs[s]), 4),
            "judge_usd": round(sum(r["summary"]["judge"]["cost_usd"] for r in runs[s]), 4),
        }

    print("\nBy question type, answer accuracy (mean of runs):")
    for qtype in runs["hybrid"][0]["summary"]["by_type"]:
        row = {s: spread([r["summary"]["by_type"][qtype]["answer_accuracy"] for r in runs[s]]) for s in STRATEGIES}
        n = runs["hybrid"][0]["summary"]["by_type"][qtype]["questions"]
        report["by_type"][qtype] = {"questions": n, **row,
                                    "recall_at_5": {s: spread([r["summary"]["by_type"][qtype]["retrieval_recall_at_5"]
                                                               for r in runs[s]])["mean"] for s in STRATEGIES}}
        print(f"  {qtype:15} n={n:2}  vector {row['vector']['mean']}  hybrid {row['hybrid']['mean']}")

    # Per question: correctness in every run, any unsupported claim, any fallback or failed translation.
    for s in STRATEGIES:
        for run in runs[s]:
            for q in run["questions"]:
                entry = report["questions"].setdefault(q["id"], {"type": q["type"], "question": q["question"]})
                cell = entry.setdefault(s, {"correctness": [], "hallucination": [], "warnings": 0, "errors": []})
                scores = q["scores"] or {}
                cell["correctness"].append(scores.get("correctness"))
                cell["hallucination"].append(scores.get("hallucination"))
                cell["warnings"] += bool(q.get("warnings"))
                if q["error"]:
                    cell["errors"].append(q["error"])

    print("\nQuestions not correct in every run (correctness per run):")
    for qid, entry in report["questions"].items():
        cells = {s: entry[s]["correctness"] for s in STRATEGIES if s in entry}
        if any(c != 1.0 for s in cells for c in cells[s]):
            print(f"  {qid:3} {entry['type']:14} vector {cells.get('vector')}  hybrid {cells.get('hybrid')} | "
                  f"{entry['question'][:55]}")
    flagged = {s: sum(e[s]["warnings"] for e in report["questions"].values() if s in e) for s in STRATEGIES}
    print(f"\nAnswers with a fallback warning: {flagged}")
    print(f"Costs (all runs): {report['costs']}")

    report["per_film"] = per_film_comparison(runs)

    out = FINAL / "summary.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out.relative_to(FINAL.parents[2])}")


if __name__ == "__main__":
    main()
