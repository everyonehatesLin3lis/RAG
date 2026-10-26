"""Analytics: what makes the system better, faster or cheaper. Reads evaluation runs; no API calls.

Four questions, each answered by runs of the same 49 evaluation questions that differ in one setting only:

1. Where does the time go?   per-step timings of the baseline runs (translation, search, generation, tool rounds)
2. Context size vs cost      5, 8 (baseline) and 10 chunks sent to the model: accuracy vs input tokens vs cost
3. Query translation         on (baseline) vs off: retrieval, accuracy, latency, cost
4. Answer model              MiMo v2.6 Flash (baseline) vs other cheap models: accuracy, latency, cost

Baseline = today's system: hybrid search, 8 chunks, translation by Gemini 3.1 Flash Lite, answers by MiMo, tools over
MCP (evaluation/results/final2/hybrid-r*.json, the Phase 29 final runs). Experiments ran once each
(evaluation/results/experiments/hybrid-<tag>.json); the baseline ran twice, and the gap between its two runs shows how
much a single run moves on its own. A difference smaller than that gap is not evidence.

Run from the repo root: python evaluation/analytics_report.py  ->  evaluation/results/analytics.json
"""

import json
from pathlib import Path
from statistics import mean, median

RESULTS = Path(__file__).resolve().parent / "results"
BASELINE = sorted((RESULTS / "final2").glob("hybrid-r[0-9].json"))
EXPERIMENTS = RESULTS / "experiments"
STEPS = ("translation", "embedding_and_search", "keyword_search", "generation")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


def describe(runs: list[dict]) -> dict:
    """One setup (one or more runs of it): quality from the judge, retrieval, speed and cost per answer."""
    rows = [q for run in runs for q in run["questions"] if not q["error"]]
    summaries = [run["summary"] for run in runs]

    def from_summary(key):
        values = [s[key] for s in summaries if s.get(key) is not None]
        return {"mean": round(mean(values), 4), "min": min(values), "max": max(values)} if values else None

    return {
        "runs": len(runs),
        "settings": summaries[0].get("settings"),
        "answer_accuracy": from_summary("answer_accuracy"),
        "groundedness": from_summary("groundedness"),
        "hallucination_rate": from_summary("hallucination_rate"),
        "relevance": from_summary("relevance"),
        "expected_tool_used_rate": from_summary("expected_tool_used_rate"),
        "retrieval_hit_rate": from_summary("retrieval_hit_rate"),
        "retrieval_recall_at_5": from_summary("retrieval_recall_at_5"),
        "errors": sum(len(s["errors"]) for s in summaries),
        "latency_ms_median": round(median(q["latency_ms"] for q in rows)),
        "latency_ms_p95": round(p95([q["latency_ms"] for q in rows])),
        "generation_ms_median": round(median(q["timings_ms"].get("generation", 0) for q in rows)),
        "input_tokens_avg": round(mean(q["input_tokens"] for q in rows)),
        "output_tokens_avg": round(mean(q["output_tokens"] for q in rows)),
        "cost_usd_per_answer": round(mean(q["cost_usd"] for q in rows), 6),
        "cost_usd_per_1000_answers": round(1000 * mean(q["cost_usd"] for q in rows), 2),
    }


def time_breakdown(runs: list[dict]) -> dict:
    rows = [q for run in runs for q in run["questions"] if not q["error"] and q["timings_ms"]]
    total = mean(q["timings_ms"]["total"] for q in rows)
    steps = {}
    for step in STEPS:
        values = [q["timings_ms"].get(step, 0) for q in rows]
        steps[step] = {"median_ms": round(median(values)), "mean_ms": round(mean(values)),
                       "share_of_total": round(mean(values) / total, 3)}
    with_tools = [q["timings_ms"]["generation"] for q in rows if q["tools"]]
    without = [q["timings_ms"]["generation"] for q in rows if not q["tools"]]
    by_type = {}
    for qtype in sorted({q["type"] for q in rows}):
        group = [q["timings_ms"]["total"] for q in rows if q["type"] == qtype]
        by_type[qtype] = round(median(group))
    return {
        "answers": len(rows),
        "total_ms_mean": round(total),
        "total_ms_median": round(median(q["timings_ms"]["total"] for q in rows)),
        "steps": steps,
        "generation_ms_median_with_tools": round(median(with_tools)) if with_tools else None,
        "generation_ms_median_without_tools": round(median(without)) if without else None,
        "answers_with_tools": len(with_tools),
        "total_ms_median_by_type": by_type,
    }


def experiment(tag: str) -> list[dict]:
    path = EXPERIMENTS / f"hybrid-{tag}.json"
    return [load(path)] if path.exists() else []


def show(title: str, setups: dict[str, dict], keys: list[tuple[str, str]]) -> None:
    print(f"\n{title}")
    names = list(setups)
    print(f"  {'':28}" + "".join(f"{n:>24}" for n in names))
    for key, label in keys:
        cells = []
        for n in names:
            value = setups[n].get(key)
            if isinstance(value, dict):
                value = f"{value['mean']}" + (f" ({value['min']}–{value['max']})" if setups[n]["runs"] > 1 else "")
            cells.append(f"{value!s:>24}")
        print(f"  {label:28}" + "".join(cells))


QUALITY = [("answer_accuracy", "Answer accuracy"), ("groundedness", "Groundedness"),
           ("hallucination_rate", "Unsupported-claim rate"), ("relevance", "Relevance")]
SPEED_COST = [("latency_ms_median", "Latency median (ms)"), ("latency_ms_p95", "Latency p95 (ms)"),
              ("input_tokens_avg", "Input tokens / answer"), ("output_tokens_avg", "Output tokens / answer"),
              ("cost_usd_per_1000_answers", "Cost per 1,000 answers ($)"), ("errors", "Pipeline errors")]


def main() -> None:
    baseline_runs = [load(p) for p in BASELINE]
    if not baseline_runs:
        raise SystemExit("no baseline runs in evaluation/results/final2")
    baseline = describe(baseline_runs)
    report = {"baseline": baseline, "time_breakdown": time_breakdown(baseline_runs)}

    tb = report["time_breakdown"]
    print(f"1. Where the time goes (baseline, {tb['answers']} answers): median {tb['total_ms_median']} ms, "
          f"mean {tb['total_ms_mean']} ms")
    for step, s in tb["steps"].items():
        print(f"  {step:22} median {s['median_ms']:6} ms   mean {s['mean_ms']:6} ms   {s['share_of_total']:.0%} of the time")
    print(f"  generation with a tool call: median {tb['generation_ms_median_with_tools']} ms "
          f"({tb['answers_with_tools']} answers), without: {tb['generation_ms_median_without_tools']} ms")
    print(f"  median total by question type: {tb['total_ms_median_by_type']}")

    context = {}
    if runs := experiment("k5"):
        context["5 chunks"] = describe(runs)
    context["8 chunks (baseline)"] = baseline
    if runs := experiment("k10"):
        context["10 chunks"] = describe(runs)
    report["context_size"] = context
    show("2. Context size vs cost", context,
         [("retrieval_hit_rate", "Correct film reached model"), *QUALITY, *SPEED_COST])

    translation = {"translation on (baseline)": baseline}
    if off := experiment("no-translation"):
        translation["translation off"] = describe(off)
    report["translation"] = translation
    show("3. Query translation on/off", translation,
         [("retrieval_hit_rate", "Correct film reached model"), ("retrieval_recall_at_5", "Recall@5"),
          *QUALITY, ("expected_tool_used_rate", "Expected tool used"), *SPEED_COST])

    models = {"mimo-v2.6-flash (baseline)": baseline}
    for tag in ("gpt-6-luna", "gemini-3.1-flash-lite"):
        if runs := experiment(tag):
            models[tag] = describe(runs)
    report["answer_models"] = models
    show("4. Answer model", models,
         [*QUALITY, ("expected_tool_used_rate", "Expected tool used"), ("generation_ms_median", "Generation median (ms)"),
          *SPEED_COST])

    cold = EXPERIMENTS / "cold_start.json"
    if cold.exists():
        report["cold_start"] = load(cold)
        c = report["cold_start"]["client_seconds"]
        print(f"\n5. Deployed site: first words after {c['cold']['first_token']} s cold, {c['warm']['first_token']} s warm")

    out = RESULTS / "analytics.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
