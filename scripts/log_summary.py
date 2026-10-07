"""Monitoring (Phase 15): summarise logs/requests.jsonl.

Prints requests by status, error codes, latency (median / p95 / max), cost and tokens per answer, tool use,
tool errors, translation fallbacks and usage per model.

Run from the repo root (backend venv active): python scripts/log_summary.py [--last N] [--json]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import request_log  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--last", type=int, help="only the last N requests")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args()

    entries = request_log.read()
    if args.last:
        entries = entries[-args.last:]
    summary = request_log.summarise(entries)

    if args.json:
        print(json.dumps(summary, indent=2))
        return
    if not entries:
        print(f"No requests logged yet in {request_log.log_path()}")
        return

    lat, cost, tokens = summary["latency_ms"], summary["cost_usd"], summary["tokens"]
    print(f"Log: {request_log.log_path()}  ({entries[0]['timestamp']} … {entries[-1]['timestamp']})")
    print(f"Requests:      {summary['requests']}  {summary['by_status']}  error rate {summary['error_rate']:.1%}")
    if summary["error_codes"]:
        print(f"Errors:        {summary['error_codes']}")
    print(f"Latency:       median {lat['median'] / 1000:.1f} s, p95 {lat['p95'] / 1000:.1f} s, max {lat['max'] / 1000:.1f} s")
    if cost["average_per_answer"] is not None:
        print(f"Cost:          ${cost['total']:.6f} total, ${cost['average_per_answer']:.6f} per answer")
        print(f"Tokens:        {tokens['total']:,} total, {tokens['average_per_answer']:,} per answer")
        print(f"Retrieval:     {summary['retrieved_chunks_avg']} chunks per answer on average")
        print(f"Tools:         {summary['tool_calls'] or 'none'} in {summary['answers_using_tools_pct']}% of answers")
        if summary["tool_errors"]:
            print(f"Tool errors:   {summary['tool_errors']}")
        print(f"Translation:   {summary['translation_fallbacks']} fallbacks to the original question")
        for name, m in summary["models"].items():
            print(f"Model:         {name:32} {m['tokens']:>8,} tokens  ${m['cost_usd']:.6f}")


if __name__ == "__main__":
    main()
