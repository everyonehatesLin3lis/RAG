"""Analytics: how long the deployed site makes a visitor wait, cold (after a quiet spell) and warm.

Sends the same question to the stream endpoint twice and prints when each event arrived, plus the server's own step
timings. Run it after the service has been idle for 15+ minutes: the first call then includes Cloud Run starting a
container (and the MCP server inside it) and Neon waking up. Costs about $0.002 (two answers).

    python scripts/cold_start_check.py https://movie-copilot-api-<hash>.europe-west1.run.app

Results from 2026-10-26: evaluation/results/experiments/cold_start.json.
"""

import json
import sys
import time
import urllib.request

QUESTION = "What did critics say about Zodiac?"


def ask(api: str) -> tuple[dict, dict]:
    body = json.dumps({"message": QUESTION}).encode()
    request = urllib.request.Request(f"{api}/api/chat/stream", data=body, headers={"Content-Type": "application/json"})
    started, arrived, server = time.time(), {}, {}
    with urllib.request.urlopen(request, timeout=120) as response:
        arrived["first_byte"] = time.time() - started
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            arrived.setdefault(event.get("type"), time.time() - started)
            if event.get("type") == "metadata":
                server = event["data"]["debug"]["timings_ms"]
    return {k: round(v, 2) for k, v in arrived.items()}, server


def main() -> None:
    api = sys.argv[1].rstrip("/")
    for label in ("cold", "warm"):
        arrived, server = ask(api)
        print(f"{label}: seconds after sending {arrived}")
        print(f"      server steps (ms) {server}")


if __name__ == "__main__":
    main()
