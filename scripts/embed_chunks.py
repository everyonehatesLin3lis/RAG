"""Phase 5: embed every RAG chunk that has no embedding yet (openai/text-embedding-3-small via OpenRouter).

Costs money: always look at the plan line first. Try a handful before a full run:
    python scripts/embed_chunks.py --dry-run      # only print what would be sent
    python scripts/embed_chunks.py --limit 5      # trial on 5 chunks
    python scripts/embed_chunks.py                # everything still pending

Run from the repo root with the backend venv active. Safe to re-run: embedded chunks are skipped.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.db import get_engine  # noqa: E402
from app.embedding_job import embed_in_batches, fetch_pending, plan_message, save_embeddings  # noqa: E402
from app.embeddings import embed_texts  # noqa: E402
from app.errors import AppError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="embed at most N chunks (trial run)")
    parser.add_argument("--batch-size", type=int, default=100, help="texts per API call (default 100)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit without calling the API")
    args = parser.parse_args()

    engine = get_engine()
    with engine.connect() as conn:
        pending = fetch_pending(conn, limit=args.limit)

    characters = sum(len(content) for _, content in pending)
    # Implements: specs/5.md#AC-005: say how much will be sent before sending anything.
    print(f"model: {get_settings().embedding_model}")
    print(f"plan:  {plan_message(len(pending), args.batch_size)}, ~{characters // 4:,} tokens")
    if args.dry_run or not pending:
        return

    def save(rows):
        with engine.begin() as conn:  # commit each batch, so a later failure keeps finished work
            save_embeddings(conn, rows)

    def progress(done, total):
        print(f"  {done}/{total} embedded", flush=True)

    started = time.monotonic()
    try:
        done = embed_in_batches(pending, embed_texts, save, args.batch_size, progress)
    except AppError as exc:
        sys.exit(f"stopped: {exc.code}: {exc.message} (finished batches are saved; re-run to continue)")

    with engine.connect() as conn:
        remaining = len(fetch_pending(conn))
    print(f"done:  {done} chunks in {time.monotonic() - started:.1f}s; chunks still without embedding: {remaining}")


if __name__ == "__main__":
    main()
