"""Experiment: does query translation help retrieval for casual questions, and is reasoning worth its time?

16 casual questions, each about one known film, are searched three ways (top 8 chunks, as the chat does):
- raw:            the question embedded as typed
- translated:     semantic_query from translation with reasoning off (the default)
- translated+r:   semantic_query from translation with reasoning on

A question is a hit@k when one of the top k chunks belongs to the expected film.
Small and directional, not the Phase 20 evaluation. About 80 small API calls.

Run from the repo root (backend venv active): python scripts/compare_query_translation.py
"""

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app import embeddings, llm, retrieval  # noqa: E402
from app.db import get_engine  # noqa: E402
from app.query_translation import TranslatedQuery, build_translation_messages  # noqa: E402

RESULTS = ROOT / "docs" / "experiments" / "query_translation_comparison.json"
K = 8

QUESTIONS = [
    ("that one where leo dicaprio steals ideas in dreams, worth it?", "Inception", 2010),
    ("ok so the movie where hugh jackman's kid gets kidnapped, any good?", "Prisoners", 2013),
    ("the fincher one about the san fran serial killer with the codes, is it long?", "Zodiac", 2007),
    ("brad pitt soap guy fight movie, what do critics think", "Fight Club", 1999),
    ("matt damon stuck on mars eating potatoes lol", "The Martian", 2015),
    ("that drummer movie with the psycho teacher, is it actually good", "Whiplash", 2014),
    ("keanu killing everyone cause of his dog", "John Wick", 2014),
    ("the one where bill murray keeps waking up on the same day", "Groundhog Day", 1993),
    ("guy with no short term memory tattoos clues on himself", "Memento", 2000),
    ("creepy movie where the girlfriend's parents hypnotize the black guy", "Get Out", 2017),
    ("rat that cooks in paris, would kids like it?", "Ratatouille", 2007),
    ("old man ties balloons to his house and flies off", "Up", 2009),
    ("joker movie with heath ledger, why is everyone obsessed", "The Dark Knight", 2008),
    ("that sam mendes ww1 film that looks like one long take", "1917", 2019),
    ("natalie portman ballet dancer going crazy", "Black Swan", 2010),
    ("leo on an island with a mental hospital, big twist ending", "Shutter Island", 2010),
]


def rank_of(session, query: str, movie_id: str) -> int | None:
    chunks = retrieval.search_chunks(session, embeddings.embed_query(query), k=K)
    return next((i + 1 for i, c in enumerate(chunks) if c.movie_id == movie_id), None)


def translate(question: str, reasoning: bool) -> tuple[str, dict, float]:
    start = time.monotonic()
    result = llm.structured(build_translation_messages(question), TranslatedQuery, reasoning=reasoning)
    seconds = time.monotonic() - start
    result = result or TranslatedQuery.passthrough(question)
    return result.semantic_query, result.model_dump(), seconds


def summarise(name: str, ranks: list, seconds: list | None) -> dict:
    row = {
        "variant": name,
        "hit_at_1": round(sum(r == 1 for r in ranks) / len(ranks), 3),
        "hit_at_8": round(sum(r is not None for r in ranks) / len(ranks), 3),
        "translation_seconds_median": round(statistics.median(seconds), 1) if seconds else None,
    }
    print(f"{name:14} hit@1={row['hit_at_1']:.2f}  hit@{K}={row['hit_at_8']:.2f}  translation median={row['translation_seconds_median']}s")
    return row


def main() -> None:
    engine = get_engine()
    with engine.connect() as conn:
        ids = {(r.title, r.year): r.id for r in conn.execute(text("SELECT id, title, year FROM movies"))}

    rows, ranks, times = [], {"raw": [], "translated": [], "translated+r": []}, {"translated": [], "translated+r": []}
    with Session(engine) as session:
        for question, title, year in QUESTIONS:
            movie_id = ids[(title, year)]
            row = {"question": question, "expected": title}
            row["raw_rank"] = rank_of(session, question, movie_id)
            ranks["raw"].append(row["raw_rank"])
            for name, reasoning in [("translated", False), ("translated+r", True)]:
                query, full, seconds = translate(question, reasoning)
                rank = rank_of(session, query, movie_id)
                ranks[name].append(rank)
                times[name].append(seconds)
                row[name] = {"translation": full, "rank": rank, "seconds": round(seconds, 1)}
            rows.append(row)
            print(f"{title:16} raw={row['raw_rank']} off={row['translated']['rank']} on={row['translated+r']['rank']}"
                  f" | off: {row['translated']['translation']['semantic_query'][:70]}", flush=True)

    print()
    summary = [summarise("raw", ranks["raw"], None)]
    summary += [summarise(n, ranks[n], times[n]) for n in ("translated", "translated+r")]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"k": K, "summary": summary, "questions": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
