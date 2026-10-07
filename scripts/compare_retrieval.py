"""Phase 18.3: vector-only vs hybrid retrieval on the same questions (retrieval only, no answers generated).

Questions:
- 30 known-answer questions from the Phase 5 embedding experiment (22 plot, 8 critic opinion),
- 16 casual questions from the Phase 7 translation experiment,
- 8 two-film comparison questions (the case Phase 11 found one-sided).

Each question is translated once and both searches run once (vector top 10, keyword top 10). The two strategies
are then scored on the same results, so they differ only in how the 8 chunks for the model are chosen:
- vector: the vector top 8,
- hybrid: the top 8 of the Reciprocal Rank Fusion of both lists.

Metrics: hit@1 (the first chunk is the expected film), hit@8 (the expected film is among the 8 chunks), and for
two-film questions both@8 (both films are among the 8). About 54 translation calls: ~$0.02.

Run from the repo root (backend venv active): python scripts/compare_retrieval.py
Results: docs/experiments/hybrid_comparison.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app import fusion, retrieval  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import get_engine  # noqa: E402
from app.query_translation import translate_query  # noqa: E402
from compare_embeddings import QUESTIONS as KNOWN_ANSWER  # noqa: E402
from compare_query_translation import QUESTIONS as CASUAL  # noqa: E402

RESULTS = ROOT / "docs" / "experiments" / "hybrid_comparison.json"

TWO_FILM = [
    ("Compare Prisoners with Zodiac", [("Prisoners", 2013), ("Zodiac", 2007)]),
    ("Which is better, Inception or Interstellar?", [("Inception", 2010), ("Interstellar", 2014)]),
    ("Joker or The Dark Knight: which Joker performance do critics prefer?", [("Joker", 2019), ("The Dark Knight", 2008)]),
    ("How does Alien compare to Aliens?", [("Alien", 1979), ("Aliens", 1986)]),
    ("Is Gone Girl darker than Shutter Island?", [("Gone Girl", 2014), ("Shutter Island", 2010)]),
    ("Mad Max: Fury Road or John Wick for an action night?", [("Mad Max: Fury Road", 2015), ("John Wick", 2014)]),
    ("Compare Toy Story and Toy Story 3", [("Toy Story", 1995), ("Toy Story 3", 2010)]),
    ("The Godfather vs GoodFellas: which gangster film do critics rate higher?", [("The Godfather", 1972), ("GoodFellas", 1990)]),
]


def score(chunks, expected_ids: list[str]) -> dict:
    movies = [c.movie_id for c in chunks]
    first = next((i + 1 for i, m in enumerate(movies) if m in expected_ids), None)
    return {
        "hit_at_1": bool(movies) and movies[0] in expected_ids,
        "hit_at_8": first is not None,
        "all_at_8": all(e in movies for e in expected_ids),
        "first_rank": first,
        "films": len(set(movies)),
    }


def main() -> None:
    settings = get_settings()
    engine = get_engine()
    with engine.connect() as conn:
        ids = {(r.title, r.year): r.id for r in conn.execute(text("SELECT id, title, year FROM movies"))}

    questions = (
        [("known-answer", q, [(t, y)]) for q, t, y, _ in KNOWN_ANSWER]
        + [("casual", q, [(t, y)]) for q, t, y in CASUAL]
        + [("two-film", q, films) for q, films in TWO_FILM]
    )
    rows = []
    with Session(engine) as session:
        for group, question, films in questions:
            expected = [ids[f] for f in films]
            translation = translate_query(question)
            vector = retrieval.retrieve(translation.semantic_query, session, k=settings.hybrid_candidates)
            keyword = retrieval.keyword_search(session, translation.keywords, k=settings.hybrid_candidates)
            row = {"group": group, "question": question, "expected": [f"{t} ({y})" for t, y in films],
                   "keywords": translation.keywords}
            for strategy in ("vector", "hybrid"):
                chosen = fusion.select(vector, keyword, strategy, settings.retrieval_top_k, k=settings.rrf_k).chunks
                row[strategy] = score(chosen, expected)
            rows.append(row)
            v, h = row["vector"], row["hybrid"]
            mark = "" if (v["hit_at_8"], v["all_at_8"]) == (h["hit_at_8"], h["all_at_8"]) else "  <-- differs"
            print(f"{group:12} vec@1={v['hit_at_1']!s:5} hyb@1={h['hit_at_1']!s:5} vec all@8={v['all_at_8']!s:5} "
                  f"hyb all@8={h['all_at_8']!s:5} | {question[:60]}{mark}", flush=True)

    summary = {}
    for group in ("known-answer", "casual", "two-film", "all"):
        picked = [r for r in rows if group == "all" or r["group"] == group]
        summary[group] = {
            strategy: {
                "questions": len(picked),
                "hit_at_1": round(sum(r[strategy]["hit_at_1"] for r in picked) / len(picked), 3),
                "hit_at_8": round(sum(r[strategy]["hit_at_8"] for r in picked) / len(picked), 3),
                "all_at_8": round(sum(r[strategy]["all_at_8"] for r in picked) / len(picked), 3),
                "distinct_films_in_8": round(sum(r[strategy]["films"] for r in picked) / len(picked), 2),
            }
            for strategy in ("vector", "hybrid")
        }
    print()
    for group, s in summary.items():
        v, h = s["vector"], s["hybrid"]
        print(f"{group:12} n={v['questions']:2}  hit@1 {v['hit_at_1']:.2f} -> {h['hit_at_1']:.2f}   "
              f"hit@8 {v['hit_at_8']:.2f} -> {h['hit_at_8']:.2f}   all@8 {v['all_at_8']:.2f} -> {h['all_at_8']:.2f}   "
              f"films in 8: {v['distinct_films_in_8']} -> {h['distinct_films_in_8']}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"settings": {"candidates": settings.hybrid_candidates, "top_k": settings.retrieval_top_k,
                                                "rrf_k": settings.rrf_k}, "summary": summary, "questions": rows},
                                  indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
