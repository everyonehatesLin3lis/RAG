"""Experiment: does qwen/qwen3-embedding-8b retrieve better than openai/text-embedding-3-small on our chunks?

Both models are searched the same way: exact cosine similarity in numpy over all 10,540 chunks (no index),
with 30 questions whose answer movie is known. 22 describe a plot or theme without naming the film,
8 ask what critics said. A question counts as a hit@k when any of the top k chunks belongs to the expected movie.

Variants:
- openai-3-small         the production embeddings, read from rag_chunks.embedding
- qwen3-8b               Qwen vectors at full size, query sent as is
- qwen3-8b+instruct      same, but the query gets the instruction prefix Qwen's model card recommends
- qwen3-8b+instruct@1536 the previous, cut to the first 1,536 numbers and re-normalised. Qwen3-Embedding is trained
                         (Matryoshka) so a prefix of the vector still works; 1,536 is what our vector(1536) column holds.

The Qwen document vectors are cached in data/experiments/ (git-ignored), so a re-run does not pay again.
This is a small, directional comparison, not the Phase 20 evaluation.

Run from the repo root (backend venv active):
    python scripts/compare_embeddings.py --dry-run     # plan only
    python scripts/compare_embeddings.py --limit 5     # trial: embed 5 chunks with Qwen and stop
    python scripts/compare_embeddings.py               # embed the rest with Qwen (cached), then compare
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import text  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import get_engine  # noqa: E402
from app.embedding_job import plan_message  # noqa: E402
from app.embeddings import build_embedder, embed_texts  # noqa: E402

QWEN = "qwen/qwen3-embedding-8b"
QWEN_DIMS = 4096
QWEN_PRICE_PER_M = 0.01  # USD per million input tokens on OpenRouter when this was written
OPENAI_PRICE_PER_M = 0.02
CACHE = ROOT / "data" / "experiments" / "qwen3-embedding-8b.npz"
RESULTS = ROOT / "docs" / "experiments" / "embedding_comparison.json"
QWEN_INSTRUCT = "Instruct: Given a question about movies, retrieve film reviews or movie descriptions that answer it\nQuery: "
BATCH = 100

# (question, expected title, year, kind)
QUESTIONS = [
    ("a thief who plants an idea in someone's mind by entering their dreams", "Inception", 2010, "plot"),
    ("a father kidnaps and tortures the man he believes took his daughter", "Prisoners", 2013, "plot"),
    ("a cartoonist becomes obsessed with a serial killer who sends coded letters to newspapers in San Francisco", "Zodiac", 2007, "plot"),
    ("an insomniac office worker starts an underground fighting club with a soap salesman", "Fight Club", 1999, "plot"),
    ("astronauts travel through a wormhole to find a new home for humanity as crops fail on Earth", "Interstellar", 2014, "plot"),
    ("a man with short-term memory loss uses tattoos and polaroid photos to hunt his wife's killer", "Memento", 2000, "plot"),
    ("a young jazz drummer is pushed to breaking point by an abusive music teacher", "Whiplash", 2014, "plot"),
    ("a botanist stranded alone on Mars grows potatoes to survive", "The Martian", 2015, "plot"),
    ("a husband becomes the prime suspect when his wife disappears on their anniversary", "Gone Girl", 2014, "plot"),
    ("a couple have their memories of each other erased after a painful breakup", "Eternal Sunshine of the Spotless Mind", 2004, "plot"),
    ("a cynical weatherman relives the same day over and over", "Groundhog Day", 1993, "plot"),
    ("a linguist tries to communicate with aliens whose language changes how she experiences time", "Arrival", 2016, "plot"),
    ("a lonely hitman takes in a twelve-year-old girl whose family was murdered by a corrupt cop", "Léon: The Professional", 1994, "plot"),
    ("a retired assassin takes revenge on gangsters who killed the puppy his late wife gave him", "John Wick", 2014, "plot"),
    ("a Black man visits his white girlfriend's family and uncovers a horrifying secret", "Get Out", 2017, "plot"),
    ("a rat who dreams of becoming a chef in a Paris restaurant", "Ratatouille", 2007, "plot"),
    ("an old widower flies his house to South America tied to thousands of balloons", "Up", 2009, "plot"),
    ("a ballerina competing for the lead in Swan Lake slowly descends into madness", "Black Swan", 2010, "plot"),
    ("a US marshal investigates a disappearance at an island hospital for the criminally insane", "Shutter Island", 2010, "plot"),
    ("a teenager accidentally travels back to 1955 in a time machine built from a car", "Back to the Future", 1985, "plot"),
    ("a Harvard student builds a social networking website and gets sued by his best friend", "The Social Network", 2010, "plot"),
    ("a killer with a cattle gun hunts a man who took drug money he found in the Texas desert", "No Country for Old Men", 2007, "plot"),
    ("critics call it a bleak, morally complex kidnapping thriller with stunning cinematography by Roger Deakins", "Prisoners", 2013, "opinion"),
    ("reviewers said Heath Ledger's Joker was a terrifying, career-defining performance", "The Dark Knight", 2008, "opinion"),
    ("critics called this relentless desert chase with practical stunts an action masterpiece", "Mad Max: Fury Road", 2015, "opinion"),
    ("reviewers found this meticulous procedural about obsession with an unsolved murder case long but hypnotic", "Zodiac", 2007, "opinion"),
    ("praised as a fun, twisty modern take on an Agatha Christie whodunit with a wealthy family", "Knives Out", 2019, "opinion"),
    ("a gory, darkly funny Tarantino revenge western set in the slavery-era South", "Django Unchained", 2012, "opinion"),
    ("reviewers loved the neon-soaked style and the silent, intense getaway driver", "Drive", 2011, "opinion"),
    ("critics admired how the war film appears to be one continuous shot following two soldiers delivering a message", "1917", 2019, "opinion"),
]


def normalise(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def load_chunks() -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    with get_engine().connect() as conn:
        rows = conn.execute(text("SELECT id, movie_id, content, embedding::text AS e FROM rag_chunks ORDER BY id")).all()
    ids = np.array([r.id for r in rows])
    openai_vectors = np.array([json.loads(r.e) for r in rows], dtype=np.float32)
    return ids, [r.movie_id for r in rows], [r.content for r in rows], openai_vectors


def load_cache() -> dict[int, np.ndarray]:
    if not CACHE.exists():
        return {}
    data = np.load(CACHE)
    return dict(zip(data["ids"].tolist(), data["vectors"]))


def save_cache(cache: dict[int, np.ndarray]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    ids = np.array(sorted(cache))
    np.savez_compressed(CACHE, ids=ids, vectors=np.stack([cache[i] for i in ids]))


def evaluate(name, query_vectors, doc_vectors, chunk_movies, expected_ids, kinds, latency_ms=None) -> dict:
    scores = normalise(query_vectors) @ normalise(doc_vectors).T
    hits = {1: [], 5: [], 10: []}
    reciprocal_ranks, per_question = [], []
    for q, expected in enumerate(expected_ids):
        order = np.argsort(-scores[q])[:10]
        movies = [chunk_movies[i] for i in order]
        rank = next((r + 1 for r, m in enumerate(movies) if m == expected), None)
        for k in hits:
            hits[k].append(rank is not None and rank <= k)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        per_question.append(rank)

    def share(values, kind=None):
        picked = [v for v, k in zip(values, kinds) if kind is None or k == kind]
        return round(float(np.mean(picked)), 3)

    result = {
        "variant": name,
        "dimensions": int(doc_vectors.shape[1]),
        "hit_at_1": share(hits[1]),
        "hit_at_5": share(hits[5]),
        "hit_at_10": share(hits[10]),
        "mrr_at_10": share(reciprocal_ranks),
        "hit_at_5_plot": share(hits[5], "plot"),
        "hit_at_5_opinion": share(hits[5], "opinion"),
        "query_latency_ms_avg": latency_ms,
        "rank_of_expected_movie": per_question,
    }
    print(
        f"{name:24} dims={result['dimensions']:<5} hit@1={result['hit_at_1']:.2f} hit@5={result['hit_at_5']:.2f} "
        f"hit@10={result['hit_at_10']:.2f} MRR@10={result['mrr_at_10']:.2f}  "
        f"(hit@5 plot {result['hit_at_5_plot']:.2f}, opinion {result['hit_at_5_opinion']:.2f})"
    )
    return result


def timed_queries(embedder, questions, dims) -> tuple[np.ndarray, float]:
    """Embed each question in its own call, as the chat will, and time it."""
    vectors, times = [], []
    for q in questions:
        start = time.monotonic()
        vectors.append(embed_texts([q], embedder=embedder, dimensions=dims)[0])
        times.append((time.monotonic() - start) * 1000)
    return np.array(vectors, dtype=np.float32), round(float(np.median(times)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, help="embed at most N chunks with Qwen, then stop (trial)")
    args = parser.parse_args()

    settings = get_settings()
    qwen = build_embedder(settings.model_copy(update={"embedding_model": QWEN}))
    openai_embedder = build_embedder(settings)

    ids, chunk_movies, contents, openai_vectors = load_chunks()
    cache = load_cache()
    todo = [(int(i), c) for i, c in zip(ids, contents) if int(i) not in cache][: args.limit]
    tokens = sum(len(c) for _, c in todo) // 4
    print(f"Qwen plan: {plan_message(len(todo), BATCH)}, ~{tokens:,} tokens, ~${tokens / 1e6 * QWEN_PRICE_PER_M:.4f} (estimate)")
    if args.dry_run:
        return

    started = time.monotonic()
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        vectors = embed_texts([c for _, c in batch], embedder=qwen, dimensions=QWEN_DIMS)
        cache.update({i: np.array(v, dtype=np.float32) for (i, _), v in zip(batch, vectors)})
        save_cache(cache)  # after every batch, so a failure keeps finished work
        print(f"  {start + len(batch)}/{len(todo)} embedded", flush=True)
    qwen_seconds = round(time.monotonic() - started, 1)
    if args.limit:
        print(f"trial done: {len(todo)} chunks, {qwen_seconds}s, vector length {len(next(iter(cache.values())))}")
        return
    if len(cache) < len(ids):
        sys.exit(f"Qwen cache has {len(cache)} of {len(ids)} chunks; run again without --limit.")

    with get_engine().connect() as conn:
        titles = {(r.title, r.year): r.id for r in conn.execute(text("SELECT id, title, year FROM movies"))}
    expected = [titles[(title, year)] for _, title, year, _ in QUESTIONS]
    kinds = [kind for *_, kind in QUESTIONS]
    questions = [q for q, *_ in QUESTIONS]
    qwen_docs = np.stack([cache[int(i)] for i in ids])

    print(f"\n{len(QUESTIONS)} questions ({kinds.count('plot')} plot, {kinds.count('opinion')} opinion), {len(ids)} chunks\n")
    q_openai, ms_openai = timed_queries(openai_embedder, questions, settings.embedding_dimensions)
    q_qwen, ms_qwen = timed_queries(qwen, questions, QWEN_DIMS)
    q_qwen_i, ms_qwen_i = timed_queries(qwen, [QWEN_INSTRUCT + q for q in questions], QWEN_DIMS)

    results = [
        evaluate("openai-3-small", q_openai, openai_vectors, chunk_movies, expected, kinds, ms_openai),
        evaluate("qwen3-8b", q_qwen, qwen_docs, chunk_movies, expected, kinds, ms_qwen),
        evaluate("qwen3-8b+instruct", q_qwen_i, qwen_docs, chunk_movies, expected, kinds, ms_qwen_i),
        evaluate("qwen3-8b+instruct@1536", q_qwen_i[:, :1536], qwen_docs[:, :1536], chunk_movies, expected, kinds, ms_qwen_i),
    ]
    print(f"\nmedian query latency: openai {ms_openai} ms, qwen {ms_qwen} ms")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "questions": [{"question": q, "expected": t, "year": y, "kind": k} for q, t, y, k in QUESTIONS],
                "chunks": len(ids),
                "qwen_doc_embedding_seconds_this_run": qwen_seconds if todo else None,
                "price_per_million_tokens_usd": {"openai-3-small": OPENAI_PRICE_PER_M, "qwen3-8b": QWEN_PRICE_PER_M},
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"saved {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
