"""Phase 4: load the selected subset into PostgreSQL and build RAG chunks.

Steps (EXECUTION_PLAN.md 4.1–4.6):
1. Load data/subset/*.jsonl (produced from the Hugging Face files by select_subset.py).
2. Clean and normalise each row into a validated MovieRecord / ReviewRecord; invalid rows are skipped and counted.
3. Upsert movies and reviews.
4. Build one review document per review and one profile document per movie, chunk them, upsert rag_chunks.

Safe to re-run: rows are upserted by their IDs, chunks by metadata->>'chunk_key'. A chunk whose text is
unchanged keeps its embedding; a changed chunk has its embedding cleared so Phase 5 re-embeds only that one.
Embedding is not done here (Phase 5). No API calls.

Run from the repo root with the backend venv active: python scripts/ingest.py [--movies N]
"""

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from pydantic import ValidationError  # noqa: E402
from sqlalchemy import case, func, select, text  # noqa: E402
from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: E402

from app.db import get_engine  # noqa: E402
from app.documents import profile_chunks, review_chunks  # noqa: E402
from app.models import Movie, RagChunk, Review  # noqa: E402
from app.records import MovieRecord, ReviewRecord  # noqa: E402
from dataset_utils import normalise_score  # noqa: E402

SUBSET_DIR = ROOT / "data" / "subset"
BATCH = 1000
# TMDB bookkeeping tags, not descriptions of the film: noise for search.
IGNORED_KEYWORDS = {"aftercreditsstinger", "duringcreditsstinger"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_list(value: str | None) -> list[str]:
    """'Drama, Thriller, Crime' -> ['Drama', 'Thriller', 'Crime']"""
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def to_movie(row: dict) -> MovieRecord:
    return MovieRecord(
        id=row["tconst"],
        title=row["title"],
        year=int(row["year"]) if row.get("year") else None,
        director=row.get("directors"),
        rating=row.get("averageRating"),
        genres=split_list(row.get("genres")),
        description=row.get("overview"),
        cast=split_list(row.get("cast")),
        writers=split_list(row.get("writers")),
        keywords=[k for k in split_list(row.get("keywords")) if k not in IGNORED_KEYWORDS],
        runtime_minutes=row.get("runtime") or None,  # 0 means unknown in the source
        imdb_votes=row.get("numVotes"),
        original_language=row.get("original_language"),
        rt_slug=row["rt_slug"],
        rt_match=row["match"],
    )


def to_review(row: dict) -> ReviewRecord:
    published = date.fromisoformat(row["creationDate"][:10]) if row.get("creationDate") else None
    if published and published.year < 1900:  # the source uses 1800-01-01 as a placeholder
        published = None
    return ReviewRecord(
        id=row["reviewId"],
        movie_id=row["tconst"],
        text=row.get("reviewText") or "",
        rating=normalise_score(row.get("originalScore")),
        critic=row.get("criticName"),
        publication=row.get("publicatioName"),
        original_score=row.get("originalScore"),
        sentiment="fresh" if row.get("scoreSentiment") == 1 else "rotten",
        top_critic=bool(row.get("isTopCritic")),
        published=published,
        url=row.get("reviewUrl"),
    )


def validate(rows: list[dict], convert, skipped: Counter, label: str) -> list:
    records = []
    for row in rows:
        try:
            records.append(convert(row))
        except (ValidationError, ValueError, KeyError) as exc:
            reason = exc.errors()[0]["loc"][0] if isinstance(exc, ValidationError) else type(exc).__name__
            skipped[f"{label}: {reason}"] += 1
    return records


def batched(items: list, size: int = BATCH):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def upsert_movies(conn, movies: list[MovieRecord]) -> None:
    table = Movie.__table__
    for batch in batched(movies):
        stmt = pg_insert(table).values(
            [
                {
                    "id": m.id, "title": m.title, "year": m.year, "director": m.director, "rating": m.rating,
                    "genres": m.genres, "description": m.description, "metadata": m.extras(),
                }
                for m in batch
            ]
        )
        updated = {c: stmt.excluded[c] for c in ["title", "year", "director", "rating", "genres", "description", "metadata"]}
        conn.execute(stmt.on_conflict_do_update(index_elements=[table.c.id], set_=updated))


def upsert_reviews(conn, reviews: list[ReviewRecord]) -> None:
    table = Review.__table__
    for batch in batched(reviews):
        stmt = pg_insert(table).values(
            [
                {
                    "id": r.id, "movie_id": r.movie_id, "review_text": r.text, "review_rating": r.rating,
                    "source": r.source, "metadata": r.extras(),
                }
                for r in batch
            ]
        )
        updated = {c: stmt.excluded[c] for c in ["movie_id", "review_text", "review_rating", "source", "metadata"]}
        conn.execute(stmt.on_conflict_do_update(index_elements=[table.c.id], set_=updated))


def upsert_chunks(conn, chunks) -> None:
    table = RagChunk.__table__
    for batch in batched(chunks):
        stmt = pg_insert(table).values(
            [{"movie_id": c.movie_id, "content": c.content, "metadata": c.metadata} for c in batch]
        )
        conn.execute(
            stmt.on_conflict_do_update(
                index_elements=[text("(metadata->>'chunk_key')")],
                set_={
                    "movie_id": stmt.excluded.movie_id,
                    "content": stmt.excluded.content,
                    "metadata": stmt.excluded.metadata,
                    # Keep the embedding only if the text it was computed from is unchanged.
                    "embedding": case((table.c.content == stmt.excluded.content, table.c.embedding), else_=None),
                },
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--movies", type=int, help="only ingest the first N movies (quick trial run)")
    args = parser.parse_args()

    skipped: Counter = Counter()
    movies = validate(read_jsonl(SUBSET_DIR / "movies.jsonl"), to_movie, skipped, "movie")
    if args.movies:
        movies = movies[: args.movies]
    movies_by_id = {m.id: m for m in movies}

    review_rows = [r for r in read_jsonl(SUBSET_DIR / "reviews.jsonl") if r.get("tconst") in movies_by_id]
    reviews = validate(review_rows, to_review, skipped, "review")

    chunks = [c for m in movies for c in profile_chunks(m)]
    chunks += [c for r in reviews for c in review_chunks(movies_by_id[r.movie_id], r)]

    movie_ids = list(movies_by_id)
    with get_engine().begin() as conn:  # one transaction: all or nothing
        upsert_movies(conn, movies)
        upsert_reviews(conn, reviews)
        upsert_chunks(conn, chunks)
        # Remove reviews and chunks of these movies that are no longer in the subset.
        conn.execute(
            text("DELETE FROM reviews WHERE movie_id = ANY(:movies) AND NOT (id = ANY(:ids))"),
            {"movies": movie_ids, "ids": [r.id for r in reviews]},
        )
        conn.execute(
            text("DELETE FROM rag_chunks WHERE movie_id = ANY(:movies) AND NOT (metadata->>'chunk_key' = ANY(:keys))"),
            {"movies": movie_ids, "keys": [c.metadata["chunk_key"] for c in chunks]},
        )
        stored = conn.execute(
            select(
                select(func.count()).select_from(Movie).scalar_subquery(),
                select(func.count()).select_from(Review).scalar_subquery(),
                select(func.count()).select_from(RagChunk).scalar_subquery(),
                select(func.count()).select_from(RagChunk).where(RagChunk.embedding.is_not(None)).scalar_subquery(),
            )
        ).one()

    by_type = Counter(c.metadata["doc_type"] for c in chunks)
    split_docs = Counter(c.metadata["doc_type"] for c in chunks if c.metadata["chunk_index"] == 1)
    rated = sum(r.rating is not None for r in reviews)
    print(f"movies ingested:  {len(movies)}")
    print(f"reviews ingested: {len(reviews)} ({rated} with a 0-10 rating)")
    print(f"skipped rows:     {dict(skipped) or 'none'}")
    print(f"chunks built:     {len(chunks)} {dict(by_type)}; documents split into 2+ chunks: {dict(split_docs) or 'none'}")
    print(f"database now:     movies={stored[0]} reviews={stored[1]} chunks={stored[2]} chunks_with_embedding={stored[3]}")


if __name__ == "__main__":
    main()
