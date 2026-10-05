"""Phase 2.3: select the first working subset (about 10,000 English reviews).

Rules:
- Movies: English-language, matched safely to a Rotten Tomatoes page (see dataset_utils.match_rt_to_tmdb),
  the most voted on IMDb first, so test questions about well-known films have data.
- Reviews: non-empty, English, no encoding damage, no exact duplicates. Per movie we prefer reviews
  with a usable score (needed by rating_summary), then top critics, then longer text.

Output (JSON Lines, read by Phase 4 ingestion): data/subset/movies.jsonl, data/subset/reviews.jsonl
Run from the repo root: python scripts/select_subset.py
"""

from pathlib import Path

import pandas as pd

from dataset_utils import (
    RAW_MOVIES,
    RAW_REVIEWS,
    SAFE_MATCHES,
    is_broken_text,
    is_english,
    load_movies,
    match_rt_to_tmdb,
    normalise_score,
    resolve_collisions,
)

N_MOVIES = 500
REVIEWS_PER_MOVIE = 20
MIN_REVIEWS_PER_MOVIE = 10
MUST_INCLUDE = ["Prisoners", "Zodiac", "Inception"]  # movies used as examples in the plan

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "subset"

MOVIE_COLUMNS = [
    "tconst", "title", "year", "directors", "writers", "cast", "genres", "overview", "keywords",
    "runtime", "averageRating", "numVotes", "original_language", "rt_slug", "match",
]
REVIEW_COLUMNS = [
    "reviewId", "rt_slug", "tconst", "creationDate", "criticName", "isTopCritic", "publicatioName",
    "originalScore", "scoreSentiment", "reviewText", "reviewUrl",
]


def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    reviews = reviews.assign(reviewText=reviews["reviewText"].fillna("").str.strip())
    reviews = reviews[reviews["reviewText"] != ""]
    reviews = reviews.drop_duplicates("reviewId").drop_duplicates(["id", "reviewText"])
    reviews = reviews[reviews["reviewText"].map(is_english) & ~reviews["reviewText"].map(is_broken_text)]
    return reviews.rename(columns={"id": "rt_slug"})


def main() -> None:
    reviews = clean_reviews(pd.read_csv(ROOT / RAW_REVIEWS))
    movies = load_movies(ROOT / RAW_MOVIES)

    counts = reviews["rt_slug"].value_counts()
    matches = resolve_collisions(match_rt_to_tmdb(pd.Series(counts.index), movies), counts)
    matches = matches[matches["match"].isin(SAFE_MATCHES)]

    candidates = movies.loc[matches["tmdb_index"]].assign(
        rt_slug=matches["rt_slug"].values, match=matches["match"].values
    )
    candidates = candidates[
        (candidates["original_language"] == "en")
        & (candidates["rt_slug"].map(counts) >= MIN_REVIEWS_PER_MOVIE)
    ]
    chosen = candidates.sort_values("numVotes", ascending=False).head(N_MOVIES)

    missing = [t for t in MUST_INCLUDE if t not in set(chosen["title"])]
    if missing:
        raise SystemExit(f"Plan example movies missing from the subset: {missing}")

    picked = reviews[reviews["rt_slug"].isin(chosen["rt_slug"])].copy()
    picked["has_score"] = picked["originalScore"].map(normalise_score).notna()
    picked["length"] = picked["reviewText"].str.len()
    picked = (
        picked.sort_values(["has_score", "isTopCritic", "length"], ascending=False)
        .groupby("rt_slug")
        .head(REVIEWS_PER_MOVIE)
    )
    picked["tconst"] = picked["rt_slug"].map(chosen.set_index("rt_slug")["tconst"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chosen = chosen.assign(year=chosen["year"].astype("Int64"))
    chosen[MOVIE_COLUMNS].to_json(OUT_DIR / "movies.jsonl", orient="records", lines=True, force_ascii=False)
    picked.sort_values(["tconst", "reviewId"])[REVIEW_COLUMNS].to_json(
        OUT_DIR / "reviews.jsonl", orient="records", lines=True, force_ascii=False
    )

    print(f"movies:  {len(chosen)}  (IMDb votes {int(chosen['numVotes'].min()):,}–{int(chosen['numVotes'].max()):,})")
    print(f"reviews: {len(picked)}  (with usable score: {picked['has_score'].mean():.0%}, top critics: {picked['isTopCritic'].mean():.0%})")
    print(f"match types: {chosen['match'].value_counts().to_dict()}")
    print(f"years: {int(chosen['year'].min())}–{int(chosen['year'].max())}")
    print(f"written to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
