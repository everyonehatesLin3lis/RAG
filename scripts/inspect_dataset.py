"""Phase 2: inspect the raw review and movie files before building anything on them.

Checks from EXECUTION_PLAN.md 2.2: number of records, fields, missing values, review lengths,
duplicate movies, languages, rating formats. Plus how well the two sources join.

Run from the repo root: python scripts/inspect_dataset.py
Writes a summary to docs/dataset_inspection.json.
"""

import json
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
    score_format,
)

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "docs" / "dataset_inspection.json"
PLAN_EXAMPLES = ["Prisoners", "Zodiac", "Inception"]


def pct(part: float, whole: float) -> float:
    return round(100 * part / whole, 2) if whole else 0.0


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * (60 - len(title))}")


def inspect_reviews(reviews: pd.DataFrame) -> dict:
    section("REVIEWS (contemmcm/rotten_tomatoes)")
    n = len(reviews)
    text = reviews["reviewText"].fillna("").str.strip()
    lengths = text.str.len()
    words = text.str.split().str.len()

    summary = {
        "rows": n,
        "columns": list(reviews.columns),
        "movies": int(reviews["id"].nunique()),
        "missing_pct": {c: pct(reviews[c].isna().sum(), n) for c in reviews.columns},
        "empty_text": int((text == "").sum()),
        "duplicate_review_ids": int(reviews["reviewId"].duplicated().sum()),
        "duplicate_movie_text_pairs": int(reviews.assign(t=text).duplicated(["id", "t"]).sum()),
        "length_chars": lengths[text != ""].describe(percentiles=[0.5, 0.9, 0.99]).round(0).to_dict(),
        "length_words": words[text != ""].describe(percentiles=[0.5, 0.9, 0.99]).round(0).to_dict(),
        "over_500_chars": int((lengths > 500).sum()),
        "non_english": int((~text.map(is_english)).sum()),
        "broken_text": int(text.map(is_broken_text).sum()),
        "score_formats": reviews["originalScore"].map(score_format).value_counts().to_dict(),
        "score_normalisable_pct": pct(reviews["originalScore"].map(normalise_score).notna().sum(), n),
        "fresh_pct": pct((reviews["scoreSentiment"] == 1).sum(), n),
        "top_critic_pct": pct(reviews["isTopCritic"].sum(), n),
        "date_range": [str(reviews["creationDate"].min()), str(reviews["creationDate"].max())],
        "reviews_per_movie": reviews["id"].value_counts().describe(percentiles=[0.5, 0.9]).round(1).to_dict(),
    }
    for key, value in summary.items():
        if key != "columns":
            print(f"{key:28} {value}")
    print(f"{'columns':28} {summary['columns']}")
    return summary


def inspect_movies(movies: pd.DataFrame) -> dict:
    section("MOVIES (HenryWaltson/TMDB-IMDB-Movies-Dataset)")
    n = len(movies)
    needed = ["title", "release_date", "genres", "directors", "overview", "averageRating", "numVotes", "cast", "keywords", "tconst"]
    summary = {
        "raw_rows": movies.attrs["raw_rows"],
        "rows_after_dropping_duplicate_imdb_ids": n,
        "columns": list(movies.columns),
        "missing_pct_needed_fields": {c: pct(movies[c].isna().sum(), n) for c in needed},
        "duplicate_title_year": int(movies.duplicated(["slug", "year"]).sum()),
        "original_language_top": movies["original_language"].value_counts().head(8).to_dict(),
        "adult": int(movies["adult"].sum()),
        "imdb_rating": movies["averageRating"].describe().round(2).to_dict(),
        "num_votes": movies["numVotes"].describe(percentiles=[0.5, 0.9, 0.99]).round(0).to_dict(),
        "movies_with_50k_votes": int((movies["numVotes"] >= 50_000).sum()),
        "genres_example": movies.loc[movies["numVotes"].idxmax(), "genres"],
    }
    for key, value in summary.items():
        if key != "columns":
            print(f"{key:28} {value}")
    print(f"{'columns':28} {summary['columns']}")
    return summary


def inspect_join(reviews: pd.DataFrame, movies: pd.DataFrame) -> dict:
    section("JOIN (Rotten Tomatoes slug -> TMDB title + year)")
    review_counts = reviews["id"].value_counts()
    matches = match_rt_to_tmdb(pd.Series(review_counts.index), movies)
    matches["reviews"] = matches["rt_slug"].map(review_counts)

    by_type = matches.groupby("match").agg(movies=("rt_slug", "size"), reviews=("reviews", "sum"))
    by_type["reviews_pct"] = (100 * by_type["reviews"] / by_type["reviews"].sum()).round(2)
    print(by_type.to_string())

    # Several RT slugs can land on the same TMDB movie (e.g. a numeric-prefix duplicate page).
    matched = matches.dropna(subset=["tmdb_index"]).astype({"tmdb_index": int})
    collisions = int(matched["tmdb_index"].duplicated().sum())
    print(f"\nRT slugs sharing a TMDB movie: {collisions}")

    print("\nPlan examples:")
    examples = {}
    for title in PLAN_EXAMPLES:
        rows = matched[matched["tmdb_index"].map(movies["title"]) == title]
        found = [
            {
                "rt_slug": r.rt_slug,
                "match": r.match,
                "reviews": int(r.reviews),
                "year": int(movies.at[r.tmdb_index, "year"]),
                "director": movies.at[r.tmdb_index, "directors"],
            }
            for r in rows.itertuples()
        ]
        examples[title] = found
        print(f"  {title:10} {found}")

    return {
        "by_match_type": by_type.reset_index().to_dict(orient="records"),
        "rt_slugs_sharing_tmdb_movie": collisions,
        "plan_examples": examples,
        "_matched": matched,  # used by subset_options, not written to JSON
    }


def subset_options(reviews: pd.DataFrame, movies: pd.DataFrame, matched: pd.DataFrame) -> list[dict]:
    """How many movies/reviews a first subset could have under a few popularity thresholds."""
    section("SUBSET OPTIONS (English movies, safe matches, clean English reviews)")
    best = resolve_collisions(matched, reviews["id"].value_counts())
    safe = best[best["match"].isin(SAFE_MATCHES)]
    safe = safe.assign(
        votes=safe["tmdb_index"].map(movies["numVotes"]),
        lang=safe["tmdb_index"].map(movies["original_language"]),
    )
    safe = safe[safe["lang"] == "en"]

    text = reviews["reviewText"].fillna("").str.strip()
    clean = reviews[(text != "") & text.map(is_english) & ~text.map(is_broken_text)]
    clean_counts = clean.drop_duplicates(["id", "reviewText"])["id"].value_counts()
    safe = safe.assign(clean_reviews=safe["rt_slug"].map(clean_counts).fillna(0).astype(int))

    options = []
    for min_votes in [200_000, 100_000, 50_000, 25_000]:
        pool = safe[(safe["votes"] >= min_votes) & (safe["clean_reviews"] >= 10)]
        row = {
            "min_imdb_votes": min_votes,
            "movies": len(pool),
            "reviews_available": int(pool["clean_reviews"].sum()),
            "reviews_if_cap_20_per_movie": int(pool["clean_reviews"].clip(upper=20).sum()),
        }
        options.append(row)
        print(row)
    return options


def main() -> None:
    reviews = pd.read_csv(ROOT / RAW_REVIEWS)
    movies = load_movies(ROOT / RAW_MOVIES)

    summary = {"reviews": inspect_reviews(reviews), "movies": inspect_movies(movies)}
    join = inspect_join(reviews, movies)
    matched = join.pop("_matched")
    summary["join"] = join
    summary["subset_options"] = subset_options(reviews, movies, matched)

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSummary written to {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
