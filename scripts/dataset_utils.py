"""Helpers shared by dataset inspection (Phase 2) and ingestion (Phase 4).

The two sources have no shared ID. Rotten Tomatoes identifies a movie by a URL slug
("prisoners_2013", "the-social-network", "1193230-state_of_play"); TMDB-IMDB has a title,
release date and IMDb ID. We match them by turning the TMDB title into the same slug form.
"""

import re
import unicodedata

import pandas as pd

RAW_REVIEWS = "data/raw/original.csv"
RAW_MOVIES = "data/raw/TMDB  IMDB Movies Dataset.csv"

# --- slugs ---------------------------------------------------------------------------------

_YEAR_SUFFIX = re.compile(r"^(?P<base>.+)_(?P<year>(?:19|20)\d{2})$")
# Old RT page IDs look like "1193230-state_of_play". Short numbers are part of the title ("7_prisoners", "28_days_later").
_NUMERIC_PREFIX = re.compile(r"^\d{5,}-")
# An ambiguous title is accepted when the most voted movie has at least this many times the votes of the next one.
DOMINANCE_RATIO = 10


def slugify(title: str) -> str:
    """'Schindler's List' -> 'schindlers_list', 'Amélie' -> 'amelie', 'Fast & Furious' -> 'fast_and_furious'."""
    text = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode()
    text = text.lower().replace("&", " and ").replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def parse_rt_slug(slug: str) -> tuple[str, int | None]:
    """'1193230-state_of_play' -> ('state_of_play', None), 'prisoners_2013' -> ('prisoners', 2013)."""
    clean = _NUMERIC_PREFIX.sub("", slug).replace("-", "_")
    match = _YEAR_SUFFIX.match(clean)
    if match:
        return match["base"], int(match["year"])
    return clean, None


def load_movies(path: str) -> pd.DataFrame:
    """Load TMDB-IMDB, drop duplicate rows of the same IMDb title, add `year` and `slug`."""
    raw = pd.read_csv(path, low_memory=False)
    movies = raw.sort_values("numVotes", ascending=False).drop_duplicates("tconst").reset_index(drop=True)
    movies.attrs["raw_rows"] = len(raw)
    movies["year"] = pd.to_datetime(movies["release_date"], errors="coerce").dt.year
    movies["slug"] = movies["title"].fillna("").map(slugify)
    return movies


def match_rt_to_tmdb(rt_slugs: pd.Series, movies: pd.DataFrame) -> pd.DataFrame:
    """Match each Rotten Tomatoes slug to at most one TMDB movie.

    Returns one row per slug with the matched TMDB index (or None) and how it matched:
    - "slug+year": slug ends in a year and a TMDB movie has that title released within one year
    - "unique": exactly one TMDB movie has that title
    - "dominant": several share the title, but one has 10x the IMDb votes of the next (Fincher's Zodiac)
    - "ambiguous": several share the title with no clear winner; we take the most voted one (riskier)
    - "unmatched": no TMDB title fits
    """
    by_slug: dict[str, list[tuple[int, int | None, int]]] = {}
    for idx, slug, year, votes in zip(movies.index, movies["slug"], movies["year"], movies["numVotes"]):
        by_slug.setdefault(slug, []).append((idx, None if pd.isna(year) else int(year), int(votes)))

    rows = []
    for rt_slug in rt_slugs:
        base, year = parse_rt_slug(rt_slug)
        match, how = None, "unmatched"

        if year is not None:
            near = [c for c in by_slug.get(base, []) if c[1] is not None and abs(c[1] - year) <= 1]
            if near:
                match, how = max(near, key=lambda c: c[2])[0], "slug+year"
            else:
                # Titles that really end in a year, e.g. "blade_runner_2049"
                base = f"{base}_{year}"

        if match is None:
            candidates = by_slug.get(base, [])
            if len(candidates) == 1:
                match, how = candidates[0][0], "unique"
            elif len(candidates) > 1:
                first, second = sorted(candidates, key=lambda c: c[2], reverse=True)[:2]
                match = first[0]
                how = "dominant" if first[2] >= DOMINANCE_RATIO * second[2] else "ambiguous"

        rows.append({"rt_slug": rt_slug, "tmdb_index": match, "match": how})
    return pd.DataFrame(rows)


SAFE_MATCHES = ["slug+year", "unique", "dominant"]


def resolve_collisions(matches: pd.DataFrame, review_counts: pd.Series) -> pd.DataFrame:
    """Keep one RT slug per TMDB movie: the strongest match type, then the most reviews.

    Collisions happen when RT has several pages for the same title, e.g. 'dune_2021' (Villeneuve)
    and '1006364-dune' (Lynch, 1984), which both land on the most voted TMDB 'Dune'.
    """
    rank = {how: i for i, how in enumerate(SAFE_MATCHES + ["ambiguous"])}
    matched = matches.dropna(subset=["tmdb_index"]).astype({"tmdb_index": int})
    matched = matched.assign(_rank=matched["match"].map(rank), _reviews=matched["rt_slug"].map(review_counts))
    best = matched.sort_values(["_rank", "_reviews"], ascending=[True, False]).drop_duplicates("tmdb_index")
    return best.drop(columns=["_rank", "_reviews"])


# --- review scores ----------------------------------------------------------------------------

_FRACTION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$")
_LETTERS = ["F", "D-", "D", "D+", "C-", "C", "C+", "B-", "B", "B+", "A-", "A", "A+"]


def normalise_score(raw: object) -> float | None:
    """Critic scores come as '3.5/4', '63/100', 'B-minus', 'A+'. Map them to 0–10, else None."""
    if raw is None or pd.isna(raw):
        return None
    text = str(raw).strip()

    fraction = _FRACTION.match(text)
    if fraction:
        num, den = float(fraction[1]), float(fraction[2])
        if den > 0 and 0 <= num <= den:
            return round(num / den * 10, 2)
        return None

    letter = text.upper().replace(" ", "").replace("-MINUS", "-").replace("MINUS", "-").replace("-PLUS", "+").replace("PLUS", "+")
    if letter in _LETTERS:
        return round(_LETTERS.index(letter) / (len(_LETTERS) - 1) * 10, 2)
    return None


def score_format(raw: object) -> str:
    """Bucket a raw score for the inspection report: 'x/4', 'x/5', 'x/10', 'x/100', 'letter', 'missing', 'other'."""
    if raw is None or pd.isna(raw):
        return "missing"
    text = str(raw).strip()
    fraction = _FRACTION.match(text)
    if fraction:
        den = fraction[2].rstrip("0").rstrip(".") if "." in fraction[2] else fraction[2]
        return f"x/{den}"
    return "letter" if normalise_score(text) is not None else "other"


# --- language -----------------------------------------------------------------------------------

# Rotten Tomatoes is an English site, so most reviews are English and many are very short
# ("... awful film."). Instead of asking "is this English?", we ask "does it contain more
# function words from another language than English ones?". Words shared between languages
# (a, no, in, die, son...) are left out of the foreign lists.
_ENGLISH_STOPWORDS = set(
    "the and or but of to on at for with is are was were be been it its this that these those "
    "as by from not so if than then too very can will just about into over more most his her their "
    "they he she you we what which who all one has have had do does did".split()
)
_FOREIGN_STOPWORDS = set(
    # Spanish / Portuguese
    "el la los las del que y en una es por con para se su pero muy más sus como "
    "os em um é não uma ao mais dos das pelo pela "
    # French
    "le les des du et est une dans pour pas qui sur avec ce cette mais très "
    # German
    "der das und ist nicht ein eine mit zu sich auf dem den sehr "
    # Italian
    "il di che per non della è sono gli".split()
)

# Encoding damage seen in the source: "%u0161"-style escapes, U+FFFD replacement characters, HTML entities.
_BROKEN_TEXT = re.compile(r"%u[0-9A-Fa-f]{4}|�|&#\d+;|&[a-z]+;")


def is_broken_text(text: object) -> bool:
    return bool(_BROKEN_TEXT.search(str(text)))


def language_hits(text: object) -> tuple[int, int]:
    """(English stopword count, foreign stopword count)."""
    words = re.findall(r"[^\W\d_]+", str(text).lower())
    return sum(w in _ENGLISH_STOPWORDS for w in words), sum(w in _FOREIGN_STOPWORDS for w in words)


def is_english(text: object) -> bool:
    english, foreign = language_hits(text)
    return english >= foreign
