"""The three tools (Phase 9): exact answers from the database instead of from the model's memory.

"Which is rated higher, Zodiac or Prisoners?" has one correct answer, sitting in a table. Retrieval would hand the
model a few reviews and hope it infers ratings; a tool looks the numbers up. In Phase 10 the model decides when to
call a tool and with which arguments, but our code runs it, and it treats those arguments as untrusted input:

1. Validate them with a Pydantic schema before anything touches the database.
2. Build SQL only from fixed query shapes; argument values are bound parameters, never part of the SQL text.
3. Never raise for an expected problem. Return {"error": {...}} so the model can read it and recover
   (ask "which Beauty and the Beast?", or try again with a year).

Each tool is a plain function taking a database session, wrapped as a LangChain tool by `langchain_tools`.
Phase 24 moves them behind an MCP server with the same names and schemas.
"""

import json
import re
from collections.abc import Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from sqlalchemy import any_, func, literal, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.genres import KNOWN_GENRES, normalise_genre
from app.models import Movie, Review

# --- input schemas ------------------------------------------------------------------------------------
# Implements: specs/9.md#AC-004. The descriptions are what the model reads when deciding how to call a tool.

TITLE_HELP = "Exact movie title. Add the year in brackets to pick between movies with the same title, e.g. 'Beauty and the Beast (1991)'."


def _clean_title(value: str) -> str:
    return " ".join(value.split())


class FilterMoviesInput(BaseModel):
    year_min: int | None = Field(default=None, ge=1900, le=2100, description="Earliest release year to include.")
    genre: str | None = Field(default=None, description=f"One genre, from: {', '.join(KNOWN_GENRES)}.")
    rating_min: float | None = Field(default=None, ge=0, le=10, description="Minimum IMDb rating, 0-10.")
    limit: int = Field(default=10, ge=1, le=25, description="How many movies to return (best rated first).")

    @field_validator("genre")
    @classmethod
    def _known_genre(cls, value: str | None) -> str | None:
        if value is None:
            return None
        genre = normalise_genre(value)
        if genre is None:
            raise ValueError(f"unknown genre {value!r}; use one of: {', '.join(KNOWN_GENRES)}")
        return genre

    @model_validator(mode="after")
    def _at_least_one_filter(self) -> "FilterMoviesInput":
        if self.year_min is None and self.genre is None and self.rating_min is None:
            raise ValueError("give at least one of year_min, genre, rating_min")
        return self


class CompareMoviesInput(BaseModel):
    movie_a: str = Field(min_length=1, max_length=200, description=TITLE_HELP)
    movie_b: str = Field(min_length=1, max_length=200, description=TITLE_HELP)

    _clean = field_validator("movie_a", "movie_b")(classmethod(lambda cls, v: _clean_title(v)))


class RatingSummaryInput(BaseModel):
    movie: str = Field(min_length=1, max_length=200, description=TITLE_HELP)

    _clean = field_validator("movie")(classmethod(lambda cls, v: _clean_title(v)))

    @field_validator("movie")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("movie title is empty")
        return value


# --- errors ---------------------------------------------------------------------------------------------


def _error(code: str, message: str, **extra) -> dict:
    return {"error": {"code": code, "message": message, **extra}}


def _invalid(exc: ValidationError) -> dict:
    details = "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}" for e in exc.errors()
    )
    return _error("INVALID_ARGUMENTS", f"Invalid tool arguments: {details}")


def _run(schema: type[BaseModel], args: dict, body: Callable[[BaseModel], dict]) -> dict:
    """Validate first (no query on bad input), then run; database failures become a structured error."""
    try:
        parsed = schema.model_validate(args)
    except ValidationError as exc:
        return _invalid(exc)
    try:
        return body(parsed)
    except SQLAlchemyError:
        return _error("TOOL_FAILED", "The movie database could not be queried.")


# --- finding a movie by title ---------------------------------------------------------------------------

_TITLE_WITH_YEAR = re.compile(r"^(?P<title>.+?)\s*\((?P<year>\d{4})\)$")


# Implements: specs/9.md#AC-005, #AC-006, #AC-007
def resolve_movie(session: Session, name: str) -> Movie | dict:
    """Case-insensitive exact title match, optionally narrowed by '(year)'. Returns the Movie or an error dict."""
    match = _TITLE_WITH_YEAR.match(name)
    title, year = (match["title"], int(match["year"])) if match else (name, None)

    statement = select(Movie).where(func.lower(Movie.title) == title.lower())
    if year is not None:
        statement = statement.where(Movie.year == year)
    movies = session.scalars(statement.order_by(Movie.year)).all()

    if len(movies) == 1:
        return movies[0]
    if len(movies) > 1:
        return _error(
            "AMBIGUOUS_TITLE",
            f"{len(movies)} movies are titled {title!r}. Ask which one, or retry with the year in brackets.",
            candidates=[{"title": m.title, "year": m.year} for m in movies],
        )

    # icontains(autoescape=True) escapes % and _ so the text is matched literally, still as a bound parameter.
    suggestions = session.execute(
        select(Movie.title, Movie.year)
        .where(Movie.title.icontains(title, autoescape=True))
        .order_by(Movie.metadata_["imdb_votes"].as_integer().desc().nulls_last(), Movie.title)
        .limit(5)
    ).all()
    return _error(
        "MOVIE_NOT_FOUND",
        f"No movie titled {name!r} in the database.",
        suggestions=[{"title": t, "year": y} for t, y in suggestions],
    )


def _rating(movie: Movie) -> float | None:
    return float(movie.rating) if movie.rating is not None else None


# --- the tools --------------------------------------------------------------------------------------------


# Implements: specs/9.md#AC-001, #AC-004, #AC-007
def filter_movies(session: Session, **args) -> dict:
    def body(p: FilterMoviesInput) -> dict:
        conditions = []
        if p.year_min is not None:
            conditions.append(Movie.year >= p.year_min)
        if p.genre is not None:
            conditions.append(literal(p.genre) == any_(Movie.genres))  # :genre = ANY(genres)
        if p.rating_min is not None:
            conditions.append(Movie.rating >= p.rating_min)

        total = session.scalar(select(func.count()).select_from(Movie).where(*conditions))
        movies = session.scalars(
            select(Movie).where(*conditions).order_by(Movie.rating.desc().nulls_last(), Movie.title).limit(p.limit)
        ).all()
        return {
            "filters": p.model_dump(exclude={"limit"}, exclude_none=True),
            "total_matches": total,
            "returned": len(movies),
            "movies": [
                {"title": m.title, "year": m.year, "imdb_rating": _rating(m), "genres": list(m.genres)} for m in movies
            ],
        }

    return _run(FilterMoviesInput, args, body)


# Implements: specs/9.md#AC-002, #AC-004, #AC-005, #AC-006, #AC-007
def compare_movies(session: Session, **args) -> dict:
    def body(p: CompareMoviesInput) -> dict:
        found = []
        for argument, name in (("movie_a", p.movie_a), ("movie_b", p.movie_b)):
            movie = resolve_movie(session, name)
            if isinstance(movie, dict):
                movie["error"]["argument"] = argument
                return movie
            found.append(movie)

        rows = []
        for movie in found:
            review_count = session.scalar(select(func.count()).select_from(Review).where(Review.movie_id == movie.id))
            rows.append({
                "title": movie.title, "year": movie.year, "imdb_rating": _rating(movie),
                "genres": list(movie.genres), "review_count": review_count,
            })

        a, b = rows
        label = (lambda r: r["title"]) if a["title"] != b["title"] else (lambda r: f"{r['title']} ({r['year']})")
        if a["imdb_rating"] is None or b["imdb_rating"] is None:
            higher = None
        elif a["imdb_rating"] == b["imdb_rating"]:
            higher = "tie"
        else:
            higher = label(max(rows, key=lambda r: r["imdb_rating"]))
        return {"movies": rows, "higher_imdb_rating": higher}

    return _run(CompareMoviesInput, args, body)


BUCKETS = ["0-2", "2-4", "4-6", "6-8", "8-10"]


# Implements: specs/9.md#AC-003, #AC-004, #AC-005, #AC-006, #AC-007
def rating_summary(session: Session, **args) -> dict:
    def body(p: RatingSummaryInput) -> dict:
        movie = resolve_movie(session, p.movie)
        if isinstance(movie, dict):
            return movie

        scores = session.scalars(select(Review.review_rating).where(Review.movie_id == movie.id)).all()
        rated = [float(s) for s in scores if s is not None]
        distribution = dict.fromkeys(BUCKETS, 0)
        for score in rated:
            distribution[BUCKETS[min(int(score // 2), 4)]] += 1  # 10.0 goes into 8-10

        return {
            "movie": {"title": movie.title, "year": movie.year},
            "average_critic_rating": round(sum(rated) / len(rated), 2) if rated else None,
            "number_of_reviews": len(scores),
            "rated_reviews": len(rated),
            "rating_distribution": distribution,
            "unrated_reviews": len(scores) - len(rated),
            "note": f"Based on the {len(scores)} critic reviews in this dataset, not every review of the film. "
                    "Critic scores are normalised to 0-10.",
        }

    return _run(RatingSummaryInput, args, body)


# --- LangChain wrappers -----------------------------------------------------------------------------------

DESCRIPTIONS = {
    "filter_movies": (
        "Find movies in the database by minimum release year, genre and/or minimum IMDb rating (0-10). "
        "Use it for lists like 'thrillers since 2010 rated above 7.5'. Returns up to `limit` movies, best IMDb "
        "rating first, and total_matches so you know if there are more. At least one filter is required."
    ),
    "compare_movies": (
        "Compare two movies side by side: year, IMDb rating, genres and number of critic reviews in the database, "
        "and which has the higher IMDb rating. Use it for questions like 'which is rated higher, Zodiac or Prisoners?'."
    ),
    "rating_summary": (
        "Summarise critics' scores for one movie: the average critic rating (0-10), the number of reviews, and how "
        "the scores are spread across 2-point buckets. Lead with the average; mention the spread only if the user "
        "asks how opinions were divided. Counts cover the reviews in this database, not every review of the film."
    ),
}


# Implements: specs/9.md#AC-008
def langchain_tools(session: Session) -> list[StructuredTool]:
    """The three tools as LangChain tools bound to one database session. Outputs are JSON text for the model."""

    def make(name: str, fn: Callable[..., dict], schema: type[BaseModel]) -> StructuredTool:
        return StructuredTool.from_function(
            func=lambda **kwargs: json.dumps(fn(session, **kwargs), ensure_ascii=False),
            name=name,
            description=DESCRIPTIONS[name],
            args_schema=schema,
            # LangChain validates against args_schema before calling us; return our structured error, don't raise.
            handle_validation_error=lambda exc: json.dumps(_invalid(exc)),
        )

    return [
        make("filter_movies", filter_movies, FilterMoviesInput),
        make("compare_movies", compare_movies, CompareMoviesInput),
        make("rating_summary", rating_summary, RatingSummaryInput),
    ]
