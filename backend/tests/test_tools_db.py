"""Spec 9 (specs/9.md) tool tests against the real local database, inside a rolled-back transaction.

Test movies use years 2096–2099 and titles starting with "Testfilm", so they never mix with the real 500 movies.
"""

import pytest
from sqlalchemy import event, text

from app import tools
from app.models import Movie, Review


@pytest.fixture
def films(session):
    session.add_all(
        [
            Movie(id="tt_t1", title="Testfilm Alpha", year=2099, rating=8.5, genres=["Thriller", "Drama"]),
            Movie(id="tt_t2", title="Testfilm Beta", year=2098, rating=6.0, genres=["Comedy"]),
            Movie(id="tt_t3", title="Testfilm Gamma", year=2097, rating=7.5, genres=["Thriller"]),
            Movie(id="tt_t4", title="Testfilm Twin", year=2096, rating=7.0, genres=["Family"]),
            Movie(id="tt_t5", title="Testfilm Twin", year=2097, rating=6.5, genres=["Family"]),
        ]
    )
    session.flush()
    for i, rating in enumerate([9.0, 8.0, 7.0, 3.0, None]):
        session.add(Review(id=990_000_000 + i, movie_id="tt_t1", review_text=f"r{i}", review_rating=rating,
                           source="rotten_tomatoes"))
    session.add(Review(id=990_000_100, movie_id="tt_t2", review_text="ok", review_rating=5.0, source="rotten_tomatoes"))
    session.flush()
    return session


# --- AC-001: filter_movies ------------------------------------------------------------------------


def test_ac001_filter_applies_every_given_filter(films):
    result = tools.filter_movies(films, year_min=2097, genre="Thriller", rating_min=7.0)

    assert result["total_matches"] == 2
    assert result["movies"] == [
        {"title": "Testfilm Alpha", "year": 2099, "imdb_rating": 8.5, "genres": ["Thriller", "Drama"]},
        {"title": "Testfilm Gamma", "year": 2097, "imdb_rating": 7.5, "genres": ["Thriller"]},
    ]


def test_ac001_results_are_best_rated_first_and_capped_with_total(films):
    result = tools.filter_movies(films, year_min=2096, limit=2)

    assert result["total_matches"] == 5
    assert result["returned"] == 2
    assert [m["title"] for m in result["movies"]] == ["Testfilm Alpha", "Testfilm Gamma"]


def test_ac001_genre_is_case_insensitive_and_aliases_work(films):
    assert tools.filter_movies(films, year_min=2096, genre="thriller")["total_matches"] == 2


# --- AC-002: compare_movies -------------------------------------------------------------------------


def test_ac002_compare_returns_both_movies_and_review_counts(films):
    result = tools.compare_movies(films, movie_a="Testfilm Alpha", movie_b="testfilm beta")

    assert result["movies"] == [
        {"title": "Testfilm Alpha", "year": 2099, "imdb_rating": 8.5, "genres": ["Thriller", "Drama"], "review_count": 5},
        {"title": "Testfilm Beta", "year": 2098, "imdb_rating": 6.0, "genres": ["Comedy"], "review_count": 1},
    ]
    assert result["higher_imdb_rating"] == "Testfilm Alpha"


# --- AC-003: rating_summary --------------------------------------------------------------------------


def test_ac003_summary_has_average_count_and_distribution(films):
    result = tools.rating_summary(films, movie="Testfilm Alpha")

    assert result["movie"] == {"title": "Testfilm Alpha", "year": 2099}
    assert result["average_critic_rating"] == pytest.approx((9 + 8 + 7 + 3) / 4, abs=0.01)
    assert result["number_of_reviews"] == 5
    assert result["rated_reviews"] == 4
    assert result["rating_distribution"] == {"0-2": 0, "2-4": 1, "4-6": 0, "6-8": 1, "8-10": 2}
    assert result["unrated_reviews"] == 1
    assert "in this dataset" in result["note"]


# --- AC-005: missing movie ----------------------------------------------------------------------------


def test_ac005_unknown_title_is_movie_not_found_with_suggestions(films):
    result = tools.rating_summary(films, movie="Testfilm")

    assert result["error"]["code"] == "MOVIE_NOT_FOUND"
    assert "Testfilm Alpha" in [s["title"] for s in result["error"]["suggestions"]]
    assert len(result["error"]["suggestions"]) <= 5


def test_ac005_compare_says_which_movie_was_not_found(films):
    result = tools.compare_movies(films, movie_a="Testfilm Alpha", movie_b="No Such Film Anywhere")

    assert result["error"]["code"] == "MOVIE_NOT_FOUND"
    assert result["error"]["argument"] == "movie_b"


# --- AC-006: ambiguous title ------------------------------------------------------------------------


def test_ac006_shared_title_is_ambiguous_with_candidates(films):
    result = tools.rating_summary(films, movie="Testfilm Twin")

    assert result["error"]["code"] == "AMBIGUOUS_TITLE"
    assert result["error"]["candidates"] == [
        {"title": "Testfilm Twin", "year": 2096},
        {"title": "Testfilm Twin", "year": 2097},
    ]


def test_ac006_a_year_in_brackets_picks_one(films):
    result = tools.compare_movies(films, movie_a="Testfilm Twin (2096)", movie_b="Testfilm Twin (2097)")

    assert [m["year"] for m in result["movies"]] == [2096, 2097]


# --- AC-007: parameterised, fixed-shape SQL -------------------------------------------------------------


def test_ac007_argument_values_are_bound_parameters_not_sql_text(films):
    statements = []
    connection = films.connection()
    listener = lambda conn, cursor, statement, params, context, many: statements.append((statement, params))  # noqa: E731
    event.listen(connection, "before_cursor_execute", listener)
    try:
        evil = "x'); DROP TABLE movies; --"
        result = tools.rating_summary(films, movie=evil)
    finally:
        event.remove(connection, "before_cursor_execute", listener)

    assert result["error"]["code"] == "MOVIE_NOT_FOUND"
    assert statements, "the tool should have queried"
    assert all("DROP TABLE" not in sql for sql, _ in statements)
    assert any("DROP TABLE" in str(params) for _, params in statements)
    assert films.execute(text("SELECT count(*) FROM movies WHERE id = 'tt_t1'")).scalar() == 1


# --- AC-004 at the database boundary: invalid arguments run no query -----------------------------------


def test_ac004_invalid_arguments_run_no_query(films):
    statements = []
    connection = films.connection()
    listener = lambda conn, cursor, statement, params, context, many: statements.append(statement)  # noqa: E731
    event.listen(connection, "before_cursor_execute", listener)
    try:
        result = tools.filter_movies(films, rating_min=42)
    finally:
        event.remove(connection, "before_cursor_execute", listener)

    assert result["error"]["code"] == "INVALID_ARGUMENTS"
    assert statements == []
