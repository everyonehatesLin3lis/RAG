"""Spec 28 AC-006: each of the four tools with valid parameters, invalid parameters, a missing movie and a duplicate
title, on the real (rolled-back) database. The data really has Beauty and the Beast twice (1991 and 2017)."""

import pytest

from app import tools

VALID = {
    "filter_movies": ({"genre": "thriller", "year_min": 2010, "rating_min": 7.5}, "total_matches"),
    "compare_movies": ({"movie_a": "Zodiac", "movie_b": "prisoners"}, "higher_imdb_rating"),
    "rating_summary": ({"movie": "Prisoners"}, "average_critic_rating"),
    "get_movie_metadata": ({"movie": "Prisoners"}, "director"),
}
INVALID = {
    "filter_movies": [{}, {"rating_min": 11}, {"genre": "Cyberpunk"}, {"year_min": "last year"}, {"limit": 0}],
    "compare_movies": [{}, {"movie_a": "Zodiac"}, {"movie_a": "", "movie_b": "Zodiac"}, {"movie_a": 7, "movie_b": "Zodiac"}],
    "rating_summary": [{}, {"movie": ""}, {"movie": "   "}, {"movie": None}],
    "get_movie_metadata": [{}, {"movie": ""}, {"movie": ["Zodiac"]}],
}
TITLE_TOOLS = {  # how each title tool takes one title
    "compare_movies": lambda title: {"movie_a": title, "movie_b": "Zodiac"},
    "rating_summary": lambda title: {"movie": title},
    "get_movie_metadata": lambda title: {"movie": title},
}


# Implements: specs/28.md#AC-006 (valid parameters)
@pytest.mark.parametrize("name", list(VALID))
def test_valid_parameters_give_a_result(name, session):
    args, key = VALID[name]

    result = tools.run_tool(session, name, args)

    assert "error" not in result and result[key] is not None


# Implements: specs/28.md#AC-006 (invalid parameters)
@pytest.mark.parametrize("name, args", [(n, a) for n, cases in INVALID.items() for a in cases])
def test_invalid_parameters_are_invalid_arguments(name, args, session):
    assert tools.run_tool(session, name, args)["error"]["code"] == "INVALID_ARGUMENTS"


# Implements: specs/28.md#AC-006 (missing movie)
@pytest.mark.parametrize("name", list(TITLE_TOOLS))
def test_a_missing_movie_is_movie_not_found_with_suggestions(name, session):
    result = tools.run_tool(session, name, TITLE_TOOLS[name]("Dark Knight"))

    assert result["error"]["code"] == "MOVIE_NOT_FOUND"
    assert {"title": "The Dark Knight", "year": 2008} in result["error"]["suggestions"]


# Implements: specs/28.md#AC-006 (duplicate titles)
@pytest.mark.parametrize("name", list(TITLE_TOOLS))
def test_a_duplicate_title_is_ambiguous_until_the_year_is_given(name, session):
    ambiguous = tools.run_tool(session, name, TITLE_TOOLS[name]("Beauty and the Beast"))
    resolved = tools.run_tool(session, name, TITLE_TOOLS[name]("Beauty and the Beast (2017)"))

    assert ambiguous["error"]["code"] == "AMBIGUOUS_TITLE"
    assert [c["year"] for c in ambiguous["error"]["candidates"]] == [1991, 2017]
    assert "error" not in resolved
