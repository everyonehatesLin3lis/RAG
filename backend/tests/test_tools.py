"""Spec 9 (specs/9.md) unit tests: input schemas and LangChain wrappers. No database, no API calls."""

import json

import pytest
from pydantic import ValidationError

from app import tools
from app.tools import CompareMoviesInput, FilterMoviesInput, RatingSummaryInput


class NoDatabase:
    """A session that fails the test if a tool touches the database."""

    def execute(self, *args, **kwargs):
        raise AssertionError("the tool queried the database with invalid arguments")

    scalars = scalar = execute


# --- AC-004: Pydantic schemas validate before anything runs -------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        {},                                    # at least one filter required
        {"rating_min": 11},                    # out of range
        {"year_min": 1500},                    # out of range
        {"genre": "Cyberpunk"},                # not a genre in our data
        {"genre": "Thriller", "limit": 100},   # limit capped at 25
    ],
)
def test_ac004_filter_schema_rejects_bad_arguments(args):
    with pytest.raises(ValidationError):
        FilterMoviesInput(**args)


def test_ac004_filter_schema_normalises_genre_and_defaults_limit():
    parsed = FilterMoviesInput(genre="sci-fi")
    assert parsed.genre == "Science Fiction"
    assert parsed.limit == 10


@pytest.mark.parametrize("schema, args", [
    (CompareMoviesInput, {"movie_a": "", "movie_b": "Zodiac"}),
    (CompareMoviesInput, {"movie_a": "Zodiac"}),
    (RatingSummaryInput, {"movie": "   "}),
    (RatingSummaryInput, {"movie": "x" * 201}),
])
def test_ac004_title_schemas_reject_bad_arguments(schema, args):
    with pytest.raises(ValidationError):
        schema(**args)


@pytest.mark.parametrize(
    "call",
    [
        lambda s: tools.filter_movies(s, genre="Cyberpunk"),
        lambda s: tools.compare_movies(s, movie_a="", movie_b="Zodiac"),
        lambda s: tools.rating_summary(s, movie=""),
    ],
)
def test_ac004_invalid_arguments_return_a_structured_error_without_querying(call):
    result = call(NoDatabase())
    assert result["error"]["code"] == "INVALID_ARGUMENTS"
    assert result["error"]["message"]


# --- AC-008: LangChain tools with the plan's names, descriptions and schemas ---------------------------


def test_ac008_langchain_tools_have_plan_names_and_schemas():
    built = {t.name: t for t in tools.langchain_tools(NoDatabase())}

    assert set(built) == {"filter_movies", "compare_movies", "rating_summary"}
    assert built["filter_movies"].args_schema is FilterMoviesInput
    assert built["compare_movies"].args_schema is CompareMoviesInput
    assert built["rating_summary"].args_schema is RatingSummaryInput
    for tool in built.values():
        assert len(tool.description) > 40


def test_ac008_rating_summary_description_says_lead_with_the_average():
    built = {t.name: t for t in tools.langchain_tools(NoDatabase())}
    assert "average" in built["rating_summary"].description.lower()


def test_ac008_invalid_arguments_through_langchain_return_the_structured_error():
    built = {t.name: t for t in tools.langchain_tools(NoDatabase())}

    output = built["filter_movies"].invoke({"rating_min": 42})

    assert json.loads(output)["error"]["code"] == "INVALID_ARGUMENTS"
