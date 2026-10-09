"""Phase 29: questions naming two or more films are retrieved per film, each with an equal share of the chunks.

The real retrieval on the real (rolled-back) database; translation and embedding are replayed or faked, the answer
model is left out (we check what it would be given).
"""

import json
from pathlib import Path

import pytest

from app import embeddings, query_translation, rag, retrieval, tool_calling
from app.query_translation import TranslatedQuery

WHIPLASH_VECTOR = next(q["embedding"] for q in json.loads(
    (Path(__file__).parent / "fixtures" / "rag_questions.json").read_text(encoding="utf-8"))["questions"]
    if q["id"] == "K1")


def given_to_the_model(session, monkeypatch, movies: list[str], keywords: list[str] | None = None):
    monkeypatch.setattr(query_translation, "translate_query", lambda m, h=None: TranslatedQuery(
        semantic_query="comparison of the films", keywords=keywords or [], movies=movies))
    monkeypatch.setattr(embeddings, "embed_query", lambda text: WHIPLASH_VECTOR)
    monkeypatch.setattr(tool_calling, "run_with_tools", lambda messages, tools: ("", []))
    return rag.answer_question("Compare them", session)


def test_two_named_films_get_four_chunks_each_alternating(session, monkeypatch):
    # The vector is a Whiplash question's: one search over everything would give Whiplash most of the 8 slots.
    result = given_to_the_model(session, monkeypatch, ["Whiplash", "La La Land"], keywords=["Whiplash", "La La Land"])

    films = [c.movie_title for c in result.sources]
    assert films.count("Whiplash") == 4 and films.count("La La Land") == 4
    assert films[:2] == ["Whiplash", "La La Land"]  # alternating, best of each first
    assert result.debug.per_film == ["Whiplash (2014)", "La La Land (2016)"]


def test_three_named_films_share_the_eight_slots(session, monkeypatch):
    result = given_to_the_model(session, monkeypatch, ["Whiplash", "La La Land", "Gravity"])

    films = [c.movie_title for c in result.sources]
    assert sorted(films.count(f) for f in ("Whiplash", "La La Land", "Gravity")) == [2, 3, 3]


@pytest.mark.parametrize("movies", [
    ["Whiplash"],                                   # one film: the normal search
    ["Whiplash", "The Godfather Part VII"],         # the second title is not in the data (invented by the model?)
    ["Whiplash", "'; DROP TABLE movies; --"],       # a hostile title is just a title that matches nothing
])
def test_fewer_than_two_real_films_use_the_normal_search(movies, session, monkeypatch):
    result = given_to_the_model(session, monkeypatch, movies)

    assert result.debug.per_film == []
    assert len(result.sources) == 8


def test_a_shared_title_keeps_both_films_in_one_share(session):
    found = retrieval.named_movies(session, ["Beauty and the Beast", "Up", "beauty and the beast (2017)"])

    assert found[0] == ("Beauty and the Beast", ["tt0101414", "tt2771200"])
    assert found[1][0] == "Up (2009)" and len(found) == 2  # the (2017) one is already in the first share


def test_shares_and_interleaving():
    assert rag._shares(8, 2) == [4, 4] and rag._shares(8, 3) == [3, 3, 2] and rag._shares(8, 4) == [2, 2, 2, 2]
    assert rag._interleave([[1, 2, 3], [4], [5, 6]]) == [1, 4, 5, 2, 6, 3]


def test_translation_movies_are_tidied_and_capped():
    query = TranslatedQuery(semantic_query="q", movies=[" Zodiac ", "Zodiac", "", "Heat", "Up", "Her", "Us"])

    assert query.movies == ["Zodiac", "Heat", "Up", "Her"]
