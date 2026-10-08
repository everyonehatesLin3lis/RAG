"""Spec 28 AC-003, AC-004, AC-005: what retrieval gives the model for known, vague and no-answer questions.

The real pipeline on the real database, with two parts replayed from a recording so the tests are free and give the
same result every time: the query translation and the question's embedding, both recorded by
`scripts/record_rag_questions.py` into `tests/fixtures/rag_questions.json`. Only the answer model is left out: these
tests check the chunks the model would receive. The answers themselves are checked live (`test_live.py`).
"""

import json
from pathlib import Path

import pytest

from app import embeddings, query_translation, rag, tool_calling
from app.models import Movie
from app.query_translation import TranslatedQuery

RECORDING = json.loads((Path(__file__).parent / "fixtures" / "rag_questions.json").read_text(encoding="utf-8"))
QUESTIONS = {q["id"]: q for q in RECORDING["questions"]}


def chunks_sent_to_the_model(question: dict, session, monkeypatch):
    """Run rag.answer_question with the recorded translation and embedding; return the chunks given to the model."""
    monkeypatch.setattr(query_translation, "translate_query", lambda message, history=None: TranslatedQuery(
        semantic_query=question["semantic_query"], keywords=question["keywords"]))
    monkeypatch.setattr(embeddings, "embed_query", lambda text: question["embedding"])
    given = []
    monkeypatch.setattr(tool_calling, "run_with_tools", lambda messages, tools: given.append(messages) or ("", []))

    result = rag.answer_question(question["question"], session)

    return result.sources, given


def by_kind(kind: str) -> list[str]:
    return [i for i, q in QUESTIONS.items() if q["kind"] == kind]


def test_the_recording_matches_the_current_embedding_model():
    from app.config import get_settings

    assert RECORDING["embedding_model"] == get_settings().embedding_model, "run scripts/record_rag_questions.py again"


# Implements: specs/28.md#AC-003
@pytest.mark.parametrize("qid", by_kind("known"))
def test_known_question_retrieves_the_named_film(qid, session, monkeypatch):
    question = QUESTIONS[qid]

    sources, _ = chunks_sent_to_the_model(question, session, monkeypatch)

    assert question["expected_movie"] in {c.movie_title for c in sources}


# Implements: specs/28.md#AC-004
@pytest.mark.parametrize("qid", by_kind("vague"))
def test_vague_description_retrieves_the_intended_film(qid, session, monkeypatch):
    question = QUESTIONS[qid]
    assert question["expected_movie"].lower() not in question["question"].lower()  # really does not name it

    sources, _ = chunks_sent_to_the_model(question, session, monkeypatch)

    assert question["expected_movie"] in {c.movie_title for c in sources}


# Implements: specs/28.md#AC-004
@pytest.mark.parametrize("qid", by_kind("mood"))
def test_vague_mood_request_retrieves_mostly_films_of_that_genre(qid, session, monkeypatch):
    question = QUESTIONS[qid]

    sources, _ = chunks_sent_to_the_model(question, session, monkeypatch)

    in_genre = [c for c in sources if question["expected_genre"] in session.get(Movie, c.movie_id).genres]
    assert len(in_genre) > len(sources) / 2, [c.movie_title for c in sources]


# Implements: specs/28.md#AC-005
@pytest.mark.parametrize("qid", by_kind("no_answer"))
def test_film_not_in_the_data_retrieves_nothing_about_it_and_the_model_is_told_to_say_so(qid, session, monkeypatch):
    question = QUESTIONS[qid]
    assert session.query(Movie).filter(Movie.title.ilike(question["missing_movie"])).count() == 0

    sources, given = chunks_sent_to_the_model(question, session, monkeypatch)

    assert question["missing_movie"].lower() not in " ".join(c.movie_title.lower() for c in sources)
    assert "not enough information" in given[0][0].content.lower()  # the system rule the answer must follow
