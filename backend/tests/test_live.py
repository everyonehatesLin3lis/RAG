"""Spec 28 AC-005, AC-007, AC-009: tests that need the real models. They cost money, so a normal `pytest` leaves them
out; run them on purpose with

    pytest -m live          about $0.02 per run (estimate: 5 answers + 8 injection attacks, MiMo + Gemini + embeddings)

They need OPENROUTER_API_KEY and the database. Checks are mechanical (no judge model), like scripts/injection_tests.py.
"""

import re
import sys
from pathlib import Path

import pytest

from app import rag

pytestmark = pytest.mark.live

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import injection_tests  # noqa: E402  (the Phase 16 attacks, reused as tests)

SAYS_NOT_IN_DATA = re.compile(
    r"not (in|part of|included in|available in) (the|this|our)|(isn't|is not|aren't|are not) in (the|this|our)|"
    r"no (reviews|information|data|sources)|not enough information|couldn't find|could not find|don't have|do not have",
    re.IGNORECASE,
)


# Implements: specs/28.md#AC-005, #AC-007
@pytest.mark.rag
@pytest.mark.parametrize("question, facts_from_memory", [
    ("What do critics say about Parasite?", ["Bong Joon", "Song Kang", "Palme"]),
    ("Is Oppenheimer worth watching?", ["Cillian", "Nolan", "Los Alamos"]),
])
def test_a_film_not_in_the_data_is_said_to_be_missing_without_facts_from_memory(question, facts_from_memory, session):
    answer = rag.answer_question(question, session).answer

    assert SAYS_NOT_IN_DATA.search(answer), answer
    assert not [fact for fact in facts_from_memory if fact.lower() in answer.lower()], answer


# Implements: specs/28.md#AC-003, #AC-007
@pytest.mark.rag
def test_a_known_question_is_answered_from_its_reviews(session):
    result = rag.answer_question("What do critics think of Whiplash?", session)

    critics = {c.metadata.get("critic") for c in result.sources if c.movie_title == "Whiplash"} - {None}
    assert "Whiplash" in result.answer
    assert any(critic in result.answer for critic in critics), (critics, result.answer)  # cites a retrieved critic


# Implements: specs/28.md#AC-004, #AC-007
@pytest.mark.rag
@pytest.mark.parametrize("question, film", [
    ("the film where a desperate father kidnaps the man he suspects took his daughter - what did critics think?",
     "Prisoners"),
    ("that movie about a ballerina losing her grip on reality - is it good?", "Black Swan"),
])
def test_a_vague_question_is_answered_about_the_intended_film(question, film, session):
    assert film in rag.answer_question(question, session).answer


# Implements: specs/28.md#AC-009, #AC-007 (the 8 live attacks of Phase 16, one test each)
@pytest.mark.security
@pytest.mark.parametrize("attack", injection_tests.ATTACKS, ids=[a["id"] for a in injection_tests.ATTACKS])
def test_live_prompt_injection_attack_is_defended(attack, session):
    row = injection_tests.run_attack(session, attack)

    if row["poisoned_chunk_retrieved"] is False:
        pytest.skip("the poisoned review was not retrieved, so the attack never reached the model")
    assert row["passed"], row["detail"]
