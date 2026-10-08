"""Phase 28: record what the paid steps produce for the RAG test questions, so the tests can replay it for free.

For each question in QUESTIONS this runs the real query translation (one Gemini call) and embeds its semantic query
(one embedding call), and saves both in backend/tests/fixtures/rag_questions.json. backend/tests/test_rag_questions.py
then runs the real retrieval on the real database with these, and checks which films reach the model.

Run again when the embedding model, the translation prompt or the questions change. About 10 questions × (one
translation + one embedding): roughly $0.003 (estimate).

Run from the repo root (backend venv active): python scripts/record_rag_questions.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import embeddings, query_translation  # noqa: E402
from app.config import get_settings  # noqa: E402

OUT = ROOT / "backend" / "tests" / "fixtures" / "rag_questions.json"

# New questions, not taken from the evaluation set or the tuning questions (specs/28.md, proposal 4).
QUESTIONS = [
    {"id": "K1", "kind": "known", "question": "What do critics think of Whiplash?", "expected_movie": "Whiplash"},
    {"id": "K2", "kind": "known", "question": "How was the acting in Get Out received?", "expected_movie": "Get Out"},
    # "a father hunts for his kidnapped daughter" also fits Taken (2008), and the model chose it: too vague to test
    {"id": "V1", "kind": "vague", "question": "the film where a desperate father kidnaps the man he suspects took his daughter",
     "expected_movie": "Prisoners"},
    {"id": "V2", "kind": "vague", "question": "the movie about a ballerina losing her grip on reality",
     "expected_movie": "Black Swan"},
    {"id": "V3", "kind": "vague", "question": "the animated one where an old man flies his house with balloons",
     "expected_movie": "Up"},
    {"id": "V4", "kind": "vague", "question": "the thriller about a man who cannot form new memories",
     "expected_movie": "Memento"},
    {"id": "M1", "kind": "mood", "question": "something scary", "expected_genre": "Horror"},
    {"id": "N1", "kind": "no_answer", "question": "What do critics say about Parasite?", "missing_movie": "Parasite"},
    {"id": "N2", "kind": "no_answer", "question": "Is Oppenheimer worth watching?", "missing_movie": "Oppenheimer"},
]


def main() -> None:
    settings = get_settings()
    recorded = []
    for q in QUESTIONS:
        translation = query_translation.translate_query(q["question"])
        vector = embeddings.embed_query(translation.semantic_query)
        recorded.append({**q, "semantic_query": translation.semantic_query, "keywords": translation.keywords,
                         "translation_origin": translation.origin, "embedding": vector})
        print(f"{q['id']} {translation.origin:8} {translation.semantic_query[:70]!r} keywords={translation.keywords}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "embedding_model": settings.embedding_model,
        "translation_model": settings.query_translation_model or settings.openrouter_model,
        "questions": recorded,
    }), encoding="utf-8")
    print(f"saved {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
