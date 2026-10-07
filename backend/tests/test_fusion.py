"""Spec 18 (specs/18.md) tests: Reciprocal Rank Fusion and the retrieval strategy. No database, no API calls."""

import pytest
from pydantic import ValidationError

from app import fusion, query_translation, rag, retrieval, tool_calling
from app.config import Settings, get_settings
from app.query_translation import TranslatedQuery
from app.retrieval import RetrievedChunk


def chunk(chunk_id: int, movie: str = "Film", distance: float | None = 0.3, keyword_score: float | None = None):
    return RetrievedChunk(chunk_id, f"tt{chunk_id}", movie, 2000, f"Movie: {movie}\n\nText {chunk_id}.", distance,
                          {"doc_type": "review"}, keyword_score=keyword_score)


# --- AC-002: Reciprocal Rank Fusion -----------------------------------------------------------------------


def test_ac002_rrf_scores_are_summed_reciprocal_ranks():
    vector = [chunk(1), chunk(2), chunk(3)]
    keyword = [chunk(3, distance=None, keyword_score=0.5), chunk(4, distance=None, keyword_score=0.4)]

    fused = fusion.reciprocal_rank_fusion({"vector": vector, "keyword": keyword}, k=60)

    scores = {f.chunk.id: f.score for f in fused}
    assert scores[1] == pytest.approx(1 / 61)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)  # 3rd in vector, 1st in keyword
    assert scores[4] == pytest.approx(1 / 62)
    # Found by both lists beats found by one. 2 and 4 tie (rank 2 in one list each): lower id first.
    assert [f.chunk.id for f in fused] == [3, 1, 2, 4]


def test_ac002_a_chunk_in_both_lists_appears_once_with_both_ranks_and_both_scores():
    fused = fusion.reciprocal_rank_fusion(
        {"vector": [chunk(7, distance=0.2)], "keyword": [chunk(7, distance=None, keyword_score=0.3)]}, k=60
    )

    [only] = fused
    assert only.ranks == {"vector": 1, "keyword": 1}
    assert only.chunk.distance == 0.2 and only.chunk.keyword_score == 0.3


def test_ac002_ties_are_broken_deterministically_in_favour_of_vector_rank():
    # Same score (rank 1 in one list each): the chunk vector search ranked goes first, whatever its id.
    fused = fusion.reciprocal_rank_fusion({"vector": [chunk(9)], "keyword": [chunk(5, distance=None)]}, k=60)
    assert [f.chunk.id for f in fused] == [9, 5]
    # Two keyword-only chunks with the same score: better keyword rank first, then lower id.
    fused = fusion.reciprocal_rank_fusion({"keyword": [chunk(8, distance=None)], "other": [chunk(3, distance=None)]}, k=60)
    assert [f.chunk.id for f in fused] == [8, 3]


def test_ac002_larger_k_flattens_the_rank_difference():
    small = fusion.reciprocal_rank_fusion({"vector": [chunk(1), chunk(2)]}, k=1)
    large = fusion.reciprocal_rank_fusion({"vector": [chunk(1), chunk(2)]}, k=60)
    assert small[0].score / small[1].score > large[0].score / large[1].score


# --- AC-003 / AC-005: selection ---------------------------------------------------------------------------


def test_ac003_hybrid_selects_the_top_k_of_the_fused_list():
    vector = [chunk(i) for i in range(1, 11)]
    keyword = [chunk(i, distance=None) for i in (20, 21, 1)]

    selection = fusion.select(vector, keyword, strategy="hybrid", top_k=8)

    assert len(selection.chunks) == 8
    assert selection.chunks[0].id == 1  # top of both lists
    assert 20 in [c.id for c in selection.chunks]  # a keyword-only find made it in
    assert [f.chunk.id for f in selection.fused[:8]] == [c.id for c in selection.chunks]


def test_ac005_without_keyword_results_hybrid_equals_vector():
    vector = [chunk(i) for i in range(1, 11)]

    selection = fusion.select(vector, [], strategy="hybrid", top_k=8)

    assert [c.id for c in selection.chunks] == list(range(1, 9))


def test_ac004_vector_strategy_ignores_keyword_results():
    vector = [chunk(i) for i in range(1, 11)]
    keyword = [chunk(i, distance=None) for i in (20, 21)]

    selection = fusion.select(vector, keyword, strategy="vector", top_k=8)

    assert [c.id for c in selection.chunks] == list(range(1, 9))
    assert selection.fused == []


# --- AC-004: configuration ----------------------------------------------------------------------------------


def test_ac004_strategy_and_sizes_are_configuration():
    s = Settings(_env_file=None)
    assert (s.retrieval_strategy, s.hybrid_candidates, s.rrf_k, s.retrieval_top_k) == ("hybrid", 10, 60, 8)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, retrieval_strategy="magic")


# --- AC-001 / AC-003: the pipeline runs both searches and uses the fused selection -----------------------


@pytest.fixture
def pipeline(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        query_translation, "translate_query",
        lambda message, history=None: TranslatedQuery(semantic_query="sq", keywords=["Hugh Jackman"]),
    )

    def fake_retrieve(query, session, k=None):
        calls["vector_k"] = k
        return [chunk(i, "Logan") for i in range(1, 11)]

    def fake_keyword(session, keywords, k=None):
        calls["keyword"] = (keywords, k)
        return [chunk(30, "Prisoners", distance=None, keyword_score=0.2), chunk(1, "Logan", distance=None)]

    def fake_loop(messages, tool_list):
        calls["prompt"] = messages[-1].content
        return "answer", []

    monkeypatch.setattr(retrieval, "retrieve", fake_retrieve)
    monkeypatch.setattr(retrieval, "keyword_search", fake_keyword)
    monkeypatch.setattr(tool_calling, "run_with_tools", fake_loop)
    return calls


def test_ac001_hybrid_runs_both_searches_with_ten_candidates_each(pipeline):
    rag.answer_question("movies starring Hugh Jackman", session=None)

    assert pipeline["vector_k"] == 10
    assert pipeline["keyword"] == (["Hugh Jackman"], 10)


def test_ac003_the_model_and_sources_get_the_fused_selection(pipeline):
    result = rag.answer_question("movies starring Hugh Jackman", session=None)

    ids = [c.id for c in result.sources]
    assert len(ids) == 8 and ids[0] == 1 and 30 in ids  # keyword-only Prisoners chunk was selected
    assert 'chunk_id="30"' in pipeline["prompt"]
    assert result.debug.selected_chunk_ids == ids
    assert result.debug.strategy == "hybrid"
    top = result.debug.fused[0]
    assert (top.chunk.id, top.ranks) == (1, {"vector": 1, "keyword": 2})


def test_ac004_vector_strategy_sends_the_vector_top_k(pipeline, monkeypatch):
    monkeypatch.setattr(get_settings(), "retrieval_strategy", "vector")

    result = rag.answer_question("movies starring Hugh Jackman", session=None)

    assert pipeline["vector_k"] == 8
    assert [c.id for c in result.sources] == list(range(1, 9))
    assert result.debug.fused == []


def test_ac003_the_api_shows_where_each_selected_chunk_came_from(pipeline):
    from app import main

    out = main.to_debug(rag.answer_question("movies starring Hugh Jackman", session=None).debug)

    assert out.strategy == "hybrid"
    first, second = out.fused_results[0], out.fused_results[1]
    assert (first.chunk_id, first.found_by, first.vector_rank, first.keyword_rank) == ("1", ["vector", "keyword"], 1, 2)
    assert first.fused_score == pytest.approx(1 / 61 + 1 / 62, abs=1e-6)
    assert (second.chunk_id, second.found_by, second.keyword_rank, second.selected) == ("30", ["keyword"], 1, True)
    assert sum(r.selected for r in out.fused_results) == 8
    assert out.selected_chunks == [r.chunk_id for r in out.fused_results if r.selected]
