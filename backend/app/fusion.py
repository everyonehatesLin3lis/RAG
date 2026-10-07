"""Hybrid search (Phase 18): merge vector and keyword results with Reciprocal Rank Fusion.

The two searches score chunks on unrelated scales (cosine distance vs full-text rank), so their scores cannot be
added. Reciprocal Rank Fusion (RRF) uses only each chunk's *position* in each list:

    score(chunk) = sum over the lists it appears in of  1 / (k + rank)

With k = 60: rank 1 is worth 1/61, rank 10 is worth 1/70. A chunk near the top of *both* lists beats one at the top
of a single list, so evidence that both meaning and exact words agree on rises to the top, and a noisy keyword match
that vector search ignores has to compete on one list only. The larger k is, the less rank 1 counts over rank 10.

The top RETRIEVAL_TOP_K chunks of the fused list are what the model receives. With strategy "vector" the fusion is
skipped and the model gets the vector top K, which is how the two strategies are compared (Phase 18.3, 21).
"""

from dataclasses import dataclass, field, replace

from app.retrieval import RetrievedChunk


@dataclass
class FusedChunk:
    chunk: RetrievedChunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # list name -> 1-based rank in that list


@dataclass
class Selection:
    chunks: list[RetrievedChunk]  # sent to the model, in order
    fused: list[FusedChunk]  # the whole fused ranking (empty for the vector strategy)


def _merge(kept: RetrievedChunk, other: RetrievedChunk) -> RetrievedChunk:
    """The same chunk found by both searches: keep one copy with the vector distance and the keyword score."""
    return replace(
        kept,
        distance=kept.distance if kept.distance is not None else other.distance,
        keyword_score=kept.keyword_score if kept.keyword_score is not None else other.keyword_score,
    )


# Implements: specs/18.md#AC-002
def reciprocal_rank_fusion(lists: dict[str, list[RetrievedChunk]], k: int = 60) -> list[FusedChunk]:
    fused: dict[int, FusedChunk] = {}
    for name, chunks in lists.items():
        for rank, chunk in enumerate(chunks, start=1):
            entry = fused.get(chunk.id)
            if entry is None:
                fused[chunk.id] = FusedChunk(chunk=chunk, score=0.0)
                entry = fused[chunk.id]
            else:
                entry.chunk = _merge(entry.chunk, chunk)
            entry.score += 1 / (k + rank)
            entry.ranks[name] = rank
    # Highest score first. Ties (e.g. rank 1 in one list each) go to the chunk vector search ranked higher, then the
    # better keyword rank, then the lower id, so the order is reproducible. Measured on 54 questions, preferring the
    # vector rank was better than or equal to an arbitrary tie-break on every metric (README, Phase 18).
    missing = float("inf")
    return sorted(
        fused.values(),
        key=lambda f: (-f.score, f.ranks.get("vector", missing), f.ranks.get("keyword", missing), f.chunk.id),
    )


# Implements: specs/18.md#AC-003, #AC-004, #AC-005
def select(
    vector: list[RetrievedChunk], keyword: list[RetrievedChunk], strategy: str, top_k: int, k: int = 60
) -> Selection:
    if strategy == "vector":
        return Selection(chunks=vector[:top_k], fused=[])
    fused = reciprocal_rank_fusion({"vector": vector, "keyword": keyword}, k=k)
    return Selection(chunks=[f.chunk for f in fused[:top_k]], fused=fused)
