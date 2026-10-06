"""Batch job that embeds RAG chunks (Phase 5). The command-line entry point is scripts/embed_chunks.py.

Only chunks with no embedding yet are sent, so re-running after a failure or after re-ingestion pays only
for new or changed chunks. Each batch is embedded, checked, and saved before the next one starts, so a
failure loses at most the batch in flight.
"""

import math
from collections.abc import Callable

from sqlalchemy import Connection, text

Vector = list[float]


# Implements: specs/5.md#AC-004, #AC-005
def fetch_pending(conn: Connection, limit: int | None = None) -> list[tuple[int, str]]:
    """(id, content) of chunks that still need an embedding, oldest first."""
    rows = conn.execute(
        text("SELECT id, content FROM rag_chunks WHERE embedding IS NULL ORDER BY id LIMIT :limit"),
        {"limit": limit},  # LIMIT NULL means no limit in PostgreSQL
    )
    return [(row.id, row.content) for row in rows]


# Implements: specs/5.md#AC-004
def save_embeddings(conn: Connection, rows: list[tuple[int, Vector]]) -> None:
    conn.execute(
        text("UPDATE rag_chunks SET embedding = CAST(:embedding AS vector) WHERE id = :id"),
        [{"id": chunk_id, "embedding": str(vector)} for chunk_id, vector in rows],
    )


# Implements: specs/5.md#AC-005
def api_calls_needed(chunks: int, batch_size: int) -> int:
    return math.ceil(chunks / batch_size)


def plan_message(chunks: int, batch_size: int) -> str:
    return f"{chunks} chunks to embed in {api_calls_needed(chunks, batch_size)} API calls (batches of {batch_size})"


# Implements: specs/5.md#AC-004, #AC-006
def embed_in_batches(
    chunks: list[tuple[int, str]],
    embed: Callable[[list[str]], list[Vector]],
    save: Callable[[list[tuple[int, Vector]]], None],
    batch_size: int,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Embed and save chunks batch by batch. `embed` raises on a bad response, so nothing from that batch is saved."""
    done = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embed([content for _, content in batch])
        save([(chunk_id, vector) for (chunk_id, _), vector in zip(batch, vectors)])
        done += len(batch)
        if progress:
            progress(done, len(chunks))
    return done
