"""embedding vector(1536) and HNSW cosine index

Implements: specs/5.md#AC-002, #AC-003

The embedding model is openai/text-embedding-3-small, which returns 1,536 numbers per text. Fixing the
column to vector(1536) makes PostgreSQL reject anything else, and an HNSW index needs a fixed dimension.

HNSW (Hierarchical Navigable Small World) is a graph of vectors linked to their near neighbours. A search
walks the graph towards the query instead of comparing it with every row, so it stays fast as the table
grows, at the cost of being approximate. vector_cosine_ops makes the index serve `embedding <=> query`
(cosine distance), the same distance we search with. Default build settings (m=16, ef_construction=64).

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-06

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0003"
down_revision: Union[str, Sequence[str], None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Fails if any stored vector has a different length, which is what we want.
    op.execute("ALTER TABLE rag_chunks ALTER COLUMN embedding TYPE vector(1536)")
    op.execute("CREATE INDEX ix_rag_chunks_embedding ON rag_chunks USING hnsw (embedding vector_cosine_ops)")


def downgrade() -> None:
    op.execute("DROP INDEX ix_rag_chunks_embedding")
    op.execute("ALTER TABLE rag_chunks ALTER COLUMN embedding TYPE vector")
