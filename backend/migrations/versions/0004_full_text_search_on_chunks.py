"""full-text search on chunks: generated tsvector column and GIN index

Keyword search (Phase 17) uses PostgreSQL's built-in full-text search instead of a separate search engine.

- search_vector holds each chunk's text as a tsvector: the words reduced to their stems ("Thrillers" -> "thriller"),
  common words dropped, positions kept so phrases like "Hugh Jackman" can be matched in order.
- GENERATED ALWAYS ... STORED: PostgreSQL computes it from content on every insert or update, so ingestion does not
  need to know it exists and it can never go stale.
- The GIN index maps each stem to the chunks containing it, so a lookup does not scan every row.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-07

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0004"
down_revision: Union[str, Sequence[str], None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE rag_chunks ADD COLUMN search_vector tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    )
    op.execute("CREATE INDEX ix_rag_chunks_search_vector ON rag_chunks USING gin (search_vector)")


def downgrade() -> None:
    op.execute("DROP INDEX ix_rag_chunks_search_vector")
    op.execute("ALTER TABLE rag_chunks DROP COLUMN search_vector")
