"""unique chunk key: one row per metadata->>'chunk_key' so ingestion can upsert chunks

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-05 22:56:39.355860

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE UNIQUE INDEX ux_rag_chunks_chunk_key ON rag_chunks ((metadata->>'chunk_key'))")


def downgrade() -> None:
    op.execute("DROP INDEX ux_rag_chunks_chunk_key")
