"""Add stable deduplication keys for source rows without an X post ID.

Revision ID: 0003_import_fingerprint
Revises: 0002_source_occurrences
Create Date: 2026-07-11
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003_import_fingerprint"
down_revision: str | None = "0002_source_occurrences"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "source_posts",
        sa.Column("import_fingerprint", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ux_source_posts_import_fingerprint",
        "source_posts",
        ["import_fingerprint"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ux_source_posts_import_fingerprint",
        table_name="source_posts",
    )
    op.drop_column("source_posts", "import_fingerprint")
