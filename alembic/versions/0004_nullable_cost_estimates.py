"""Preserve unknown cost estimates as SQL NULL instead of synthetic zero.

Revision ID: 0004_nullable_cost_estimates
Revises: 0003_import_fingerprint
Create Date: 2026-07-11
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004_nullable_cost_estimates"
down_revision: str | None = "0003_import_fingerprint"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("llm_runs", "api_usage", "publications")


def upgrade() -> None:
    for table_name in _TABLES:
        with op.batch_alter_table(table_name, recreate="always") as batch_op:
            batch_op.alter_column(
                "estimated_cost_usd",
                existing_type=sa.Numeric(14, 6),
                nullable=True,
            )


def downgrade() -> None:
    connection = op.get_bind()
    for table_name in _TABLES:
        unknown_count = connection.scalar(
            sa.text(
                f"SELECT COUNT(*) FROM {table_name} "  # noqa: S608
                "WHERE estimated_cost_usd IS NULL"
            )
        )
        if unknown_count:
            raise RuntimeError(
                "Downgrade from 0004_nullable_cost_estimates is blocked: "
                f"{table_name} contains unknown cost estimates. Back up the "
                "database and either retain this revision or explicitly assign "
                "operator-approved estimates before retrying. Unknown prices "
                "will not be rewritten as zero automatically."
            )

    for table_name in _TABLES:
        with op.batch_alter_table(table_name, recreate="always") as batch_op:
            batch_op.alter_column(
                "estimated_cost_usd",
                existing_type=sa.Numeric(14, 6),
                nullable=False,
            )
