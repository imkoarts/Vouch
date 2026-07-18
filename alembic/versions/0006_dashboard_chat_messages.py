"""Add durable dashboard chat history.

Revision ID: 0006_dashboard_chat_messages
Revises: 0005_x_activity_events
Create Date: 2026-07-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_dashboard_chat_messages"
down_revision: str | None = "0005_x_activity_events"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "dashboard_chat_messages",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("command", sa.String(length=100), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_dashboard_chat_messages"),
    )
    op.create_index(
        "ix_dashboard_chat_messages_created",
        "dashboard_chat_messages",
        ["created_at", "id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_dashboard_chat_messages_created",
        table_name="dashboard_chat_messages",
    )
    op.drop_table("dashboard_chat_messages")
