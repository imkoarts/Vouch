"""Add durable X Activity event deduplication records.

Revision ID: 0005_x_activity_events
Revises: 0004_nullable_cost_estimates
Create Date: 2026-07-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_x_activity_events"
down_revision: str | None = "0004_nullable_cost_estimates"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "x_activity_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("event_uuid", sa.String(length=128), nullable=False),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("filtered_user_id", sa.String(length=64), nullable=True),
        sa.Column("tag", sa.String(length=200), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("source_post_id", sa.String(length=36), nullable=True),
        sa.Column("idea_id", sa.String(length=36), nullable=True),
        sa.Column("reply_candidate_id", sa.String(length=36), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("raw_event", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("error_details", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["idea_id"], ["ideas.id"], name="fk_x_activity_events_idea_id_ideas", ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["reply_candidate_id"],
            ["reply_candidates.id"],
            name="fk_x_activity_events_reply_candidate_id_reply_candidates",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["source_post_id"],
            ["source_posts.id"],
            name="fk_x_activity_events_source_post_id_source_posts",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_x_activity_events"),
        sa.UniqueConstraint("event_uuid", name="uq_x_activity_events_event_uuid"),
    )
    op.create_index(
        "ix_x_activity_events_filtered_user_id",
        "x_activity_events",
        ["filtered_user_id"],
        unique=False,
    )
    op.create_index(
        "ix_x_activity_events_idea_id",
        "x_activity_events",
        ["idea_id"],
        unique=False,
    )
    op.create_index(
        "ix_x_activity_events_reply_candidate_id",
        "x_activity_events",
        ["reply_candidate_id"],
        unique=False,
    )
    op.create_index(
        "ix_x_activity_events_source_post_id",
        "x_activity_events",
        ["source_post_id"],
        unique=False,
    )
    op.create_index(
        "ix_x_activity_events_status_received",
        "x_activity_events",
        ["status", "received_at"],
        unique=False,
    )
    op.create_index(
        "ix_x_activity_events_type_received",
        "x_activity_events",
        ["event_type", "received_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_x_activity_events_type_received", table_name="x_activity_events")
    op.drop_index("ix_x_activity_events_status_received", table_name="x_activity_events")
    op.drop_index("ix_x_activity_events_source_post_id", table_name="x_activity_events")
    op.drop_index("ix_x_activity_events_reply_candidate_id", table_name="x_activity_events")
    op.drop_index("ix_x_activity_events_idea_id", table_name="x_activity_events")
    op.drop_index("ix_x_activity_events_filtered_user_id", table_name="x_activity_events")
    op.drop_table("x_activity_events")
