"""Add passwordless accounts, account voice profiles, and X Activity provenance.

Revision ID: 0007_accounts_voice
Revises: 0006_dashboard_chat_messages
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0007_accounts_voice"
down_revision: str | None = "0006_dashboard_chat_messages"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "user_accounts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("auth_provider", sa.String(length=32), nullable=False),
        sa.Column("auth_subject", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("storage_key", sa.String(length=36), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_user_accounts")),
        sa.UniqueConstraint("email", name=op.f("uq_user_accounts_email")),
        sa.UniqueConstraint("storage_key", name=op.f("uq_user_accounts_storage_key")),
        sa.UniqueConstraint(
            "auth_provider", "auth_subject", name="uq_user_accounts_provider_subject"
        ),
    )
    op.create_table(
        "auth_sessions",
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("csrf_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user_accounts.id"],
            name=op.f("fk_auth_sessions_user_id_user_accounts"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("token_hash", name=op.f("pk_auth_sessions")),
    )
    op.create_index("ix_auth_sessions_user_expires", "auth_sessions", ["user_id", "expires_at"])
    op.create_table(
        "voice_profiles",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("account_type", sa.String(length=32), nullable=False),
        sa.Column("x_username", sa.String(length=50), nullable=True),
        sa.Column("language", sa.String(length=16), nullable=False),
        sa.Column("tone", sa.JSON(), nullable=False),
        sa.Column("response_preferences", sa.JSON(), nullable=False),
        sa.Column("vocabulary", sa.JSON(), nullable=False),
        sa.Column("sentence_patterns", sa.JSON(), nullable=False),
        sa.Column("humor_boundaries", sa.JSON(), nullable=False),
        sa.Column("banned_tendencies", sa.JSON(), nullable=False),
        sa.Column("guidance", sa.Text(), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("source_digest", sa.String(length=64), nullable=True),
        sa.Column("analysis_provider", sa.String(length=32), nullable=True),
        sa.Column("analysis_model", sa.String(length=200), nullable=True),
        sa.Column("analysis_metadata", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user_accounts.id"],
            name=op.f("fk_voice_profiles_user_id_user_accounts"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_voice_profiles")),
        sa.UniqueConstraint("user_id", name=op.f("uq_voice_profiles_user_id")),
    )


def downgrade() -> None:
    op.drop_table("voice_profiles")
    op.drop_index("ix_auth_sessions_user_expires", table_name="auth_sessions")
    op.drop_table("auth_sessions")
    op.drop_table("user_accounts")
