"""Add hashed local passwordless OTP challenges.

Revision ID: 0008_local_passwordless
Revises: 0007_accounts_voice
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0008_local_passwordless"
down_revision: str | None = "0007_accounts_voice"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "local_otp_challenges",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("email_hash", sa.String(length=64), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("salt", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failed_attempts", sa.Integer(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_local_otp_challenges")),
    )
    op.create_index(
        "ix_local_otp_email_created",
        "local_otp_challenges",
        ["email_hash", "created_at"],
    )
    op.create_index("ix_local_otp_expires", "local_otp_challenges", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_local_otp_expires", table_name="local_otp_challenges")
    op.drop_index("ix_local_otp_email_created", table_name="local_otp_challenges")
    op.drop_table("local_otp_challenges")
