"""Account authentication and account-bound voice profile persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UTCDateTime, new_uuid


class UserAccount(TimestampMixin, Base):
    """One authenticated operator; email is runtime data, never release configuration."""

    __tablename__ = "user_accounts"
    __table_args__ = (
        UniqueConstraint("auth_provider", "auth_subject", name="uq_user_accounts_provider_subject"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    auth_provider: Mapped[str] = mapped_column(String(32), nullable=False)
    auth_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    storage_key: Mapped[str] = mapped_column(
        String(36), nullable=False, unique=True, default=new_uuid
    )

    voice_profile: Mapped[VoiceProfile | None] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )


class AuthSession(Base):
    """Hashed, revocable application session created after provider OTP verification."""

    __tablename__ = "auth_sessions"
    __table_args__ = (Index("ix_auth_sessions_user_expires", "user_id", "expires_at"),)

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("user_accounts.id", ondelete="CASCADE"), nullable=False
    )
    csrf_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime())


class LocalOtpChallenge(Base):
    """Short-lived hashed OTP for the loopback-only local passwordless flow."""

    __tablename__ = "local_otp_challenges"
    __table_args__ = (
        Index("ix_local_otp_email_created", "email_hash", "created_at"),
        Index("ix_local_otp_expires", "expires_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    email_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    salt: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    consumed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    failed_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class VoiceProfile(TimestampMixin, Base):
    """Structured writing preferences owned by exactly one authenticated account."""

    __tablename__ = "voice_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("user_accounts.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    account_type: Mapped[str] = mapped_column(String(32), nullable=False, default="personal")
    x_username: Mapped[str | None] = mapped_column(String(50))
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="en")
    tone: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    response_preferences: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    vocabulary: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    sentence_patterns: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    humor_boundaries: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    banned_tendencies: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    guidance: Mapped[str] = mapped_column(Text, nullable=False, default="")
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_digest: Mapped[str | None] = mapped_column(String(64))
    analysis_provider: Mapped[str | None] = mapped_column(String(32))
    analysis_model: Mapped[str | None] = mapped_column(String(200))
    analysis_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    user: Mapped[UserAccount] = relationship(back_populates="voice_profile")
