"""LLM telemetry, API usage, audit events, opt-outs, and settings."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    event,
    false,
    true,
)
from sqlalchemy.orm import Mapped, Mapper, mapped_column

from app.domain.enums import ApiDirection, LLMRunStatus
from app.models.base import (
    AppendOnlyTimestampMixin,
    Base,
    TimestampMixin,
    UTCDateTime,
    new_uuid,
    utc_now,
)
from app.models.enums import portable_enum

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection


class LLMRun(AppendOnlyTimestampMixin, Base):
    """Safe metadata for one provider request; raw prompts are not stored here."""

    __tablename__ = "llm_runs"
    __table_args__ = (
        UniqueConstraint("provider", "request_id", name="uq_llm_runs_provider_request"),
        CheckConstraint("input_tokens >= 0", name="llm_runs_input_tokens_nonnegative"),
        CheckConstraint("output_tokens >= 0", name="llm_runs_output_tokens_nonnegative"),
        CheckConstraint("latency_ms >= 0", name="llm_runs_latency_nonnegative"),
        CheckConstraint("retry_count >= 0", name="llm_runs_retry_nonnegative"),
        CheckConstraint("estimated_cost_usd >= 0", name="llm_runs_cost_nonnegative"),
        Index("ix_llm_runs_provider_created", "provider", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    idea_id: Mapped[str | None] = mapped_column(
        ForeignKey("ideas.id", ondelete="SET NULL"), index=True
    )
    draft_id: Mapped[str | None] = mapped_column(
        ForeignKey("drafts.id", ondelete="SET NULL"), index=True
    )
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(255))
    operation: Mapped[str] = mapped_column(String(100), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    estimated_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(14, 6), nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[LLMRunStatus] = mapped_column(
        portable_enum(LLMRunStatus, name="llm_run_status"), nullable=False
    )
    request_payload_hash: Mapped[str | None] = mapped_column(String(64))
    actual_model: Mapped[str | None] = mapped_column(String(200))
    error_details: Mapped[str | None] = mapped_column(Text)
    run_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )


class ApiUsage(AppendOnlyTimestampMixin, Base):
    """Provider usage and estimated costs used by budget gates."""

    __tablename__ = "api_usage"
    __table_args__ = (
        CheckConstraint("units >= 0", name="api_usage_units_nonnegative"),
        CheckConstraint("estimated_cost_usd >= 0", name="api_usage_cost_nonnegative"),
        Index("ix_api_usage_provider_created", "provider", "created_at"),
        Index("ix_api_usage_direction_created", "direction", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    operation: Mapped[str] = mapped_column(String(100), nullable=False)
    direction: Mapped[ApiDirection] = mapped_column(
        portable_enum(ApiDirection, name="api_direction"), nullable=False
    )
    units: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    estimated_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(14, 6), nullable=True)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    request_id: Mapped[str | None] = mapped_column(String(255))
    success: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=true()
    )
    usage_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )


class AuditEvent(AppendOnlyTimestampMixin, Base):
    """Append-only security and state-transition audit log."""

    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_audit_events_entity", "entity_type", "entity_id"),
        Index("ix_audit_events_action_created", "action", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_id: Mapped[str | None] = mapped_column(String(64))
    actor: Mapped[str] = mapped_column(String(200), nullable=False)
    old_status: Mapped[str | None] = mapped_column(String(64))
    new_status: Mapped[str | None] = mapped_column(String(64))
    content_hash: Mapped[str | None] = mapped_column(String(64))
    event_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )


class XActivityEvent(TimestampMixin, Base):
    """Durable deduplication and processing record for one X Activity event."""

    __tablename__ = "x_activity_events"
    __table_args__ = (
        UniqueConstraint("event_uuid", name="uq_x_activity_events_event_uuid"),
        Index("ix_x_activity_events_type_received", "event_type", "received_at"),
        Index("ix_x_activity_events_status_received", "status", "received_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    event_uuid: Mapped[str] = mapped_column(String(128), nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    filtered_user_id: Mapped[str | None] = mapped_column(String(64), index=True)
    tag: Mapped[str | None] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="received")
    source_post_id: Mapped[str | None] = mapped_column(
        ForeignKey("source_posts.id", ondelete="SET NULL"), index=True
    )
    idea_id: Mapped[str | None] = mapped_column(
        ForeignKey("ideas.id", ondelete="SET NULL"), index=True
    )
    reply_candidate_id: Mapped[str | None] = mapped_column(
        ForeignKey("reply_candidates.id", ondelete="SET NULL"), index=True
    )
    received_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    processed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    raw_event: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    result_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )
    error_details: Mapped[str | None] = mapped_column(Text)


class DashboardChatMessage(TimestampMixin, Base):
    """Durable local dashboard chat entry shared by browser and desktop clients."""

    __tablename__ = "dashboard_chat_messages"
    __table_args__ = (Index("ix_dashboard_chat_messages_created", "created_at", "id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    command: Mapped[str | None] = mapped_column(String(100))
    message_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )


class OptOutUser(Base):
    """X users that must be excluded before reply generation."""

    __tablename__ = "opt_out_users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    x_user_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    username: Mapped[str | None] = mapped_column(String(64))
    reason: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(100), nullable=False, default="manual")
    opted_out_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())


class AppSetting(TimestampMixin, Base):
    """Non-secret runtime override; secret values are intentionally unsupported."""

    __tablename__ = "app_settings"
    __table_args__ = (CheckConstraint("is_secret = false", name="app_settings_no_secrets"),)

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    updated_by: Mapped[str | None] = mapped_column(String(200))
    is_secret: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )


def _prevent_observability_history_mutation(
    mapper: Mapper[Any], connection: Connection, target: object
) -> None:
    del mapper, connection
    raise RuntimeError(f"{type(target).__name__} rows are append-only")


for _append_only_model in (AuditEvent, ApiUsage, LLMRun):
    event.listen(
        _append_only_model,
        "before_update",
        _prevent_observability_history_mutation,
    )
    event.listen(
        _append_only_model,
        "before_delete",
        _prevent_observability_history_mutation,
    )
