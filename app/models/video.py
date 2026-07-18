"""Approved video scripts and external rendering jobs."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UTCDateTime, new_uuid
from app.models.enums import (
    VideoGeneratorMode,
    VideoJobStatus,
    VideoScriptStatus,
    portable_enum,
)

if TYPE_CHECKING:
    from app.models.drafts import Draft, DraftVersion


class VideoScript(TimestampMixin, Base):
    """A versioned structured script whose approval is hash-scoped."""

    __tablename__ = "video_scripts"
    __table_args__ = (
        UniqueConstraint("draft_id", "script_hash", name="uq_video_scripts_draft_hash"),
        Index("ix_video_scripts_status_updated", "status", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    draft_id: Mapped[str] = mapped_column(
        ForeignKey("drafts.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    draft_version_id: Mapped[str] = mapped_column(
        ForeignKey("draft_versions.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    script_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    script_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[VideoScriptStatus] = mapped_column(
        portable_enum(VideoScriptStatus, name="video_script_status"),
        nullable=False,
        default=VideoScriptStatus.GENERATED,
    )
    approved_hash: Mapped[str | None] = mapped_column(String(64))
    approved_by: Mapped[str | None] = mapped_column(String(200))
    approved_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    rejection_reason: Mapped[str | None] = mapped_column(Text)

    draft: Mapped[Draft] = relationship(back_populates="video_scripts")
    draft_version: Mapped[DraftVersion] = relationship()
    jobs: Mapped[list[VideoJob]] = relationship(back_populates="video_script", passive_deletes=True)


class VideoJob(TimestampMixin, Base):
    """A HeyGen API job or a manual-plugin handoff request."""

    __tablename__ = "video_jobs"
    __table_args__ = (
        UniqueConstraint("video_script_id", "request_hash", name="uq_video_jobs_script_request"),
        UniqueConstraint("provider", "external_job_id", name="uq_video_jobs_provider_external"),
        CheckConstraint("file_size >= 0", name="video_jobs_file_size_nonnegative"),
        Index("ix_video_jobs_status_updated", "status", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    video_script_id: Mapped[str] = mapped_column(
        ForeignKey("video_scripts.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    mode: Mapped[VideoGeneratorMode] = mapped_column(
        portable_enum(VideoGeneratorMode, name="video_generator_mode"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    external_job_id: Mapped[str | None] = mapped_column(String(200))
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[VideoJobStatus] = mapped_column(
        portable_enum(VideoJobStatus, name="video_job_status"),
        nullable=False,
        default=VideoJobStatus.REQUESTED,
    )
    response_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    local_path: Mapped[str | None] = mapped_column(String(2048))
    mime_type: Mapped[str | None] = mapped_column(String(255))
    file_size: Mapped[int | None] = mapped_column(Integer)
    thumbnail_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    error_details: Mapped[str | None] = mapped_column(Text)

    video_script: Mapped[VideoScript] = relationship(back_populates="jobs")
