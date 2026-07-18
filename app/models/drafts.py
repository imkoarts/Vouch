"""Draft aggregate, immutable versions, and immutable approval events."""

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
    event,
)
from sqlalchemy.orm import Mapped, Mapper, mapped_column, relationship

from app.models.base import (
    AppendOnlyTimestampMixin,
    Base,
    TimestampMixin,
    UTCDateTime,
    new_uuid,
)
from app.models.enums import (
    ApprovalAction,
    ContentType,
    DraftStatus,
    FactCheckStatus,
    portable_enum,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

    from app.models.content import Idea
    from app.models.publication import Publication, ReplyCandidate
    from app.models.video import VideoScript


class Draft(TimestampMixin, Base):
    """Mutable draft aggregate pointing at an append-only version history."""

    __tablename__ = "drafts"
    __table_args__ = (
        CheckConstraint("current_version_number >= 0", name="drafts_version_nonnegative"),
        CheckConstraint("row_version >= 1", name="drafts_row_version_positive"),
        Index("ix_drafts_status_updated_at", "status", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    idea_id: Mapped[str | None] = mapped_column(
        ForeignKey("ideas.id", ondelete="RESTRICT"), index=True
    )
    reply_candidate_id: Mapped[str | None] = mapped_column(
        ForeignKey("reply_candidates.id", ondelete="RESTRICT"),
        unique=True,
        index=True,
    )
    content_type: Mapped[ContentType] = mapped_column(
        portable_enum(ContentType, name="draft_content_type"), nullable=False
    )
    status: Mapped[DraftStatus] = mapped_column(
        portable_enum(DraftStatus, name="draft_status"),
        nullable=False,
        default=DraftStatus.GENERATED,
    )
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="ru")
    provider: Mapped[str | None] = mapped_column(String(64))
    model: Mapped[str | None] = mapped_column(String(200))
    fact_check_status: Mapped[FactCheckStatus] = mapped_column(
        portable_enum(FactCheckStatus, name="fact_check_status"),
        nullable=False,
        default=FactCheckStatus.NOT_REQUIRED,
    )
    blocking_safety_flags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    media_plan: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    artifact_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    current_content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    current_version_number: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    row_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    deleted_at: Mapped[datetime | None] = mapped_column(UTCDateTime())

    # SQLAlchemy declares this special mapping as an instance attribute in its
    # typing surface even though declarative models configure it at class scope.
    __mapper_args__ = {"version_id_col": row_version}  # noqa: RUF012

    idea: Mapped[Idea | None] = relationship(back_populates="drafts")
    reply_candidate: Mapped[ReplyCandidate | None] = relationship(
        back_populates="draft", uselist=False
    )
    versions: Mapped[list[DraftVersion]] = relationship(
        back_populates="draft",
        order_by="DraftVersion.version_number",
        passive_deletes=True,
    )
    approvals: Mapped[list[DraftApproval]] = relationship(
        back_populates="draft",
        foreign_keys="DraftApproval.draft_id",
        order_by="DraftApproval.created_at",
        passive_deletes=True,
    )
    publications: Mapped[list[Publication]] = relationship(
        back_populates="draft",
        order_by="Publication.thread_position",
        passive_deletes=True,
    )
    video_scripts: Mapped[list[VideoScript]] = relationship(
        back_populates="draft", passive_deletes=True
    )


class DraftVersion(AppendOnlyTimestampMixin, Base):
    """Immutable snapshot of all publishable draft content."""

    __tablename__ = "draft_versions"
    __table_args__ = (
        UniqueConstraint("draft_id", "version_number", name="uq_draft_versions_draft_version"),
        CheckConstraint("version_number >= 1", name="draft_versions_version_positive"),
        Index("ix_draft_versions_draft_hash", "draft_id", "content_hash"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    draft_id: Mapped[str] = mapped_column(
        ForeignKey("drafts.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    rendered_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    origin: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str | None] = mapped_column(String(64))
    model: Mapped[str | None] = mapped_column(String(200))
    feedback: Mapped[str | None] = mapped_column(Text)
    critique: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    generation_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )

    draft: Mapped[Draft] = relationship(back_populates="versions")
    approval_events: Mapped[list[DraftApproval]] = relationship(
        back_populates="version",
        foreign_keys="DraftApproval.draft_version_id",
        passive_deletes=True,
    )
    publications: Mapped[list[Publication]] = relationship(
        back_populates="draft_version", passive_deletes=True
    )


class DraftApproval(AppendOnlyTimestampMixin, Base):
    """Immutable approval/revocation event scoped to an exact content hash.

    Revocation is represented by a new row referencing the approval it revokes;
    an existing approval row is never modified.
    """

    __tablename__ = "draft_approvals"
    __table_args__ = (
        UniqueConstraint("supersedes_id", name="uq_draft_approvals_supersedes"),
        CheckConstraint(
            "((action = 'approved' AND supersedes_id IS NULL) OR "
            "(action = 'revoked' AND supersedes_id IS NOT NULL))",
            name="draft_approvals_action_reference",
        ),
        Index("ix_draft_approvals_draft_created", "draft_id", "created_at"),
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
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[ApprovalAction] = mapped_column(
        portable_enum(ApprovalAction, name="approval_action"), nullable=False
    )
    actor: Mapped[str] = mapped_column(String(200), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    supersedes_id: Mapped[str | None] = mapped_column(
        ForeignKey("draft_approvals.id", ondelete="RESTRICT")
    )

    draft: Mapped[Draft] = relationship(back_populates="approvals", foreign_keys=[draft_id])
    version: Mapped[DraftVersion] = relationship(
        back_populates="approval_events", foreign_keys=[draft_version_id]
    )
    supersedes: Mapped[DraftApproval | None] = relationship(
        remote_side="DraftApproval.id", foreign_keys=[supersedes_id], uselist=False
    )


def _prevent_history_mutation(mapper: Mapper[Any], connection: Connection, target: object) -> None:
    del mapper, connection
    raise RuntimeError(f"{type(target).__name__} rows are append-only")


event.listen(DraftVersion, "before_update", _prevent_history_mutation)
event.listen(DraftVersion, "before_delete", _prevent_history_mutation)
event.listen(DraftApproval, "before_update", _prevent_history_mutation)
event.listen(DraftApproval, "before_delete", _prevent_history_mutation)
