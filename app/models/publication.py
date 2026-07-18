"""Publication idempotency records and inbound reply candidates."""

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
    false,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UTCDateTime, new_uuid
from app.models.enums import (
    PublicationKind,
    PublicationStatus,
    ReplyCandidateStatus,
    portable_enum,
)

if TYPE_CHECKING:
    from app.models.content import SourcePost
    from app.models.drafts import Draft, DraftVersion


class ReplyCandidate(TimestampMixin, Base):
    """At most one automatically prepared reply candidate per inbound post."""

    __tablename__ = "reply_candidates"
    __table_args__ = (
        UniqueConstraint("source_post_id", name="uq_reply_candidates_source_post_id"),
        Index("ix_reply_candidates_status_created", "status", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    source_post_id: Mapped[str] = mapped_column(
        ForeignKey("source_posts.id", ondelete="RESTRICT"), nullable=False
    )
    conversation_id: Mapped[str | None] = mapped_column(String(64), index=True)
    author_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    username: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[ReplyCandidateStatus] = mapped_column(
        portable_enum(ReplyCandidateStatus, name="reply_candidate_status"),
        nullable=False,
        default=ReplyCandidateStatus.DISCOVERED,
    )
    direct_mention: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    reply_to_owned_post: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    allowlisted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    eligibility_reason: Mapped[str | None] = mapped_column(Text)
    conversation_context: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    source_post: Mapped[SourcePost] = relationship(back_populates="reply_candidate")
    draft: Mapped[Draft | None] = relationship(
        back_populates="reply_candidate", uselist=False, passive_deletes=True
    )


class Publication(TimestampMixin, Base):
    """One durable, idempotent publication record per thread position.

    A row is claimed before calling X and then records the outcome. Successfully
    published positions are reused when a partial thread is resumed.
    """

    __tablename__ = "publications"
    __table_args__ = (
        UniqueConstraint(
            "draft_id",
            "content_hash",
            "thread_position",
            name="uq_publications_draft_hash_position",
        ),
        UniqueConstraint(
            "draft_id",
            "request_payload_hash",
            "thread_position",
            name="uq_publications_request_position",
        ),
        UniqueConstraint("x_post_id", name="uq_publications_x_post_id"),
        CheckConstraint("thread_position >= 0", name="publications_position_nonnegative"),
        CheckConstraint("attempt_count >= 0", name="publications_attempt_nonnegative"),
        CheckConstraint("estimated_cost_usd >= 0", name="publications_cost_nonnegative"),
        Index("ix_publications_draft_status", "draft_id", "status"),
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
    kind: Mapped[PublicationKind] = mapped_column(
        portable_enum(PublicationKind, name="publication_kind"), nullable=False
    )
    status: Mapped[PublicationStatus] = mapped_column(
        portable_enum(PublicationStatus, name="publication_status"),
        nullable=False,
        default=PublicationStatus.CLAIMED,
    )
    thread_position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    request_payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    x_post_id: Mapped[str | None] = mapped_column(String(32))
    parent_post_id: Mapped[str | None] = mapped_column(String(32))
    account_id: Mapped[str | None] = mapped_column(String(64))
    response_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    estimated_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(14, 6), nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    published_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    deletion_timestamp: Mapped[datetime | None] = mapped_column(UTCDateTime())
    error_details: Mapped[str | None] = mapped_column(Text)

    draft: Mapped[Draft] = relationship(back_populates="publications")
    draft_version: Mapped[DraftVersion] = relationship(back_populates="publications")
