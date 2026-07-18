"""Content source, source post, topic, and idea persistence models."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    true,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.domain.enums import ContentSourceKind, ContentType, IdeaStatus, TopicKind
from app.models.base import Base, TimestampMixin, UTCDateTime, new_uuid, utc_now
from app.models.enums import portable_enum

if TYPE_CHECKING:
    from app.models.drafts import Draft
    from app.models.publication import ReplyCandidate


class ContentSource(TimestampMixin, Base):
    """Configured acquisition source and its deterministic priority weight."""

    __tablename__ = "content_sources"
    __table_args__ = (
        UniqueConstraint("kind", "identifier", name="uq_content_sources_kind_identifier"),
        CheckConstraint("weight >= 0 AND weight <= 1", name="content_sources_weight_range"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    kind: Mapped[ContentSourceKind] = mapped_column(
        portable_enum(ContentSourceKind, name="content_source_kind"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    identifier: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    category: Mapped[str | None] = mapped_column(String(100))
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=true()
    )
    configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    occurrences: Mapped[list[SourcePostOccurrence]] = relationship(
        back_populates="content_source", passive_deletes=True
    )


class Topic(TimestampMixin, Base):
    """Search or evergreen topic managed by the user."""

    __tablename__ = "topics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    kind: Mapped[TopicKind] = mapped_column(
        portable_enum(TopicKind, name="topic_kind"), nullable=False
    )
    query: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=true()
    )
    configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    idea_links: Mapped[list[IdeaSource]] = relationship(
        back_populates="topic", passive_deletes=True
    )


class SourcePost(TimestampMixin, Base):
    """Normalized snapshot of an X post or a local/imported source row."""

    __tablename__ = "source_posts"
    __table_args__ = (
        UniqueConstraint("x_post_id", name="uq_source_posts_x_post_id"),
        Index(
            "ux_source_posts_import_fingerprint",
            "import_fingerprint",
            unique=True,
        ),
        Index("ix_source_posts_conversation_id", "conversation_id"),
        Index("ix_source_posts_fetched_at", "fetched_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    x_post_id: Mapped[str | None] = mapped_column(String(32))
    import_fingerprint: Mapped[str | None] = mapped_column(String(64))
    url: Mapped[str | None] = mapped_column(String(2048))
    author_id: Mapped[str | None] = mapped_column(String(64), index=True)
    username: Mapped[str | None] = mapped_column(String(64), index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str | None] = mapped_column(String(16))
    published_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    public_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    conversation_id: Mapped[str | None] = mapped_column(String(64))
    media_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    source_labels: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    occurrences: Mapped[list[SourcePostOccurrence]] = relationship(
        back_populates="source_post", passive_deletes=True
    )
    idea_links: Mapped[list[IdeaSource]] = relationship(
        back_populates="source_post", passive_deletes=True
    )
    reply_candidate: Mapped[ReplyCandidate | None] = relationship(
        back_populates="source_post", uselist=False, passive_deletes=True
    )


class SourcePostOccurrence(Base):
    """Idempotent association between one raw post and one configured source."""

    __tablename__ = "source_post_occurrences"
    __table_args__ = (
        UniqueConstraint(
            "source_post_id",
            "content_source_id",
            name="uq_source_post_occurrences_post_source",
        ),
        CheckConstraint(
            "source_weight >= 0 AND source_weight <= 1",
            name="source_post_occurrences_weight_range",
        ),
        CheckConstraint(
            "seen_count >= 1",
            name="source_post_occurrences_seen_count_positive",
        ),
        Index(
            "ix_source_post_occurrences_source_post",
            "source_post_id",
            "source_weight",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    source_post_id: Mapped[str] = mapped_column(
        ForeignKey("source_posts.id", ondelete="RESTRICT"), nullable=False
    )
    content_source_id: Mapped[str] = mapped_column(
        ForeignKey("content_sources.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    source_weight: Mapped[float] = mapped_column(Float, nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    last_seen_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    seen_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    occurrence_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )

    source_post: Mapped[SourcePost] = relationship(back_populates="occurrences")
    content_source: Mapped[ContentSource] = relationship(back_populates="occurrences")


class Idea(TimestampMixin, Base):
    """A reproducibly ranked content idea."""

    __tablename__ = "ideas"
    __table_args__ = (CheckConstraint("score >= 0", name="ideas_score_nonnegative"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    status: Mapped[IdeaStatus] = mapped_column(
        portable_enum(IdeaStatus, name="idea_status"),
        nullable=False,
        default=IdeaStatus.COLLECTED,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_format: Mapped[ContentType | None] = mapped_column(
        portable_enum(ContentType, name="idea_content_type")
    )
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    score_components: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    explanation: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    dedupe_key: Mapped[str | None] = mapped_column(String(64), unique=True)

    source_links: Mapped[list[IdeaSource]] = relationship(
        back_populates="idea", passive_deletes=True
    )
    drafts: Mapped[list[Draft]] = relationship(back_populates="idea", passive_deletes=True)


class IdeaSource(Base):
    """A source post or evergreen topic contributing to an idea."""

    __tablename__ = "idea_sources"
    __table_args__ = (
        UniqueConstraint("idea_id", "source_post_id", name="uq_idea_sources_idea_source_post"),
        UniqueConstraint("idea_id", "topic_id", name="uq_idea_sources_idea_topic"),
        CheckConstraint(
            "((source_post_id IS NOT NULL AND topic_id IS NULL) OR "
            "(source_post_id IS NULL AND topic_id IS NOT NULL))",
            name="idea_sources_exactly_one_source",
        ),
        CheckConstraint(
            "source_priority >= 0 AND contribution >= 0",
            name="idea_sources_nonnegative_scores",
        ),
        CheckConstraint("position >= 0", name="idea_sources_position_nonnegative"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    idea_id: Mapped[str] = mapped_column(
        ForeignKey("ideas.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    source_post_id: Mapped[str | None] = mapped_column(
        ForeignKey("source_posts.id", ondelete="RESTRICT"), index=True
    )
    topic_id: Mapped[str | None] = mapped_column(
        ForeignKey("topics.id", ondelete="RESTRICT"), index=True
    )
    source_priority: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    contribution: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    link_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )

    idea: Mapped[Idea] = relationship(back_populates="source_links")
    source_post: Mapped[SourcePost | None] = relationship(back_populates="idea_links")
    topic: Mapped[Topic | None] = relationship(back_populates="idea_links")
