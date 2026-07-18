"""Persistence and deterministic deduplication for content sources and ideas."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.enums import ContentSourceKind, ContentType, IdeaStatus
from app.domain.security import normalize_text
from app.models import (
    ContentSource,
    Idea,
    IdeaSource,
    SourcePost,
    SourcePostOccurrence,
)
from app.models.base import utc_now


def normalized_text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def stable_import_fingerprint(
    *,
    text: str,
    source_kind: ContentSourceKind,
    source_identifier: str,
) -> str:
    """Fingerprint a no-ID import using stable provenance and normalized content.

    Real X posts never use this key: their provider ID remains the sole identity
    contract even when two posts have identical text.
    """

    canonical = json.dumps(
        {
            "version": 1,
            "source_kind": source_kind.value,
            "source_identifier": source_identifier,
            "text": normalize_text(text),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _published_at(payload: Mapping[str, Any]) -> datetime | None:
    value = payload.get("created_at")
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value if isinstance(value, datetime) else None


class ContentRepository:
    """Repository that preserves idempotency before adding source rows."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_or_create_source(
        self,
        *,
        kind: ContentSourceKind,
        name: str,
        identifier: str = "",
        weight: float,
        configuration: Mapping[str, Any] | None = None,
    ) -> ContentSource:
        source = self.session.scalar(
            select(ContentSource).where(
                ContentSource.kind == kind,
                ContentSource.identifier == identifier,
            )
        )
        if source is None:
            source = ContentSource(
                kind=kind,
                name=name,
                identifier=identifier,
                weight=weight,
                configuration=dict(configuration or {}),
            )
            self.session.add(source)
            self.session.flush()
        else:
            source.name = name
            source.weight = weight
            source.configuration = dict(configuration or {})
        return source

    def find_by_x_post_id(self, x_post_id: str | None) -> SourcePost | None:
        """Deduplicate only the stable provider ID, never text or conversation."""

        if x_post_id is None:
            return None
        return self.session.scalar(
            select(SourcePost).where(SourcePost.x_post_id == x_post_id).limit(1)
        )

    def find_by_import_fingerprint(
        self,
        *,
        fingerprint: str,
        source: ContentSource,
        text_hash: str,
    ) -> SourcePost | None:
        """Find a no-ID import, adopting one legacy row created before the key."""

        existing = self.session.scalar(
            select(SourcePost).where(SourcePost.import_fingerprint == fingerprint).limit(1)
        )
        if existing is not None:
            return existing

        legacy = self.session.scalar(
            select(SourcePost)
            .join(
                SourcePostOccurrence,
                SourcePostOccurrence.source_post_id == SourcePost.id,
            )
            .where(
                SourcePost.x_post_id.is_(None),
                SourcePost.import_fingerprint.is_(None),
                SourcePost.text_hash == text_hash,
                SourcePostOccurrence.content_source_id == source.id,
            )
            .order_by(SourcePost.fetched_at.desc(), SourcePost.id.asc())
            .limit(1)
        )
        if legacy is not None:
            legacy.import_fingerprint = fingerprint
        return legacy

    @staticmethod
    def _refresh_snapshot(
        source_post: SourcePost,
        *,
        payload: Mapping[str, Any],
        text: str,
        x_post_id: str | None,
    ) -> None:
        """Replace mutable provider snapshot fields while preserving row identity."""

        username = _optional_string(payload.get("username"))
        author_id = _optional_string(payload.get("author_id"))
        conversation_id = _optional_string(payload.get("conversation_id"))
        source_post.url = _optional_string(payload.get("url")) or (
            f"https://x.com/i/web/status/{x_post_id}" if x_post_id else None
        )
        source_post.author_id = author_id
        source_post.username = username
        source_post.text = text
        source_post.language = _optional_string(payload.get("lang") or payload.get("language"))
        source_post.published_at = _published_at(payload)
        source_post.public_metrics = dict(payload.get("public_metrics") or {})
        source_post.conversation_id = conversation_id
        source_post.media_metadata = dict(payload.get("media_metadata") or {})
        source_post.fetched_at = utc_now()
        source_post.raw_json = dict(payload.get("raw_json") or payload)
        source_post.text_hash = normalized_text_hash(text)

    def record_occurrence(
        self,
        *,
        source_post: SourcePost,
        source: ContentSource,
    ) -> tuple[SourcePostOccurrence, bool]:
        """Associate a post with a source and retain repeat-observation metadata."""

        occurrence = self.session.scalar(
            select(SourcePostOccurrence).where(
                SourcePostOccurrence.source_post_id == source_post.id,
                SourcePostOccurrence.content_source_id == source.id,
            )
        )
        now = utc_now()
        created = occurrence is None
        if occurrence is None:
            occurrence = SourcePostOccurrence(
                source_post_id=source_post.id,
                content_source_id=source.id,
                source_weight=source.weight,
                first_seen_at=now,
                last_seen_at=now,
                seen_count=1,
                occurrence_metadata={"source_kind": source.kind.value},
            )
            self.session.add(occurrence)
        else:
            occurrence.last_seen_at = now
            occurrence.seen_count += 1
            occurrence.source_weight = source.weight
        label = source.kind.value
        if label not in source_post.source_labels:
            source_post.source_labels = [*source_post.source_labels, label]
        self.session.flush()
        return occurrence, created

    def effective_source_priority(self, source_post: SourcePost) -> float:
        """Use the highest current enabled source weight for deterministic ranking."""

        value = self.session.scalar(
            select(func.max(ContentSource.weight))
            .join(
                SourcePostOccurrence,
                SourcePostOccurrence.content_source_id == ContentSource.id,
            )
            .where(
                SourcePostOccurrence.source_post_id == source_post.id,
                ContentSource.enabled.is_(True),
            )
        )
        return float(value or 0.0)

    def import_source_post(
        self,
        *,
        payload: Mapping[str, Any],
        source: ContentSource,
    ) -> tuple[SourcePost, bool]:
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("A source post must contain text")
        x_post_id = _optional_string(payload.get("id"))
        text_hash = normalized_text_hash(text)
        import_fingerprint = (
            stable_import_fingerprint(
                text=text,
                source_kind=source.kind,
                source_identifier=source.identifier,
            )
            if x_post_id is None
            else None
        )
        if x_post_id is not None:
            duplicate = self.find_by_x_post_id(x_post_id)
        else:
            if import_fingerprint is None:  # pragma: no cover - structural invariant
                raise AssertionError("No-ID imports must have a fingerprint")
            duplicate = self.find_by_import_fingerprint(
                fingerprint=import_fingerprint,
                source=source,
                text_hash=text_hash,
            )
        label = source.kind.value
        if duplicate is not None:
            self._refresh_snapshot(
                duplicate,
                payload=payload,
                text=text,
                x_post_id=x_post_id,
            )
            self.record_occurrence(source_post=duplicate, source=source)
            return duplicate, False

        post = SourcePost(
            x_post_id=x_post_id,
            import_fingerprint=import_fingerprint,
            text=text,
            source_labels=[label],
            text_hash=text_hash,
        )
        self._refresh_snapshot(
            post,
            payload=payload,
            text=text,
            x_post_id=x_post_id,
        )
        self.session.add(post)
        self.session.flush()
        self.record_occurrence(source_post=post, source=source)
        return post, True

    def create_idea(
        self,
        *,
        source_post: SourcePost,
        title: str,
        summary: str,
        score: float,
        components: Mapping[str, float],
        explanation: Mapping[str, Any],
        recommended_format: ContentType,
        source_priority: float,
    ) -> tuple[Idea, bool]:
        dedupe_key = hashlib.sha256(f"source:{source_post.id}".encode()).hexdigest()
        existing = self.session.scalar(select(Idea).where(Idea.dedupe_key == dedupe_key))
        if existing is not None:
            existing.title = title[:500]
            existing.summary = summary
            existing.recommended_format = recommended_format
            existing.score = max(0.0, score)
            existing.score_components = dict(components)
            existing.explanation = dict(explanation)
            source_link = self.session.scalar(
                select(IdeaSource).where(
                    IdeaSource.idea_id == existing.id,
                    IdeaSource.source_post_id == source_post.id,
                )
            )
            if source_link is not None:
                source_link.source_priority = source_priority
                source_link.contribution = max(0.0, score)
                source_link.link_metadata = {"source_labels": source_post.source_labels}
            self.session.flush()
            return existing, False
        idea = Idea(
            status=IdeaStatus.RANKED,
            title=title[:500],
            summary=summary,
            recommended_format=recommended_format,
            score=max(0.0, score),
            score_components=dict(components),
            explanation=dict(explanation),
            dedupe_key=dedupe_key,
        )
        self.session.add(idea)
        self.session.flush()
        self.session.add(
            IdeaSource(
                idea_id=idea.id,
                source_post_id=source_post.id,
                source_priority=source_priority,
                contribution=max(0.0, score),
                position=0,
                link_metadata={"source_labels": source_post.source_labels},
            )
        )
        self.session.flush()
        return idea, True

    def list_ideas(self) -> Iterable[Idea]:
        return self.session.scalars(select(Idea).order_by(Idea.score.desc())).all()
