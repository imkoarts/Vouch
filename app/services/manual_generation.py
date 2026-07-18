"""Compatibility helpers for operator-requested topic generation.

Live topic research previously depended on X Recent Search. That endpoint is now
product-disabled. The dashboard manual workflow accepts a direct X post URL instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.database import session_scope
from app.domain.enums import ContentType, IdeaStatus, TopicKind
from app.models import Idea, IdeaSource, Topic
from app.x_api.read import XReadClient


class TopicResearchError(RuntimeError):
    """Raised when a custom topic cannot be grounded without Recent Search."""


def create_manual_topic_idea(
    session: Session,
    *,
    topic_text: str,
    content_type: ContentType = ContentType.SHORT_POST,
    source_priority: float = 0.05,
) -> Idea:
    """Create a topic-only idea for mock mode and compatibility tests."""

    normalized = topic_text.strip()
    if not normalized:
        raise ValueError("Topic must not be empty")
    topic = session.scalar(select(Topic).where(Topic.name == normalized[:200]))
    if topic is None:
        topic = Topic(
            name=normalized[:200],
            kind=TopicKind.SEARCH,
            query=normalized,
            description=normalized,
            enabled=True,
            configuration={"origin": "manual_topic"},
        )
        session.add(topic)
        session.flush()
    idea = Idea(
        status=IdeaStatus.RANKED,
        title=normalized[:500],
        summary=normalized,
        recommended_format=content_type,
        score=max(0.0, source_priority),
        score_components={"source_priority": source_priority},
        explanation={
            "origin": "manual_topic",
            "requested_topic": normalized,
            "why_selected": "The operator explicitly requested this topic.",
        },
    )
    session.add(idea)
    session.flush()
    session.add(
        IdeaSource(
            idea_id=idea.id,
            topic_id=topic.id,
            source_priority=source_priority,
            contribution=max(0.0, source_priority),
            position=0,
            link_metadata={"origin": "manual_topic"},
        )
    )
    session.flush()
    return idea


async def create_researched_topic_idea(
    settings: Settings,
    *,
    topic_text: str,
    content_type: ContentType = ContentType.SHORT_POST,
    actor: str = "operator",
    reader: XReadClient | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> str:
    """Reject live topic search while retaining deterministic mock compatibility."""

    normalized = " ".join(topic_text.strip().split())
    if not normalized:
        raise ValueError("Topic must not be empty")
    if settings.mock_mode:
        with session_scope() as session:
            return create_manual_topic_idea(
                session,
                topic_text=normalized,
                content_type=content_type,
            ).id

    del actor, reader, sleep
    raise TopicResearchError(
        "Topic-based X research is unavailable because Recent Search is disabled. "
        "Use the dashboard command /new <direct X post URL> instead."
    )
