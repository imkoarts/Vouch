"""Raw ingestion preserves posts and recalculates ideas from occurrences."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.models import Base, Idea, IdeaSource, SourcePost, SourcePostOccurrence
from app.models.enums import ContentSourceKind
from app.services.configuration import ConfigurationService
from app.services.idea_collector import IdeaCollector

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as database_session:
        yield database_session
    engine.dispose()


def _collector(session: Session) -> IdeaCollector:
    sources = ConfigurationService(PROJECT_ROOT / "config").load_sources()
    catalog = sources.sources.model_copy(
        update={"recent_search": sources.sources.recent_search.model_copy(update={"enabled": True})}
    )
    sources = sources.model_copy(update={"sources": catalog})
    return IdeaCollector(session, source_configuration=sources)


def test_same_text_and_conversation_with_distinct_post_ids_are_preserved(
    session: Session,
) -> None:
    collector = _collector(session)
    now = datetime.now(UTC).isoformat()
    idea_ids = collector.ingest(
        [
            {
                "id": "101",
                "text": "The same raw text",
                "conversation_id": "shared-thread",
                "created_at": now,
            },
            {
                "id": "102",
                "text": "The same raw text",
                "conversation_id": "shared-thread",
                "created_at": now,
            },
        ],
        source_kind=ContentSourceKind.RECENT_SEARCH,
        source_name="Test search",
    )

    assert len(idea_ids) == 2
    assert session.scalar(select(func.count()).select_from(SourcePost)) == 2
    assert session.scalar(select(func.count()).select_from(Idea)) == 2
    posts = session.scalars(select(SourcePost).order_by(SourcePost.x_post_id)).all()
    assert [post.x_post_id for post in posts] == ["101", "102"]
    assert [post.import_fingerprint for post in posts] == [None, None]


def test_home_occurrence_raises_existing_idea_priority(session: Session) -> None:
    collector = _collector(session)
    payload = {
        "id": "201",
        "text": "One post can appear in more than one configured source.",
        "conversation_id": "201",
        "created_at": datetime.now(UTC).isoformat(),
        "public_metrics": {"like_count": 5},
    }
    [idea_id] = collector.ingest(
        [payload],
        source_kind=ContentSourceKind.RECENT_SEARCH,
        source_name="Test search",
    )
    idea = session.get_one(Idea, idea_id)
    previous_score = idea.score
    assert idea.score_components["source_priority"] == pytest.approx(0.25)

    assert collector.ingest(
        [payload],
        source_kind=ContentSourceKind.HOME_TIMELINE,
        source_name="Home timeline",
    ) == [idea_id]

    session.refresh(idea)
    assert session.scalar(select(func.count()).select_from(SourcePost)) == 1
    assert session.scalar(select(func.count()).select_from(SourcePostOccurrence)) == 2
    assert idea.score_components["source_priority"] == pytest.approx(0.5)
    assert idea.score > previous_score
    source_link = session.scalar(select(IdeaSource).where(IdeaSource.idea_id == idea_id))
    assert source_link is not None
    assert source_link.source_priority == pytest.approx(0.5)


def test_repeated_x_post_refreshes_snapshot_and_retains_occurrence(
    session: Session,
) -> None:
    collector = _collector(session)
    first_payload = {
        "id": "401",
        "text": "Initial synthetic snapshot",
        "username": "old_name",
        "created_at": "2026-07-10T10:00:00+00:00",
        "public_metrics": {"like_count": 1},
        "media_metadata": {"type": "image", "media_key": "old"},
        "raw_json": {"snapshot": 1},
    }
    collector.ingest(
        [first_payload],
        source_kind=ContentSourceKind.RECENT_SEARCH,
        source_name="Refresh search",
    )
    post = session.scalar(select(SourcePost).where(SourcePost.x_post_id == "401"))
    assert post is not None
    occurrence = session.scalar(
        select(SourcePostOccurrence).where(SourcePostOccurrence.source_post_id == post.id)
    )
    assert occurrence is not None
    occurrence_id = occurrence.id
    post.fetched_at = datetime(2020, 1, 1, tzinfo=UTC)
    session.flush()

    second_payload = {
        "id": "401",
        "text": "Refreshed synthetic snapshot",
        "username": "new_name",
        "created_at": "2026-07-10T10:00:00+00:00",
        "public_metrics": {"like_count": 9, "reply_count": 2},
        "media_metadata": {"type": "video", "media_key": "new"},
        "raw_json": {"snapshot": 2},
    }
    collector.ingest(
        [second_payload],
        source_kind=ContentSourceKind.RECENT_SEARCH,
        source_name="Refresh search",
    )

    session.refresh(post)
    refreshed_occurrence = session.get_one(SourcePostOccurrence, occurrence_id)
    assert session.scalar(select(func.count()).select_from(SourcePost)) == 1
    assert session.scalar(select(func.count()).select_from(SourcePostOccurrence)) == 1
    assert post.text == "Refreshed synthetic snapshot"
    assert post.username == "new_name"
    assert post.public_metrics == {"like_count": 9, "reply_count": 2}
    assert post.media_metadata == {"type": "video", "media_key": "new"}
    assert post.raw_json == {"snapshot": 2}
    assert post.fetched_at > datetime(2020, 1, 1, tzinfo=UTC)
    assert refreshed_occurrence.seen_count == 2


def test_no_id_import_uses_stable_fingerprint_across_volatile_timestamps(
    session: Session,
) -> None:
    collector = _collector(session)
    first_payload = {
        "id": None,
        "text": "A reusable synthetic evergreen prompt.",
        "created_at": "2026-07-01T00:00:00+00:00",
    }
    second_payload = {
        **first_payload,
        "created_at": "2026-07-11T00:00:00+00:00",
    }

    first_ids = collector.ingest(
        [first_payload],
        source_kind=ContentSourceKind.EVERGREEN,
        source_name="Evergreen",
    )
    second_ids = collector.ingest(
        [second_payload],
        source_kind=ContentSourceKind.EVERGREEN,
        source_name="Evergreen",
    )

    assert second_ids == first_ids
    assert session.scalar(select(func.count()).select_from(SourcePost)) == 1
    assert session.scalar(select(func.count()).select_from(Idea)) == 1
    post = session.scalar(select(SourcePost))
    assert post is not None
    assert post.x_post_id is None
    assert post.import_fingerprint is not None
    assert len(post.import_fingerprint) == 64
    occurrence = session.scalar(select(SourcePostOccurrence))
    assert occurrence is not None
    assert occurrence.seen_count == 2


def test_collect_mock_is_idempotent_for_evergreen_rows(session: Session) -> None:
    collector = _collector(session)

    first_ids = collector.collect_mock()
    second_ids = collector.collect_mock()

    assert second_ids == first_ids
    assert session.scalar(select(func.count()).select_from(SourcePost)) == 3
    assert session.scalar(select(func.count()).select_from(Idea)) == 3
