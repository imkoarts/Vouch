from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.database import build_engine
from app.domain.enums import ContentType, PostLengthMode
from app.models import Base
from app.providers.mock_provider import MockLLMProvider
from app.schemas.content import GenerationRequest
from app.services.configuration import ConfigurationService
from app.services.draft_files import DraftArtifactStore
from app.services.draft_service import DraftService
from app.services.manual_generation import create_manual_topic_idea
from app.services.post_length import resolve_post_length, selection_from_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("mode", "minimum", "maximum", "content_type"),
    [
        (PostLengthMode.SHORT, 1, 280, ContentType.SHORT_POST),
        (PostLengthMode.MEDIUM, 300, 500, ContentType.LONG_POST),
        (PostLengthMode.LONGREAD, 500, 1_000, ContentType.LONG_POST),
    ],
)
def test_concrete_post_length_ranges(
    mode: PostLengthMode,
    minimum: int,
    maximum: int,
    content_type: ContentType,
) -> None:
    selection = resolve_post_length(mode)

    assert selection.resolved_mode is mode
    assert selection.minimum == minimum
    assert selection.maximum == maximum
    assert selection.content_type is content_type


def test_random_mode_is_resolved_before_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedRandom:
        def choice(self, _values: object) -> PostLengthMode:
            return PostLengthMode.MEDIUM

    monkeypatch.setattr("app.services.post_length.SystemRandom", FixedRandom)

    selection = resolve_post_length(PostLengthMode.RANDOM)

    assert selection.configured_mode is PostLengthMode.RANDOM
    assert selection.resolved_mode is PostLengthMode.MEDIUM
    assert selection.minimum == 300
    assert selection.maximum == 500


@pytest.mark.asyncio
async def test_short_draft_generation_applies_and_persists_selected_range(
    tmp_path: Path,
) -> None:
    engine = build_engine(f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    Base.metadata.create_all(engine)
    profile = ConfigurationService(PROJECT_ROOT / "config").load_profile()
    with Session(engine, expire_on_commit=False) as session:
        idea = create_manual_topic_idea(session, topic_text="Range test")
        service = DraftService(
            session,
            DraftArtifactStore(tmp_path / "drafts"),
            MockLLMProvider(),
            profile=profile,
            premium_long_posts_enabled=True,
            post_length_mode=PostLengthMode.SHORT,
        )

        draft = await service.generate_from_idea(idea.id)

        assert draft is not None
        version = service.current_version(draft)
        selection = selection_from_metadata(version.generation_metadata["post_length"])
        text = version.content["parts"][0]
        assert draft.content_type is ContentType.SHORT_POST
        assert selection is not None
        assert selection.resolved_mode is PostLengthMode.SHORT
        assert 1 <= len(text) <= 280
        assert selection.minimum == 1
        assert selection.maximum == 280


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PostLengthMode.MEDIUM, PostLengthMode.LONGREAD])
async def test_mock_draft_generation_abstains_instead_of_padding_hard_minimum(
    tmp_path: Path,
    mode: PostLengthMode,
) -> None:
    engine = build_engine(f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    Base.metadata.create_all(engine)
    profile = ConfigurationService(PROJECT_ROOT / "config").load_profile()
    with Session(engine, expire_on_commit=False) as session:
        idea = create_manual_topic_idea(session, topic_text="Range test")
        service = DraftService(
            session,
            DraftArtifactStore(tmp_path / "drafts"),
            MockLLMProvider(),
            profile=profile,
            premium_long_posts_enabled=True,
            post_length_mode=mode,
        )

        draft = await service.generate_from_idea(idea.id)
        session.refresh(idea)

        assert draft is None
        decision = idea.explanation["editorial_decision"]
        assert decision["status"] == "insufficient_context"
        assert decision["reason_code"] == "MOCK_HARD_MINIMUM_UNSUPPORTED"


@pytest.mark.asyncio
async def test_mock_provider_reports_controlled_failure_for_unsupported_medium_minimum() -> None:
    result = await MockLLMProvider().generate_structured(
        GenerationRequest(
            idea_summary="Test",
            content_type=ContentType.LONG_POST,
            language="en",
            post_length_mode="medium",
            minimum_characters=300,
            maximum_characters=500,
        )
    )

    assert result.status == "insufficient_context"
    assert result.reason_code == "MOCK_HARD_MINIMUM_UNSUPPORTED"
    assert result.variants == ()


@pytest.mark.asyncio
async def test_random_draft_persists_the_resolved_mode_before_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedRandom:
        def choice(self, _values: object) -> PostLengthMode:
            return PostLengthMode.LONGREAD

    monkeypatch.setattr("app.services.post_length.SystemRandom", FixedRandom)
    engine = build_engine(f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    Base.metadata.create_all(engine)
    profile = ConfigurationService(PROJECT_ROOT / "config").load_profile()
    with Session(engine, expire_on_commit=False) as session:
        idea = create_manual_topic_idea(session, topic_text="Random range")
        service = DraftService(
            session,
            DraftArtifactStore(tmp_path / "drafts"),
            MockLLMProvider(),
            profile=profile,
            premium_long_posts_enabled=True,
            post_length_mode=PostLengthMode.RANDOM,
        )

        draft = await service.generate_from_idea(idea.id)
        session.refresh(idea)

        assert draft is None
        decision = idea.explanation["editorial_decision"]
        assert decision["status"] == "insufficient_context"
        assert decision["reason_code"] == "MOCK_HARD_MINIMUM_UNSUPPORTED"
        assert decision["post_length"]["configured_mode"] == "random"
        assert decision["post_length"]["resolved_mode"] == "longread"
