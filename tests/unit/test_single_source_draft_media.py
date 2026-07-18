from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.database import build_engine
from app.domain.enums import ContentType, IdeaStatus
from app.models import Base, Idea, IdeaSource, SourcePost
from app.providers.mock_provider import MockLLMProvider
from app.services.configuration import ConfigurationService
from app.services.draft_files import DraftArtifactStore
from app.services.draft_service import DraftService
from app.services.source_media import SourceContextPlan

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
async def test_single_source_generation_materializes_deterministic_source_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = build_engine(f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    Base.metadata.create_all(engine)
    profile = ConfigurationService(PROJECT_ROOT / "config").load_profile()

    async def fake_download(
        plan: SourceContextPlan,
        directory: Path,
        *,
        proxy_url: str | None = None,
    ) -> dict[str, object]:
        assert plan.strategy == "attach_source_media"
        assert proxy_url == "http://user:pass@127.0.0.1:8080"
        media = directory / "media" / "source-1.mp4"
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(b"source-video")
        return {
            "type": "video",
            "reason": "source context",
            "required_files": ["media/source-1.mp4"],
            "metadata": {
                "context_strategy": "attach_source_media",
                "source_post_id": "2076265333803913327",
                "source_url": "https://x.com/bdll_10/status/2076265333803913327",
            },
        }

    monkeypatch.setattr("app.services.draft_service.download_source_media", fake_download)

    with Session(engine, expire_on_commit=False) as session:
        post = SourcePost(
            x_post_id="2076265333803913327",
            text="the only time i saw messi arrogant",
            text_hash="a" * 64,
            username="bdll_10",
            url="https://x.com/bdll_10/status/2076265333803913327",
            public_metrics={"retweet_count": 500},
            source_labels=["home_timeline"],
            media_metadata={
                "items": [
                    {
                        "media_key": "7_1",
                        "type": "video",
                        "variants": [
                            {
                                "content_type": "video/mp4",
                                "bit_rate": 2176000,
                                "url": "https://video.twimg.com/messi.mp4",
                            }
                        ],
                    }
                ]
            },
        )
        session.add(post)
        session.flush()
        idea = Idea(
            status=IdeaStatus.RANKED,
            title="Messi moment",
            summary="Write one thought from one source anchor.",
            recommended_format=ContentType.SHORT_POST,
            score=1.0,
            score_components={"single_source_anchor": 1.0},
            explanation={
                "origin": "automatic_discovery_single_source",
                "anchor_x_post_id": post.x_post_id,
                "anchor_url": post.url,
                "anchor_author_username": post.username,
                "anchor_has_media": True,
            },
            dedupe_key="b" * 64,
        )
        session.add(idea)
        session.flush()
        session.add(
            IdeaSource(
                idea_id=idea.id,
                source_post_id=post.id,
                source_priority=0.5,
                contribution=1.0,
                position=0,
            )
        )
        session.flush()
        service = DraftService(
            session,
            DraftArtifactStore(tmp_path / "drafts"),
            MockLLMProvider(),
            profile=profile,
            outbound_proxy_url="http://user:pass@127.0.0.1:8080",
        )

        draft = await service.generate_from_idea(idea.id)
        version = service.current_version(draft)

        assert draft.media_plan["type"] == "video"
        assert draft.media_plan["required_files"] == ["media/source-1.mp4"]
        media_bytes = await asyncio.to_thread(
            Path(draft.artifact_path, "media", "source-1.mp4").read_bytes
        )
        assert media_bytes == b"source-video"
        assert version.generation_metadata["source_context"]["strategy"] == "attach_source_media"
        assert version.generation_metadata["evidence_item_count"] == 1


@pytest.mark.asyncio
async def test_media_download_failure_raises_contextual_candidate_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services.draft_service import MediaContextUnavailableError
    from app.services.source_media import SourceMediaError

    engine = build_engine(f"sqlite:///{(tmp_path / 'failure.db').as_posix()}")
    Base.metadata.create_all(engine)
    profile = ConfigurationService(PROJECT_ROOT / "config").load_profile()

    async def fail_download(
        _plan: SourceContextPlan,
        _directory: Path,
        *,
        proxy_url: str | None = None,
    ) -> dict[str, object]:
        assert proxy_url is None
        raise SourceMediaError("X source media download failed after bounded variants")

    monkeypatch.setattr("app.services.draft_service.download_source_media", fail_download)

    with Session(engine, expire_on_commit=False) as session:
        post = SourcePost(
            x_post_id="2076265333803913327",
            text="the only time i saw messi arrogant",
            text_hash="c" * 64,
            username="bdll_10",
            url="https://x.com/bdll_10/status/2076265333803913327",
            source_labels=["home_timeline"],
            media_metadata={
                "items": [
                    {
                        "media_key": "7_1",
                        "type": "video",
                        "variants": [
                            {
                                "content_type": "video/mp4",
                                "bit_rate": 2176000,
                                "url": "https://video.twimg.com/messi.mp4",
                            }
                        ],
                    }
                ]
            },
        )
        session.add(post)
        session.flush()
        idea = Idea(
            status=IdeaStatus.RANKED,
            title="Messi moment",
            summary="Write one thought from one source anchor.",
            recommended_format=ContentType.SHORT_POST,
            score=1.0,
            score_components={"single_source_anchor": 1.0},
            explanation={
                "origin": "automatic_discovery_single_source",
                "anchor_x_post_id": post.x_post_id,
                "anchor_url": post.url,
                "anchor_has_media": True,
            },
            dedupe_key="d" * 64,
        )
        session.add(idea)
        session.flush()
        session.add(
            IdeaSource(
                idea_id=idea.id,
                source_post_id=post.id,
                source_priority=0.5,
                contribution=1.0,
                position=0,
            )
        )
        session.flush()

        class CountingProvider(MockLLMProvider):
            def __init__(self) -> None:
                self.calls = 0

            async def generate_structured(self, request):  # type: ignore[no-untyped-def]
                self.calls += 1
                return await super().generate_structured(request)

        provider = CountingProvider()
        service = DraftService(
            session,
            DraftArtifactStore(tmp_path / "drafts-failure"),
            provider,
            profile=profile,
        )

        with pytest.raises(MediaContextUnavailableError) as captured:
            await service.generate_from_idea(idea.id)

        assert captured.value.source_post_id == "2076265333803913327"
        assert captured.value.source_url == "https://x.com/bdll_10/status/2076265333803913327"
        assert "bounded variants" in captured.value.detail
        assert provider.calls == 0
        assert not (tmp_path / "drafts-failure").exists() or not any(
            (tmp_path / "drafts-failure").rglob("source-1.mp4")
        )
