from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.database import get_engine, reset_database_state
from app.models import Base
from app.services.manual_generation import TopicResearchError, create_researched_topic_idea

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        config_dir=PROJECT_ROOT / "config",
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        logs_dir=tmp_path / "logs",
        database_url=f"sqlite:///{(tmp_path / 'topic-research.db').as_posix()}",
        mock_mode=False,
        x_bearer_token="synthetic-bearer",
        daily_x_read_limit_usd=0.25,
    )


class FakeTopicReader:
    def __init__(self, posts: tuple[dict[str, object], ...]) -> None:
        self.posts = posts
        self.calls = 0

    async def recent_search(self, *_args, **_kwargs):
        self.calls += 1
        return self.posts


@pytest.mark.asyncio
async def test_live_topic_research_is_disabled_without_calling_recent_search(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setenv("DATABASE_URL", str(settings.database_url))
    monkeypatch.setenv("CONFIG_DIR", str(PROJECT_ROOT / "config"))
    get_settings.cache_clear()
    reset_database_state()
    Base.metadata.create_all(get_engine())
    reader = FakeTopicReader(())

    try:
        with pytest.raises(TopicResearchError, match="Recent Search is disabled"):
            await create_researched_topic_idea(
                settings,
                topic_text="prediction market token",
                reader=reader,
                sleep=lambda _seconds: _no_sleep(),
            )
        assert reader.calls == 0
    finally:
        get_settings.cache_clear()
        reset_database_state()


async def _no_sleep() -> None:
    return None
