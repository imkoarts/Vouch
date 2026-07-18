from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
import yaml
from alembic.config import Config

from alembic import command
from app.config import Settings
from app.main import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
async def test_fastapi_lifespan_autostarts_and_stops_telegram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()
    closed = asyncio.Event()

    class FakeApi:
        def __init__(
            self, token: str, *, timeout_seconds: float, proxy_url: str | None = None
        ) -> None:
            assert token == "synthetic-telegram-token"
            assert timeout_seconds == 45
            assert proxy_url is None

        async def close(self) -> None:
            closed.set()

    class FakeBot:
        def __init__(self, settings: Settings, api: FakeApi) -> None:
            del settings, api

        async def run_forever(self, stop_event: asyncio.Event | None = None) -> None:
            started.set()
            assert stop_event is not None
            try:
                await stop_event.wait()
            finally:
                stopped.set()

    async def fake_discovery_loop(
        _settings: Settings, *, stop_event: asyncio.Event, on_outcome: object
    ) -> None:
        del on_outcome
        await stop_event.wait()

    monkeypatch.setattr("app.main.TelegramBotApi", FakeApi)
    monkeypatch.setattr("app.main.TelegramReviewBot", FakeBot)
    monkeypatch.setattr("app.main.automatic_discovery_loop", fake_discovery_loop)
    database_url = f"sqlite:///{(tmp_path / 'app.db').as_posix()}"
    alembic = Config(str(PROJECT_ROOT / "alembic.ini"))
    alembic.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    alembic.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic, "head")
    config_dir = tmp_path / "config"
    shutil.copytree(PROJECT_ROOT / "config", config_dir)
    runtime_path = config_dir / "runtime.yml"
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    runtime["telegram"]["enabled"] = True
    runtime["telegram"]["autostart"] = True
    runtime_path.write_text(yaml.safe_dump(runtime), encoding="utf-8")
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=database_url,
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        config_dir=config_dir,
        telegram_bot_token="synthetic-telegram-token",
        mock_mode=True,
    )
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        await asyncio.wait_for(started.wait(), timeout=1)

    assert stopped.is_set()
    assert closed.is_set()


@pytest.mark.asyncio
async def test_unmigrated_database_does_not_start_background_integrations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructed = False

    class UnexpectedApi:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            nonlocal constructed
            constructed = True

    monkeypatch.setattr("app.main.TelegramBotApi", UnexpectedApi)
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=f"sqlite:///{(tmp_path / 'unmigrated.db').as_posix()}",
        data_dir=tmp_path / "data-unmigrated",
        drafts_dir=tmp_path / "drafts-unmigrated",
        config_dir=PROJECT_ROOT / "config",
        telegram_bot_token="synthetic-telegram-token",
        mock_mode=True,
    )
    application = create_app(settings)

    async with application.router.lifespan_context(application):
        pass

    assert constructed is False
