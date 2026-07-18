from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.config import get_settings
from app.database import reset_database_state

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_mock_discover_command_creates_one_review_draft(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'discover.db').as_posix()}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DRAFTS_DIR", str(tmp_path / "drafts"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("CONFIG_DIR", str(PROJECT_ROOT / "config"))
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("PUBLISH_ENABLED", "false")
    monkeypatch.setenv("AUTO_PUBLISH", "false")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    get_settings.cache_clear()
    reset_database_state()
    runner = CliRunner()
    try:
        initialized = runner.invoke(app, ["init"])
        assert initialized.exit_code == 0, initialized.output

        discovered = runner.invoke(app, ["ideas", "discover"])
        assert discovered.exit_code == 0, discovered.output
        payload = json.loads(discovered.stdout)

        assert payload["status"] == "draft_created"
        assert payload["draft_id"]
        assert payload["fetched_posts"] == 2
        assert payload["maximum_estimated_x_cost_usd"] == "0"
    finally:
        get_settings.cache_clear()
        reset_database_state()
