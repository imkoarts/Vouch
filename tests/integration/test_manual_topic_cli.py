from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.config import get_settings
from app.database import reset_database_state

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_generate_topic_creates_reviewable_draft_without_x_read(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'topic.db').as_posix()}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DRAFTS_DIR", str(tmp_path / "drafts"))
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
        generated = runner.invoke(
            app,
            [
                "drafts",
                "generate-topic",
                "--topic",
                "Practical AI content review",
            ],
        )
        assert generated.exit_code == 0, generated.output
        payload = json.loads(generated.stdout)

        assert payload["status"] == "needs_review"
        assert payload["draft_id"]
        assert payload["content_hash"]
        assert Path(payload["artifact_path"]).is_dir()
        assert generated.stderr == ""
    finally:
        get_settings.cache_clear()
        reset_database_state()
