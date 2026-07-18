"""Run the real CLI commands against one isolated mock workspace."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_cli(arguments: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(  # noqa: S603 - fixed interpreter/module, no shell
        [sys.executable, "-m", "app.cli", *arguments],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"CLI {' '.join(arguments)} failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return result


def _first_json_object(output: str) -> dict[str, object]:
    start = output.find("{")
    if start < 0:
        raise AssertionError(f"No JSON object in output: {output}")
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(output[start:])
    assert isinstance(value, dict)
    return value


def test_complete_cli_mock_flow_is_durable(tmp_path: Path) -> None:
    database = tmp_path / "cli.db"
    drafts = tmp_path / "drafts"
    data = tmp_path / "data"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUTF8": "1",
            "APP_ENV": "test",
            "APP_HOST": "127.0.0.1",
            "DATABASE_URL": f"sqlite:///{database.as_posix()}",
            "DATA_DIR": str(data),
            "DRAFTS_DIR": str(drafts),
            "CONFIG_DIR": str(PROJECT_ROOT / "config"),
            "MOCK_MODE": "true",
            "LLM_MODE": "mock",
            "DRAFT_PROVIDER": "mock",
            "CRITIC_PROVIDER": "mock",
            "FINAL_PROVIDER": "mock",
            "PUBLISH_ENABLED": "true",
            "AUTO_PUBLISH": "false",
            "HEYGEN_MODE": "disabled",
            "X_USER_ID": "mock-user-001",
        }
    )

    _run_cli(["init"], env)
    collected = _first_json_object(_run_cli(["ideas", "collect"], env).stdout)
    idea_ids = collected["idea_ids"]
    assert isinstance(idea_ids, list) and idea_ids
    _run_cli(["ideas", "list"], env)
    generated = _first_json_object(
        _run_cli(["drafts", "generate", "--idea", str(idea_ids[0])], env).stdout
    )
    draft_id = str(generated["draft_id"])
    content_hash = str(generated["content_hash"])
    bundle = next(drafts.glob(f"*/{draft_id}"))
    metadata_path = bundle / "metadata.json"
    stale_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stale_metadata["status"] = "stale-synthetic-value"
    metadata_path.write_text(json.dumps(stale_metadata), encoding="utf-8", newline="\n")
    rebuilt = _first_json_object(_run_cli(["drafts", "rebuild-artifacts", draft_id], env).stdout)
    assert rebuilt["rebuilt_draft_ids"] == [draft_id]
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["status"] == ("needs_review")
    _run_cli(["drafts", "show", draft_id], env)
    _run_cli(["drafts", "approve", draft_id], env)
    phrase = f"PUBLISH {draft_id} {content_hash[:12]}"
    published = _run_cli(["drafts", "publish", draft_id, "--confirm", phrase], env)
    assert '"mock": true' in published.stdout.casefold()

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT status, x_post_id FROM publications WHERE draft_id = ?",
            (draft_id,),
        ).fetchone()
        draft_row = connection.execute(
            "SELECT status, current_content_hash, idea_id FROM drafts WHERE id = ?",
            (draft_id,),
        ).fetchone()
        source_count = connection.execute(
            "SELECT COUNT(*) FROM idea_sources WHERE idea_id = ?",
            (draft_row[2],),
        ).fetchone()[0]
    assert row is not None
    assert row[0] == "succeeded"
    assert row[1]
    assert draft_row is not None
    assert draft_row[:2] == ("published", content_hash)
    assert source_count > 0
    metadata_projection = json.loads((bundle / "metadata.json").read_text("utf-8"))
    publication_projection = json.loads(
        next(drafts.glob(f"*/{draft_id}/publication.json")).read_text(encoding="utf-8")
    )
    markdown_projection = (bundle / "draft.md").read_text(encoding="utf-8")
    assert metadata_projection["status"] == draft_row[0]
    assert metadata_projection["content_hash"] == draft_row[1]
    assert metadata_projection["source_count"] == source_count
    assert "status: published" in markdown_projection
    assert f"source_count: {source_count}" in markdown_projection
    assert f"content_hash: {content_hash}" in markdown_projection
    assert publication_projection["status"] == "published"
    assert publication_projection["mock"] is True
    assert publication_projection["post_ids"]
    assert publication_projection["current_content_hash"] == content_hash
    assert publication_projection["status_counts"] == {"succeeded": 1}
