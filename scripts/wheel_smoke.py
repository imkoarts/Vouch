"""Verify that the built wheel works outside the source checkout."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WHEEL_DIRECTORY = ROOT / "dist" / "wheel-smoke"


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(  # noqa: S603 - explicit argv, no shell or user-provided executable.
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _resolve_wheel(explicit: Path | None) -> Path:
    if explicit is not None:
        wheel = explicit.resolve(strict=True)
        if wheel.suffix != ".whl":
            raise ValueError("Wheel smoke input must be a .whl file")
        return wheel
    wheels = sorted(DEFAULT_WHEEL_DIRECTORY.glob("vouch-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            "Expected exactly one wheel in dist/wheel-smoke; build it before running the smoke test"
        )
    return wheels[0].resolve()


def smoke_wheel(wheel: Path) -> None:
    """Install into an isolated target and run migrations plus mock collection."""

    with tempfile.TemporaryDirectory(prefix="vouch-wheel-smoke-") as temporary:
        workspace = Path(temporary)
        site_packages = workspace / "site"
        site_packages.mkdir()
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONPATH": str(site_packages),
                "PYTHONUTF8": "1",
                "APP_ENV": "test",
                "APP_HOST": "127.0.0.1",
                "DATABASE_URL": "sqlite:///./data/wheel-smoke.db",
                "DATA_DIR": "./data",
                "DRAFTS_DIR": "./drafts",
                "MOCK_MODE": "true",
                "LLM_MODE": "mock",
                "DRAFT_PROVIDER": "mock",
                "CRITIC_PROVIDER": "mock",
                "FINAL_PROVIDER": "mock",
                "HEYGEN_MODE": "disabled",
                "PUBLISH_ENABLED": "true",
                "AUTO_PUBLISH": "false",
                "STORE_LLM_PAYLOADS": "false",
            }
        )
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--target",
                str(site_packages),
                str(wheel),
            ],
            cwd=workspace,
            env=environment,
        )
        _run([sys.executable, "-m", "app.cli", "init"], cwd=workspace, env=environment)
        collected = _run(
            [sys.executable, "-m", "app.cli", "ideas", "collect"],
            cwd=workspace,
            env=environment,
        )
        payload = json.loads(collected.stdout)
        idea_ids = payload.get("idea_ids")
        if not isinstance(idea_ids, list) or not idea_ids:
            raise RuntimeError("Installed wheel did not collect mock ideas")

        generated = _run(
            [
                sys.executable,
                "-m",
                "app.cli",
                "drafts",
                "generate",
                "--idea",
                str(idea_ids[0]),
            ],
            cwd=workspace,
            env=environment,
        )
        generated_payload = json.loads(generated.stdout)
        draft_id = str(generated_payload["draft_id"])
        content_hash = str(generated_payload["content_hash"])
        _run(
            [sys.executable, "-m", "app.cli", "drafts", "approve", draft_id],
            cwd=workspace,
            env=environment,
        )
        confirmation = f"PUBLISH {draft_id} {content_hash[:12]}"
        _run(
            [
                sys.executable,
                "-m",
                "app.cli",
                "drafts",
                "publish",
                draft_id,
                "--confirm",
                confirmation,
            ],
            cwd=workspace,
            env=environment,
        )

        database = workspace / "data" / "wheel-smoke.db"
        with closing(sqlite3.connect(database)) as connection:
            revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()
            idea_count = connection.execute("SELECT COUNT(*) FROM ideas").fetchone()
            draft_status = connection.execute(
                "SELECT status FROM drafts WHERE id = ?", (draft_id,)
            ).fetchone()
            publication_status = connection.execute(
                "SELECT status FROM publications WHERE draft_id = ?", (draft_id,)
            ).fetchone()
            activity_table = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
                ("x_activity_events",),
            ).fetchone()
        if revision is None or revision[0] != "0008_local_passwordless":
            raise RuntimeError("Installed wheel did not migrate to the expected Alembic head")
        if activity_table != ("x_activity_events",):
            raise RuntimeError("Installed wheel is missing the X Activity event table")
        if idea_count is None or idea_count[0] < 1:
            raise RuntimeError("Installed wheel did not persist collected ideas")
        if draft_status != ("published",):
            raise RuntimeError("Installed wheel did not complete the mock draft workflow")
        if publication_status != ("succeeded",):
            raise RuntimeError("Installed wheel did not persist a successful mock publication")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, help="explicit wheel path")
    return parser.parse_args()


def main() -> int:
    wheel = _resolve_wheel(parse_args().wheel)
    smoke_wheel(wheel)
    print(f"Installed-wheel smoke passed: {wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
