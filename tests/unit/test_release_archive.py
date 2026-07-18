"""Regression tests for the allowlist-only deterministic release builder."""

from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.build_release import (
    ROOT,
    ReleaseArchiveError,
    build_manifest,
    build_release,
    is_forbidden_path,
    validate_manifest,
    validate_public_release_content,
)


def test_release_manifest_excludes_repository_and_runtime_artifacts() -> None:
    manifest = build_manifest(ROOT)
    names = {path.as_posix() for path in manifest}

    assert "app/main.py" in names
    assert "requirements.lock" in names
    assert "LICENSE" in names
    assert "launcher.py" in names
    assert "START_VOUCH.bat" in names
    assert "CONFIGURE_VOUCH.bat" in names
    assert "config/runtime.yml" in names
    assert ".codex/config.toml" in names
    assert "docs/personal-humanizer/SKILL.md" in names
    assert "docs/personal-humanizer/references/x-post-contract.md" in names
    assert "scripts/setup_local.ps1" in names
    assert "scripts/start_local.ps1" in names
    assert "requirements-dev.lock" in names
    assert "benchmarks/scenarios.json" in names
    assert all(not is_forbidden_path(path) for path in manifest)
    assert all(".venv" not in path.parts for path in manifest)
    assert Path("audit.md") not in manifest
    assert Path("codex_xbot_revision4_prompt.md") not in manifest
    assert all(path.suffix.casefold() != ".db" for path in manifest)


def test_release_manifest_validator_fails_closed_for_injected_path() -> None:
    with pytest.raises(ReleaseArchiveError, match="Forbidden release paths"):
        validate_manifest([Path("app/main.py"), Path("app/.env")])


def test_release_archive_matches_manifest_and_is_deterministic(tmp_path: Path) -> None:
    first, manifest = build_release(ROOT, tmp_path / "first.zip")
    second, second_manifest = build_release(ROOT, tmp_path / "second.zip")

    assert manifest == second_manifest
    assert first.read_bytes() == second.read_bytes()
    with ZipFile(first) as archive:
        assert archive.namelist() == [path.as_posix() for path in manifest]
        for name in ("START_VOUCH.bat", "CONFIGURE_VOUCH.bat"):
            raw = archive.read(name)
            raw.decode("ascii")
            assert b"\r\n" in raw
            assert b"\n" not in raw.replace(b"\r\n", b"")


def test_project_codex_mcp_configuration_is_secret_free_and_xapi_is_opt_in() -> None:
    content = (ROOT / ".codex" / "config.toml").read_text(encoding="utf-8")

    assert 'url = "https://docs.x.com/mcp"' in content
    assert '"https://api.x.com/mcp"' in content
    assert "[mcp_servers.xapi]" in content
    assert "enabled = false" in content
    assert "client_secret" not in content.casefold()
    assert "bearer_token" not in content.casefold()


def test_public_release_content_validation_passes_for_project_manifest() -> None:
    manifest = build_manifest(ROOT)

    validate_public_release_content(ROOT, manifest)


def test_public_release_content_validation_rejects_identity_and_language(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    runtime = config / "runtime.yml"
    private_text = (
        "\u041f\u0440\u0438\u0432\u0430\u0442\u043d\u044b\u0439 \u0442\u0435\u043a\u0441\u0442"
    )
    runtime.write_text(
        f"telegram:\n  operator_user_id: 987654321\nmessage: {private_text}\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseArchiveError, match=r"operator_user_id|Cyrillic"):
        validate_public_release_content(tmp_path, (Path("config/runtime.yml"),))


def test_public_release_content_validation_rejects_secret_like_production_value(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    secret_file = docs / "setup.md"
    synthetic_token = "1234567890:" + "abcdefghijklmnopqrstuvwxyz_ABCDE"
    secret_file.write_text(
        f"TELEGRAM_BOT_TOKEN={synthetic_token}",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseArchiveError, match="Telegram token-like"):
        validate_public_release_content(tmp_path, (Path("docs/setup.md"),))
