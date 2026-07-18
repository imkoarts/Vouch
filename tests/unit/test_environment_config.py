"""Credential settings are write-only and constrained to a fixed local allowlist."""

from pathlib import Path

import pytest

from app.config import Settings
from app.services.environment_config import credential_status, update_environment_file


def test_environment_update_preserves_unrelated_values_and_never_returns_secrets(
    tmp_path: Path,
) -> None:
    target = tmp_path / ".env"
    target.write_text("APP_ENV=test\nOPENAI_API_KEY=old-value\n", encoding="utf-8")

    update_environment_file(
        target,
        values={"openai_api_key": "synthetic-replacement"},
    )

    content = target.read_text(encoding="utf-8")
    assert "APP_ENV=test" in content
    assert "OPENAI_API_KEY=synthetic-replacement" in content
    assert "LOCAL_OTP_DELIVERY" not in content
    status = credential_status(
        Settings(_env_file=None, app_env="test", openai_api_key="synthetic-replacement")
    )
    assert status["openai_api_key"] is True
    assert "synthetic-replacement" not in repr(status)


def test_environment_update_rejects_unknown_or_multiline_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        update_environment_file(tmp_path / ".env", values={"database_url": "sqlite:///other"})
    with pytest.raises(ValueError, match="single-line"):
        update_environment_file(
            tmp_path / ".env", values={"openai_api_key": "first\nPUBLISH_ENABLED=true"}
        )
