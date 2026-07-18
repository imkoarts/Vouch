"""Regression tests for the single shared HTTP health contract."""

from pathlib import Path

import pytest

from app.health import (
    LIVE_PATH,
    READY_PATH,
    HealthContractError,
    alembic_heads,
    resolve_alembic_config,
    validate_health_response,
)


def test_health_contract_accepts_only_exact_success_payloads() -> None:
    validate_health_response(LIVE_PATH, 200, {"status": "alive"})
    validate_health_response(READY_PATH, 200, {"status": "ready"})

    with pytest.raises(HealthContractError):
        validate_health_response(LIVE_PATH, 200, {"status": "healthy"})
    with pytest.raises(HealthContractError):
        validate_health_response(READY_PATH, 200, {"status": "ready", "extra": True})
    with pytest.raises(HealthContractError):
        validate_health_response(READY_PATH, 503, {"status": "not_ready"})


def test_health_resolves_explicit_alembic_config_outside_source_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALEMBIC_CONFIG", str(root / "alembic.ini"))

    assert resolve_alembic_config() == (root / "alembic.ini").resolve()
    assert alembic_heads()
