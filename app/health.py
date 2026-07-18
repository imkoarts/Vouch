"""Shared HTTP health contract and database readiness checks.

The response validator is intentionally imported by both tests and the Docker
smoke probe.  Keeping the accepted payloads here prevents the two entry points
from silently drifting apart again.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, inspect, text

from app.resources import resolve_alembic_config_path

LIVE_PATH: Final = "/health/live"
READY_PATH: Final = "/health/ready"

HealthFailure = Literal[
    "database_unavailable",
    "database_not_writable",
    "schema_not_at_head",
]


class HealthContractError(ValueError):
    """Raised when an HTTP health response violates the public contract."""


def live_payload() -> dict[str, str]:
    """Return the exact public liveness payload."""

    return {"status": "alive"}


def ready_payload() -> dict[str, str]:
    """Return the exact successful readiness payload."""

    return {"status": "ready"}


def not_ready_payload(reason: HealthFailure) -> dict[str, str]:
    """Return a bounded diagnostic without exception text or configuration."""

    return {"status": "not_ready", "reason": reason}


def validate_health_response(
    path: str,
    status_code: int,
    payload: Mapping[str, object],
) -> None:
    """Validate a successful liveness/readiness response exactly."""

    if path == LIVE_PATH:
        expected = live_payload()
    elif path == READY_PATH:
        expected = ready_payload()
    else:
        raise HealthContractError(f"Unsupported health endpoint: {path}")
    if status_code != 200 or dict(payload) != expected:
        raise HealthContractError(
            f"{path} contract mismatch: status={status_code}, payload={dict(payload)!r}"
        )


def resolve_alembic_config() -> Path:
    """Find Alembic config in an installed container or a source checkout."""

    return resolve_alembic_config_path()


@lru_cache(maxsize=8)
def _alembic_heads(config_path: Path) -> frozenset[str]:
    """Read heads for one resolved deployment config path."""

    configuration = Config(str(config_path))
    heads = frozenset(ScriptDirectory.from_config(configuration).get_heads())
    if not heads:
        raise RuntimeError("Alembic migration head is missing")
    return heads


def alembic_heads() -> frozenset[str]:
    """Return the migration heads declared by the active deployment."""

    return _alembic_heads(resolve_alembic_config())


def database_readiness(engine: Engine) -> HealthFailure | None:
    """Check connectivity, exact migration head, and a rollback-only DB write."""

    try:
        with engine.connect() as connection:
            tables = set(inspect(connection).get_table_names())
            if "alembic_version" not in tables:
                return "schema_not_at_head"
            revisions = frozenset(
                row[0]
                for row in connection.execute(text("SELECT version_num FROM alembic_version"))
            )
            if revisions != alembic_heads():
                return "schema_not_at_head"

            # A no-op UPDATE acquires the same write permission/lock required by
            # normal transactions.  The connection context rolls it back, so a
            # readiness probe never mutates authoritative revision state.
            result = connection.execute(
                text("UPDATE alembic_version SET version_num = version_num")
            )
            if result.rowcount != len(revisions):
                return "database_not_writable"
    except Exception:  # Boundary deliberately maps details to a safe fixed code.
        return "database_unavailable"
    return None
