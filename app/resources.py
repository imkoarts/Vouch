"""Resolve source-tree or wheel-bundled runtime resources safely."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

PACKAGE_DIR: Final = Path(__file__).resolve().parent
SOURCE_ROOT: Final = PACKAGE_DIR.parent
BUNDLED_ROOT: Final = PACKAGE_DIR / "_bundled"
_REQUIRED_CONFIG_FILES: Final = frozenset(
    {"content_profile.yml", "content_sources.yml", "cost_estimates.yml", "runtime.yml"}
)


def _has_default_configuration(directory: Path) -> bool:
    return directory.is_dir() and all(
        (directory / filename).is_file() for filename in _REQUIRED_CONFIG_FILES
    )


def resolve_default_config_dir() -> Path:
    """Locate checked-in or wheel-bundled default YAML configuration.

    An explicit ``CONFIG_DIR`` is handled by Pydantic Settings and therefore
    takes precedence over this default factory.  For an unconfigured process,
    prefer a complete configuration in the current working directory, then a
    source checkout, then the package's immutable bundled defaults.
    """

    candidates = (
        Path.cwd() / "config",
        SOURCE_ROOT / "config",
        BUNDLED_ROOT / "config",
    )
    for candidate in candidates:
        if _has_default_configuration(candidate):
            return candidate.resolve()
    return (Path.cwd() / "config").resolve()


def resolve_alembic_config_path(explicit: str | Path | None = None) -> Path:
    """Locate Alembic configuration for source, Docker, or installed wheel use."""

    configured = explicit or os.environ.get("ALEMBIC_CONFIG")
    candidates = (
        Path(configured).expanduser() if configured else None,
        Path.cwd() / "alembic.ini",
        SOURCE_ROOT / "alembic.ini",
        BUNDLED_ROOT / "alembic.ini",
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise RuntimeError("Alembic configuration is missing")


__all__ = [
    "BUNDLED_ROOT",
    "PACKAGE_DIR",
    "SOURCE_ROOT",
    "resolve_alembic_config_path",
    "resolve_default_config_dir",
]
