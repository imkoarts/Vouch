"""Apply schema migrations before starting the container's web process."""

from __future__ import annotations

import os
import subprocess
import sys

from sqlalchemy.engine import make_url


def validate_database_url(database_url: str) -> None:
    """Reject SQLite URLs that cannot survive the migration subprocess."""

    url = make_url(database_url)
    if url.get_backend_name() != "sqlite":
        return
    database = url.database or ""
    if database in {"", ":memory:"} or (database.startswith("file:") and "mode=memory" in database):
        raise RuntimeError(
            "Docker startup requires file-backed SQLite; sqlite:///:memory: cannot be "
            "shared between migration and web processes."
        )


def migrate() -> None:
    """Upgrade the configured database to the source tree's Alembic head."""

    database_url = os.environ.get("DATABASE_URL", "sqlite:///./data/app.db")
    validate_database_url(database_url)
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        check=True,
    )


def main() -> int:
    """Migrate, then replace this process with the configured container command."""

    migrate()
    command = list(sys.argv[1:]) or [sys.executable, "-m", "app.cli", "serve"]
    if command[0] in {"python", "python3"}:
        command[0] = sys.executable
    os.execvp(command[0], command)  # noqa: S606 - Docker command is operator-controlled.
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
