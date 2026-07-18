"""Create an atomic, integrity-checked SQLite backup using the online backup API."""

from __future__ import annotations

import argparse
import os
import sqlite3
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class BackupError(RuntimeError):
    """Raised when a safe backup cannot be completed."""


def default_destination() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "backups" / f"app-{timestamp}.db"


def backup_database(source: Path, destination: Path, *, overwrite: bool = False) -> Path:
    """Back up a live SQLite database to an atomically replaced destination."""

    source = source.expanduser().resolve(strict=True)
    destination = destination.expanduser().resolve()
    if not source.is_file():
        raise BackupError("The SQLite source must be a regular file.")
    if source == destination:
        raise BackupError("Source and destination must be different files.")
    if destination.exists() and not overwrite:
        raise BackupError("Destination already exists; use --overwrite explicitly.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        with (
            sqlite3.connect(str(source)) as source_connection,
            sqlite3.connect(str(temporary_path)) as destination_connection,
        ):
            source_connection.execute("PRAGMA query_only=ON")
            source_connection.backup(destination_connection, pages=256, sleep=0.05)
            check = destination_connection.execute("PRAGMA integrity_check").fetchone()
            if check is None or check[0] != "ok":
                raise BackupError("SQLite integrity_check failed for the backup.")

        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        with suppress(OSError):
            temporary_path.chmod(0o600)
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a consistent SQLite backup without copying a live database file."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data" / "app.db",
        help="SQLite database path (default: data/app.db)",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Output path (default: timestamped file under backups/)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination explicitly.",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    destination = arguments.destination or default_destination()
    try:
        result = backup_database(
            arguments.source,
            destination,
            overwrite=arguments.overwrite,
        )
    except (BackupError, FileNotFoundError, sqlite3.Error, OSError) as exc:
        print(f"Backup failed: {exc}")
        return 1
    print(f"Backup created: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
