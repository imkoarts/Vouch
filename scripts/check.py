"""Cross-platform equivalent of ``make check``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(command: list[str]) -> int:
    """Run one check without invoking a shell."""

    print(f"Running: {' '.join(command)}", flush=True)
    # Every argument is an in-repository constant; no user input or shell is used.
    completed = subprocess.run(command, cwd=ROOT, check=False)  # noqa: S603
    return completed.returncode


def main() -> int:
    """Run lint, type checking, and tests in the required order."""

    commands = (
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [sys.executable, "-m", "mypy", "app"],
        [sys.executable, "-m", "pytest"],
    )
    for command in commands:
        return_code = run(command)
        if return_code != 0:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
