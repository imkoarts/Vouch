"""Installed-wheel entry point for the frozen external semantic holdout runner."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import NoReturn


def main() -> NoReturn:
    namespace = runpy.run_path(
        str(Path(__file__).with_name("_bundled") / "run_external_semantic_holdout_v5.py")
    )
    runner_main = namespace["main"]
    raise SystemExit(runner_main())


if __name__ == "__main__":
    main()
