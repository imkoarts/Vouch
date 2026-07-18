"""Build a deterministic, allowlist-only source release archive."""

from __future__ import annotations

import argparse
import re
import stat
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = ROOT / "dist" / "vouch-release.zip"

ALLOWED_ROOT_FILES = frozenset(
    {
        ".dockerignore",
        ".env.example",
        ".gitattributes",
        ".gitignore",
        ".pre-commit-config.yaml",
        "AGENTS.md",
        "BUILD_EXE.bat",
        "Dockerfile",
        "LICENSE",
        "CONFIGURE_VOUCH.bat",
        "Makefile",
        "START_DESKTOP.bat",
        "START_WEB.bat",
        "START_VOUCH.bat",
        "README.md",
        "SECURITY.md",
        "Vouch.spec",
        "alembic.ini",
        "desktop.py",
        "launcher.py",
        "docker-compose.yml",
        "pyproject.toml",
        "requirements-build.in",
        "requirements-build.lock",
        "requirements-desktop.lock",
        "requirements-dev.lock",
        "requirements-exe-build.lock",
        "requirements.lock",
    }
)

# Individual binary files outside the allowed directories can be listed here.
# At the moment, no additional binary files are required.
ALLOWED_BINARY_FILES: tuple[Path, ...] = ()

ALLOWED_DIRECTORIES = (
    ".codex",
    ".github",
    "alembic",
    "app",
    "config",
    "deploy",
    "docs",
    "benchmarks",
    "scripts",
    "tests",
)

FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "backups",
        "build",
        "data",
        "dist",
        "drafts",
        "htmlcov",
        "media",
        "venv",
    }
)

FORBIDDEN_SUFFIXES = frozenset(
    {
        ".db",
        ".key",
        ".pem",
        ".pyc",
        ".pyo",
        ".sqlite",
        ".sqlite3",
        ".zip",
    }
)

TEXT_SUFFIXES = frozenset(
    {
        ".bat",
        ".cjs",
        ".css",
        ".example",
        ".html",
        ".ini",
        ".js",
        ".json",
        ".md",
        ".ps1",
        ".py",
        ".spec",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)

CYRILLIC_PATTERN = re.compile(r"[\u0400-\u04FF]")
WINDOWS_USER_PATH_PATTERN = re.compile(r"(?i)[A-Z]:\\Users\\[^\\\r\n]+")
TELEGRAM_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9])\d{8,12}:[A-Za-z0-9_-]{30,}")
API_SECRET_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:sk|xai)-[A-Za-z0-9_-]{20,}",
    re.I,
)
AUTHENTICATED_PROXY_PATTERN = re.compile(
    r"https?://[^/\s:@{}]+:[^@\s/{}]+@(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?",
    re.I,
)
NON_NULL_OPERATOR_PATTERN = re.compile(r"(?m)^\s*operator_user_id:\s*(?!null\s*$)[1-9]\d*\s*$")


class ReleaseArchiveError(RuntimeError):
    """Raised when the release allowlist or archive boundary is violated."""


def is_forbidden_path(relative_path: Path) -> bool:
    """Return whether a project-relative path is forbidden in a release."""

    pure = PurePosixPath(relative_path.as_posix())
    lowered_parts = tuple(part.casefold() for part in pure.parts)
    name = pure.name.casefold()

    if not pure.parts or pure.is_absolute() or ".." in pure.parts:
        return True

    if any(part in FORBIDDEN_PARTS for part in lowered_parts):
        return True

    if name == ".env.example":
        return False

    if name == ".env" or name.startswith(".env."):
        return True

    if name in {"prompt.txt", "secrets.yml", "secrets.yaml"}:
        return True

    if "audit" in name and pure.suffix.casefold() in {".md", ".txt"}:
        return True

    return pure.suffix.casefold() in FORBIDDEN_SUFFIXES


def validate_manifest(manifest: Sequence[Path]) -> None:
    """Fail closed if a forbidden or duplicate path entered staging."""

    normalized = [Path(PurePosixPath(path.as_posix())) for path in manifest]

    forbidden = sorted(path.as_posix() for path in normalized if is_forbidden_path(path))

    duplicates = sorted(path.as_posix() for path in set(normalized) if normalized.count(path) > 1)

    if forbidden:
        raise ReleaseArchiveError(f"Forbidden release paths: {', '.join(forbidden)}")

    if duplicates:
        raise ReleaseArchiveError(f"Duplicate release paths: {', '.join(duplicates)}")


def build_manifest(root: Path = ROOT) -> tuple[Path, ...]:
    """Collect regular project files through an explicit allowlist."""

    root = root.resolve()
    manifest: set[Path] = set()

    missing = sorted(name for name in ALLOWED_ROOT_FILES if not (root / name).is_file())

    if missing:
        raise ReleaseArchiveError(f"Required release files are missing: {', '.join(missing)}")

    for name in ALLOWED_ROOT_FILES:
        manifest.add(Path(name))

    for relative in ALLOWED_BINARY_FILES:
        source = root / relative

        if not source.is_file():
            raise ReleaseArchiveError(f"Required release file is missing: {relative.as_posix()}")

        manifest.add(relative)

    for directory_name in ALLOWED_DIRECTORIES:
        directory = root / directory_name

        if not directory.is_dir():
            raise ReleaseArchiveError(f"Required release directory is missing: {directory_name}")

        for source in directory.rglob("*"):
            if source.is_symlink():
                raise ReleaseArchiveError(f"Symlink is forbidden in release: {source}")

            if not source.is_file():
                continue

            relative = source.relative_to(root)

            if not is_forbidden_path(relative):
                manifest.add(relative)

    result = tuple(
        sorted(
            manifest,
            key=lambda path: path.as_posix(),
        )
    )

    validate_manifest(result)
    return result


def validate_public_release_content(
    root: Path,
    manifest: Sequence[Path],
) -> None:
    """Reject identity or secret material unsuitable for a public release."""

    violations: list[str] = []

    for relative in manifest:
        if relative.suffix.casefold() not in TEXT_SUFFIXES and relative.name != ".env.example":
            continue

        raw = (root / relative).read_bytes()

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("ascii")
            except UnicodeDecodeError:
                violations.append(f"{relative.as_posix()}: text file is not UTF-8/ASCII")
                continue

        if CYRILLIC_PATTERN.search(text):
            violations.append(f"{relative.as_posix()}: contains Cyrillic text")

        if WINDOWS_USER_PATH_PATTERN.search(text):
            violations.append(f"{relative.as_posix()}: contains a local Windows user path")

        if relative.as_posix() == "config/runtime.yml" and NON_NULL_OPERATOR_PATTERN.search(text):
            violations.append("config/runtime.yml: Telegram operator_user_id must be null")

        # Synthetic credential fixtures are allowed only under tests.
        # Production files, documentation, configuration, and launchers
        # must remain credential-free.
        if not relative.parts or relative.parts[0] != "tests":
            if TELEGRAM_TOKEN_PATTERN.search(text):
                violations.append(f"{relative.as_posix()}: contains a Telegram token-like value")

            if API_SECRET_PATTERN.search(text):
                violations.append(f"{relative.as_posix()}: contains an API key-like value")

            if AUTHENTICATED_PROXY_PATTERN.search(text):
                violations.append(f"{relative.as_posix()}: contains proxy credentials")

    if violations:
        raise ReleaseArchiveError("Public release content violations: " + "; ".join(violations))


def build_release(
    root: Path = ROOT,
    destination: Path = DEFAULT_ARCHIVE,
) -> tuple[Path, tuple[Path, ...]]:
    """Write a deterministic ZIP from the validated allowlist manifest."""

    root = root.resolve()
    destination = destination.resolve()

    manifest = build_manifest(root)
    validate_public_release_content(root, manifest)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.unlink(missing_ok=True)

    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for relative in manifest:
                source = (root / relative).resolve(strict=True)

                if root not in source.parents:
                    raise ReleaseArchiveError(f"Release source escaped project root: {relative}")

                info = zipfile.ZipInfo(
                    relative.as_posix(),
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o644) << 16

                archive.writestr(
                    info,
                    source.read_bytes(),
                    compresslevel=9,
                )

        temporary.replace(destination)

    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return destination, manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="archive path (default: dist/vouch-release.zip)",
    )

    return parser.parse_args()


def main() -> int:
    """Build the release archive and print the result."""

    arguments = parse_args()

    archive, manifest = build_release(
        destination=arguments.output,
    )

    print(
        f"Release archive: {archive} "
        f"({archive.stat().st_size} bytes, "
        f"{len(manifest)} allowlisted files; "
        "forbidden scan passed)"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
