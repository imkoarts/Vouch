"""Allowlisted, write-only updates for the local runtime environment file."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path

from app.config import Settings

EDITABLE_ENV_FIELDS: dict[str, str] = {
    "x_consumer_key": "X_CONSUMER_KEY",
    "x_consumer_secret": "X_CONSUMER_SECRET",
    "x_access_token": "X_ACCESS_TOKEN",
    "x_access_token_secret": "X_ACCESS_TOKEN_SECRET",
    "x_bearer_token": "X_BEARER_TOKEN",
    "x_client_id": "X_CLIENT_ID",
    "x_client_secret": "X_CLIENT_SECRET",
    "x_user_id": "X_USER_ID",
    "openai_api_key": "OPENAI_API_KEY",
    "xai_api_key": "XAI_API_KEY",
    "heygen_api_key": "HEYGEN_API_KEY",
    "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
}

_SAFE_VALUE = re.compile(r"^[^\r\n\x00]*$")


def credential_status(settings: Settings) -> dict[str, bool]:
    """Return presence flags only; raw values never cross the web boundary."""

    result: dict[str, bool] = {}
    for field in EDITABLE_ENV_FIELDS:
        value = getattr(settings, field)
        if hasattr(value, "get_secret_value"):
            value = value.get_secret_value()
        result[field] = value is not None and str(value).strip() != ""
    return result


def update_environment_file(
    path: Path,
    *,
    values: Mapping[str, str],
    clear: set[str] | None = None,
) -> None:
    """Update only explicitly allowlisted names with an atomic same-directory replace."""

    unknown = (set(values) | set(clear or ())) - set(EDITABLE_ENV_FIELDS)
    if unknown:
        raise ValueError("Unsupported credential setting")
    normalized: dict[str, str | None] = {}
    for field, value in values.items():
        candidate = value.strip()
        if not candidate:
            continue
        if not _SAFE_VALUE.fullmatch(candidate):
            raise ValueError("Credential values must be single-line text")
        normalized[EDITABLE_ENV_FIELDS[field]] = candidate
    for field in clear or ():
        normalized[EDITABLE_ENV_FIELDS[field]] = None
    if not normalized:
        return

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    output: list[str] = []
    handled: set[str] = set()
    for line in existing.splitlines():
        key = (
            line.split("=", 1)[0].strip()
            if "=" in line and not line.lstrip().startswith("#")
            else ""
        )
        if key in normalized:
            handled.add(key)
            replacement = normalized[key]
            if replacement is not None:
                output.append(f"{key}={replacement}")
            continue
        output.append(line)
    for key, pending_value in normalized.items():
        if key not in handled and pending_value is not None:
            output.append(f"{key}={pending_value}")

    descriptor, temporary = tempfile.mkstemp(prefix=".env.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(output).rstrip() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        with suppress(OSError):
            os.chmod(path, 0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)
