"""Atomic local state for short Telegram conversations and update offsets."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any


class TelegramStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"offset": None, "pending": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"offset": None, "pending": {}}
        return payload if isinstance(payload, dict) else {"offset": None, "pending": {}}

    def _write(self, payload: dict[str, Any]) -> None:
        descriptor, name = tempfile.mkstemp(
            prefix=".telegram-state-", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.path)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(name)
            raise

    def offset(self) -> int | None:
        value = self._read().get("offset")
        return value if isinstance(value, int) else None

    def set_offset(self, offset: int) -> None:
        payload = self._read()
        payload["offset"] = offset
        self._write(payload)

    def set_pending(self, user_id: int, *, action: str, draft_id: str) -> None:
        payload = self._read()
        pending = payload.setdefault("pending", {})
        if not isinstance(pending, dict):
            pending = {}
            payload["pending"] = pending
        pending[str(user_id)] = {"action": action, "draft_id": draft_id}
        self._write(payload)

    def pop_pending(self, user_id: int) -> dict[str, str] | None:
        payload = self._read()
        pending = payload.get("pending")
        if not isinstance(pending, dict):
            return None
        value = pending.pop(str(user_id), None)
        self._write(payload)
        if not isinstance(value, dict):
            return None
        action = value.get("action")
        draft_id = value.get("draft_id")
        if isinstance(action, str) and isinstance(draft_id, str):
            return {"action": action, "draft_id": draft_id}
        return None

    def clear_pending(self, user_id: int) -> bool:
        payload = self._read()
        pending = payload.get("pending")
        if not isinstance(pending, dict) or str(user_id) not in pending:
            return False
        del pending[str(user_id)]
        self._write(payload)
        return True
