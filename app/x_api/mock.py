"""Offline X read/write simulator with deterministic IDs and fault injection."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


class AmbiguousPublicationError(RuntimeError):
    """The remote outcome is unknown and must never be automatically retried."""


class MockXClient:
    """A local fake; no method opens a socket or calls a provider SDK."""

    account_id = "mock-user-001"
    account_username = "mock_operator"

    def __init__(
        self,
        *,
        ambiguous_positions: set[int] | None = None,
        fail_positions: set[int] | None = None,
    ) -> None:
        self.ambiguous_positions = ambiguous_positions or set()
        self.fail_positions = fail_positions or set()
        self.write_calls: list[dict[str, Any]] = []

    async def get_me(self) -> Mapping[str, Any]:
        return {"id": self.account_id, "username": self.account_username}

    async def get_home_timeline(self, user_id: str) -> Sequence[Mapping[str, Any]]:
        del user_id
        now = datetime.now(UTC)
        return (
            {
                "id": "910000000000000001",
                "text": "Synthetic home item about making a manual review step visible.",
                "author_id": "710000000000000001",
                "username": "mock_builder",
                "lang": "en",
                "created_at": (now - timedelta(hours=1)).isoformat(),
                "conversation_id": "910000000000000001",
                "public_metrics": {
                    "like_count": 15,
                    "retweet_count": 2,
                    "reply_count": 1,
                },
            },
        )

    async def get_user_by_username(self, username: str) -> Mapping[str, Any]:
        normalized = username.strip().removeprefix("@").casefold()
        digest = int(hashlib.sha256(normalized.encode()).hexdigest()[:12], 16)
        return {
            "id": str(700000000000000000 + digest % 10000000000000000),
            "username": normalized,
            "name": normalized.replace("_", " ").title(),
        }

    async def get_user_posts(
        self,
        user_id: str,
        *,
        max_results: int,
        start_time: datetime,
        exclude_replies: bool = True,
        exclude_retweets: bool = True,
    ) -> tuple[Mapping[str, Any], ...]:
        del start_time, exclude_replies, exclude_retweets
        now = datetime.now(UTC)
        digest = int(hashlib.sha256(user_id.encode()).hexdigest()[:16], 16)
        post_id = str(925000000000000000 + digest % 10000000000000000)
        return (
            {
                "id": post_id,
                "text": "Synthetic selected-account post used only in mock mode.",
                "author_id": user_id,
                "username": "mock_selected_account",
                "lang": "en",
                "created_at": (now - timedelta(minutes=30)).isoformat(),
                "conversation_id": post_id,
                "public_metrics": {"like_count": 9, "retweet_count": 1},
            },
        )[:max_results]

    async def recent_search(self, query: str) -> Sequence[Mapping[str, Any]]:
        del query
        raise RuntimeError("X Recent Search is disabled by product configuration")

    async def get_post(self, post_id: str) -> Mapping[str, Any]:
        return {
            "id": post_id,
            "text": "Synthetic manually imported post used only in mock mode.",
            "author_id": "730000000000000001",
            "username": "mock_import",
            "lang": "en",
            "created_at": datetime.now(UTC).isoformat(),
            "conversation_id": post_id,
            "public_metrics": {},
        }

    async def upload_media(self, path: Path) -> str:
        payload = await asyncio.to_thread(path.read_bytes)
        digest = hashlib.sha256(path.name.encode() + payload).hexdigest()
        return str(940000000000000000 + int(digest[:12], 16) % 10000000000000000)

    async def create_post(
        self,
        *,
        text: str,
        reply_to_post_id: str | None,
        media_ids: tuple[str, ...] = (),
        quote_post_id: str | None = None,
        position: int = 0,
    ) -> Mapping[str, Any]:
        call = {
            "text": text,
            "reply_to_post_id": reply_to_post_id,
            "media_ids": media_ids,
            "quote_post_id": quote_post_id,
            "position": position,
        }
        self.write_calls.append(call)
        if position in self.ambiguous_positions:
            raise AmbiguousPublicationError("Mock outcome intentionally ambiguous")
        if position in self.fail_positions:
            raise RuntimeError("Mock publication intentionally failed before acceptance")
        digest = hashlib.sha256(repr(call).encode()).hexdigest()
        post_id = str(930000000000000000 + int(digest[:12], 16) % 10000000000000000)
        return {"id": post_id, "mock": True}

    async def delete_post(self, post_id: str) -> Mapping[str, Any]:
        return {"id": post_id, "deleted": True, "mock": True}
