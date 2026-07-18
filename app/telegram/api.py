"""Small async client for the official Telegram Bot API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx


class TelegramApiError(RuntimeError):
    pass


class TelegramBotApi:
    def __init__(
        self,
        token: str,
        *,
        timeout_seconds: float = 45.0,
        proxy_url: str | None = None,
    ) -> None:
        if not token:
            raise TelegramApiError("Telegram bot token is missing")
        self._base_url = f"https://api.telegram.org/bot{token}"
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 15.0)),
            follow_redirects=False,
            proxy=proxy_url,
            trust_env=False,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _call(
        self,
        method: str,
        *,
        data: Mapping[str, Any] | None = None,
        files: Mapping[str, Any] | None = None,
    ) -> Any:
        try:
            response = await self._client.post(
                f"{self._base_url}/{method}",
                data=dict(data or {}),
                files=files,
            )
        except httpx.HTTPError:
            # The bot token is part of the request URL. Never expose the original
            # transport exception or its request URL to logs/CLI output.
            raise TelegramApiError("Telegram transport request failed") from None
        try:
            payload = response.json()
        except ValueError as exc:
            raise TelegramApiError(f"Telegram returned HTTP {response.status_code}") from exc
        if response.status_code >= 400 or not payload.get("ok"):
            description = str(payload.get("description") or "Telegram API request failed")
            raise TelegramApiError(description)
        return payload.get("result")

    async def get_me(self) -> Mapping[str, Any]:
        result = await self._call("getMe")
        if not isinstance(result, Mapping):
            raise TelegramApiError("Telegram getMe returned an invalid payload")
        return result

    async def delete_webhook(self, *, drop_pending_updates: bool = False) -> None:
        await self._call(
            "deleteWebhook",
            data={"drop_pending_updates": str(drop_pending_updates).lower()},
        )

    async def get_updates(
        self,
        *,
        offset: int | None,
        poll_timeout: int,
    ) -> Sequence[Mapping[str, Any]]:
        data: dict[str, Any] = {
            "timeout": poll_timeout,
            "allowed_updates": '["message","callback_query"]',
        }
        if offset is not None:
            data["offset"] = offset
        result = await self._call("getUpdates", data=data)
        if not isinstance(result, list):
            raise TelegramApiError("Telegram getUpdates returned an invalid payload")
        return tuple(item for item in result if isinstance(item, Mapping))

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_markup: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        import json

        data: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": "true",
        }
        if reply_markup is not None:
            data["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False)
        result = await self._call("sendMessage", data=data)
        if not isinstance(result, Mapping):
            raise TelegramApiError("Telegram sendMessage returned an invalid payload")
        return result

    async def send_photo(
        self,
        chat_id: int,
        path: Path,
        *,
        caption: str | None = None,
    ) -> Mapping[str, Any]:
        data: dict[str, Any] = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption[:1024]
        with path.open("rb") as handle:
            result = await self._call(
                "sendPhoto",
                data=data,
                files={"photo": (path.name, handle, "application/octet-stream")},
            )
        if not isinstance(result, Mapping):
            raise TelegramApiError("Telegram sendPhoto returned an invalid payload")
        return result

    async def send_video(
        self,
        chat_id: int,
        path: Path,
        *,
        caption: str | None = None,
    ) -> Mapping[str, Any]:
        data: dict[str, Any] = {"chat_id": chat_id, "supports_streaming": "true"}
        if caption:
            data["caption"] = caption[:1024]
        with path.open("rb") as handle:
            result = await self._call(
                "sendVideo",
                data=data,
                files={"video": (path.name, handle, "video/mp4")},
            )
        if not isinstance(result, Mapping):
            raise TelegramApiError("Telegram sendVideo returned an invalid payload")
        return result

    async def answer_callback_query(self, callback_query_id: str, text: str = "") -> None:
        data: dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            data["text"] = text[:200]
        await self._call("answerCallbackQuery", data=data)
