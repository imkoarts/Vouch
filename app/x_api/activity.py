"""Official X Activity API client with explicit subscription writes and safe streaming."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx


class XActivityApiError(RuntimeError):
    """Base error for X Activity API failures without leaking credentials."""


class XActivityRateLimitError(XActivityApiError):
    def __init__(self, *, reset_at: str | None = None) -> None:
        self.reset_at = reset_at
        message = "X Activity API rate limit exceeded"
        if reset_at:
            message += f"; reset={reset_at}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class XActivitySubscription:
    event_type: str
    user_id: str
    tag: str | None = None
    subscription_id: str | None = None

    @classmethod
    def from_api(cls, value: Mapping[str, Any]) -> XActivitySubscription:
        filter_value = value.get("filter")
        user_id = ""
        if isinstance(filter_value, Mapping):
            user_id = str(filter_value.get("user_id") or "")
        event_type = str(value.get("event_type") or "")
        if not event_type or not user_id:
            raise XActivityApiError("X Activity subscription response is incomplete")
        tag_value = value.get("tag")
        subscription_id = value.get("subscription_id")
        return cls(
            event_type=event_type,
            user_id=user_id,
            tag=str(tag_value) if tag_value is not None else None,
            subscription_id=(str(subscription_id) if subscription_id is not None else None),
        )

    def request_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event_type": self.event_type,
            "filter": {"user_id": self.user_id},
        }
        if self.tag:
            payload["tag"] = self.tag
        return payload


class XActivityClient:
    """Minimal client for subscription CRUD and persistent HTTP activity streams."""

    def __init__(
        self,
        *,
        token: str,
        base_url: str = "https://api.x.com",
        timeout_seconds: float = 30.0,
        proxy_url: str | None = None,
        client_factory: Callable[..., httpx.AsyncClient] | None = None,
    ) -> None:
        if not token:
            raise ValueError("X Activity bearer token is required")
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.timeout_seconds = timeout_seconds
        self.proxy_url = proxy_url
        self._client_factory = client_factory

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }

    def _client(self, *, streaming: bool = False) -> httpx.AsyncClient:
        if self._client_factory is not None:
            return self._client_factory(streaming=streaming)
        timeout = (
            httpx.Timeout(connect=min(self.timeout_seconds, 15.0), read=None, write=30.0, pool=30.0)
            if streaming
            else httpx.Timeout(
                self.timeout_seconds,
                connect=min(self.timeout_seconds, 15.0),
            )
        )
        return httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            proxy=self.proxy_url,
            trust_env=False,
        )

    @staticmethod
    def _decode_json(response: httpx.Response) -> Mapping[str, Any]:
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise XActivityApiError("X Activity API returned invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise XActivityApiError("X Activity API returned an invalid response object")
        return payload

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code == 429:
            raise XActivityRateLimitError(reset_at=response.headers.get("x-rate-limit-reset"))
        if response.status_code >= 400:
            raise XActivityApiError(
                f"X Activity API rejected the request with HTTP {response.status_code}"
            )

    async def list_subscriptions(self) -> tuple[XActivitySubscription, ...]:
        subscriptions: list[XActivitySubscription] = []
        pagination_token: str | None = None
        try:
            async with self._client() as client:
                while True:
                    params = {"max_results": "1000"}
                    if pagination_token:
                        params["pagination_token"] = pagination_token
                    response = await client.get(
                        f"{self.base_url}/2/activity/subscriptions",
                        headers=self._headers(),
                        params=params,
                    )
                    self._raise_for_status(response)
                    payload = self._decode_json(response)
                    data = payload.get("data", [])
                    if isinstance(data, Mapping):
                        data = [data]
                    if not isinstance(data, list):
                        raise XActivityApiError(
                            "X Activity subscriptions response has invalid data"
                        )
                    subscriptions.extend(
                        XActivitySubscription.from_api(item)
                        for item in data
                        if isinstance(item, Mapping)
                    )
                    meta = payload.get("meta")
                    next_token = meta.get("next_token") if isinstance(meta, Mapping) else None
                    if next_token is None or not str(next_token):
                        break
                    pagination_token = str(next_token)
        except httpx.HTTPError as exc:
            raise XActivityApiError("X Activity subscription request failed") from exc
        return tuple(subscriptions)

    async def create_subscription(
        self, subscription: XActivitySubscription
    ) -> XActivitySubscription:
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        try:
            async with self._client() as client:
                response = await client.post(
                    f"{self.base_url}/2/activity/subscriptions",
                    headers=headers,
                    json=subscription.request_payload(),
                )
        except httpx.HTTPError as exc:
            raise XActivityApiError("X Activity subscription creation failed") from exc
        self._raise_for_status(response)
        payload = self._decode_json(response)
        data = payload.get("data")
        if isinstance(data, Mapping) and isinstance(data.get("subscription"), Mapping):
            data = data["subscription"]
        elif isinstance(data, list):
            data = data[0] if data else None
        if not isinstance(data, Mapping):
            raise XActivityApiError("X Activity create response is missing data")
        return XActivitySubscription.from_api(data)

    async def delete_subscription(self, subscription_id: str) -> bool:
        if not subscription_id:
            raise ValueError("subscription_id is required")
        try:
            async with self._client() as client:
                response = await client.delete(
                    f"{self.base_url}/2/activity/subscriptions/{subscription_id}",
                    headers=self._headers(),
                )
        except httpx.HTTPError as exc:
            raise XActivityApiError("X Activity subscription deletion failed") from exc
        self._raise_for_status(response)
        payload = self._decode_json(response)
        data = payload.get("data")
        return bool(data.get("deleted")) if isinstance(data, Mapping) else False

    async def stream_events(
        self,
        *,
        backfill_minutes: int = 0,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        if not 0 <= backfill_minutes <= 5:
            raise ValueError("backfill_minutes must be between 0 and 5")
        params = {"backfill_minutes": str(backfill_minutes)} if backfill_minutes else None
        try:
            async with (
                self._client(streaming=True) as client,
                client.stream(
                    "GET",
                    f"{self.base_url}/2/activity/stream",
                    headers=self._headers(),
                    params=params,
                ) as response,
            ):
                self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if stop_event is not None and stop_event.is_set():
                        return
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        value = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise XActivityApiError(
                            "X Activity stream returned an invalid JSON line"
                        ) from exc
                    if not isinstance(value, Mapping):
                        continue
                    errors = value.get("errors")
                    if errors and not value.get("data"):
                        raise XActivityApiError("X Activity stream returned an error event")
                    if isinstance(value.get("data"), Mapping):
                        yield value
        except httpx.HTTPError as exc:
            raise XActivityApiError("X Activity stream connection failed") from exc


def subscription_keys(
    subscriptions: Sequence[XActivitySubscription],
) -> set[tuple[str, str]]:
    return {(item.event_type, item.user_id) for item in subscriptions}
