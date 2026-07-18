"""Budget-bounded official X API reads for trend discovery.

The Creator Inspiration web page is intentionally not scraped. The runtime uses
official trends, the authenticated home timeline, explicitly selected account
timelines, and direct post lookups. X Recent Search is product-disabled.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, Literal
from urllib.parse import quote

import httpx

from app.x_api.live import oauth1_authorization_header

_LOGGER = logging.getLogger(__name__)
_TEMPORARY_HTTP_STATUSES = frozenset({408, 425, 500, 502, 503, 504})


class XReadError(RuntimeError):
    """Safe read failure that never exposes response bodies or credentials."""


class XReadRateLimitError(XReadError):
    pass


class XReadTemporaryError(XReadError):
    """A bounded retryable transport or upstream X API failure."""

    def __init__(self, message: str, *, attempts: int, status_code: int | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code


class XReadPaymentRequiredError(XReadError):
    """The X project has no usable prepaid API credits for this endpoint."""


class XReadAuthenticationError(XReadError):
    """The X credential is missing required authorization or scope."""


class XReadClient:
    """Small official X API v2 reader with explicit per-call result limits."""

    _POST_FIELDS = (
        "author_id,created_at,conversation_id,lang,public_metrics,attachments,referenced_tweets"
    )
    _EXPANSIONS = (
        "author_id,attachments.media_keys,referenced_tweets.id,"
        "referenced_tweets.id.author_id,referenced_tweets.id.attachments.media_keys"
    )
    _MEDIA_FIELDS = (
        "media_key,type,url,preview_image_url,variants,width,height,duration_ms,alt_text"
    )
    _USER_FIELDS = "username,name,profile_image_url,verified"

    def __init__(
        self,
        *,
        base_url: str,
        auth_mode: Literal["oauth1", "oauth2"],
        bearer_token: str = "",
        access_token: str = "",
        consumer_key: str = "",
        consumer_secret: str = "",
        access_token_secret: str = "",
        timeout_seconds: float = 30.0,
        proxy_url: str | None = None,
        temporary_error_max_retries: int = 2,
        temporary_error_initial_backoff_seconds: float = 2.0,
        temporary_error_max_backoff_seconds: float = 10.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_mode = auth_mode
        self.bearer_token = bearer_token
        self.access_token = access_token
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token_secret = access_token_secret
        self.timeout_seconds = timeout_seconds
        self.temporary_error_max_retries = max(0, min(temporary_error_max_retries, 5))
        self.temporary_error_initial_backoff_seconds = max(
            0.0, temporary_error_initial_backoff_seconds
        )
        self.temporary_error_max_backoff_seconds = max(
            self.temporary_error_initial_backoff_seconds,
            temporary_error_max_backoff_seconds,
        )
        self._sleep = sleep
        self.last_attempt_count = 0
        self._client_factory = client_factory or (
            lambda: httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 15.0)),
                follow_redirects=False,
                proxy=proxy_url,
                trust_env=False,
            )
        )

    def _authorization(
        self,
        method: str,
        url: str,
        params: Mapping[str, object] | None,
        *,
        user_context: bool,
    ) -> str:
        if not user_context and self.bearer_token:
            return f"Bearer {self.bearer_token}"
        if self.auth_mode == "oauth2":
            if not self.access_token:
                raise XReadError("OAuth2 user access token is missing")
            return f"Bearer {self.access_token}"
        if not all(
            (
                self.consumer_key,
                self.consumer_secret,
                self.access_token,
                self.access_token_secret,
            )
        ):
            raise XReadError("OAuth1 user-context credentials are incomplete")
        return oauth1_authorization_header(
            method=method,
            url=url,
            consumer_key=self.consumer_key,
            consumer_secret=self.consumer_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret,
            query=params,
        )

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        raw = response.headers.get("retry-after")
        if raw is None:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        return max(0.0, value)

    def _temporary_backoff(self, retry_index: int, response: httpx.Response | None) -> float:
        exponential = min(
            self.temporary_error_initial_backoff_seconds * (2**retry_index),
            self.temporary_error_max_backoff_seconds,
        )
        if response is None:
            return exponential
        server_delay = self._retry_after_seconds(response)
        if server_delay is None:
            return exponential
        return min(max(exponential, server_delay), self.temporary_error_max_backoff_seconds)

    async def _get(
        self,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
        user_context: bool = False,
    ) -> Mapping[str, Any]:
        url = f"{self.base_url}{path}"
        request_params = (
            {str(key): str(value) for key, value in params.items()} if params is not None else None
        )
        self.last_attempt_count = 0
        for retry_index in range(self.temporary_error_max_retries + 1):
            self.last_attempt_count = retry_index + 1
            headers = {
                "Authorization": self._authorization("GET", url, params, user_context=user_context),
                "Accept": "application/json",
            }
            response: httpx.Response | None = None
            try:
                async with self._client_factory() as client:
                    response = await client.get(
                        url,
                        params=request_params,
                        headers=headers,
                    )
            except httpx.HTTPError as exc:
                if retry_index >= self.temporary_error_max_retries:
                    attempts = retry_index + 1
                    raise XReadTemporaryError(
                        f"X API read transport failed after {attempts} attempts",
                        attempts=attempts,
                    ) from exc
                delay = self._temporary_backoff(retry_index, None)
                _LOGGER.warning(
                    "Temporary X read transport failure; retrying attempt %s/%s in %.1fs",
                    retry_index + 2,
                    self.temporary_error_max_retries + 1,
                    delay,
                )
                if delay > 0:
                    await self._sleep(delay)
                continue

            if response.status_code in _TEMPORARY_HTTP_STATUSES:
                if retry_index >= self.temporary_error_max_retries:
                    attempts = retry_index + 1
                    raise XReadTemporaryError(
                        (
                            "X API is temporarily unavailable "
                            f"(HTTP {response.status_code}) after {attempts} attempts"
                        ),
                        attempts=attempts,
                        status_code=response.status_code,
                    )
                delay = self._temporary_backoff(retry_index, response)
                _LOGGER.warning(
                    ("X API temporary HTTP %s; retrying read attempt %s/%s in %.1fs"),
                    response.status_code,
                    retry_index + 2,
                    self.temporary_error_max_retries + 1,
                    delay,
                )
                if delay > 0:
                    await self._sleep(delay)
                continue

            if response.status_code == 402:
                raise XReadPaymentRequiredError(
                    "X API credits are unavailable for this read request"
                )
            if response.status_code in {401, 403}:
                raise XReadAuthenticationError(
                    f"X API read authorization failed with HTTP {response.status_code}"
                )
            if response.status_code == 429:
                raise XReadRateLimitError("X API read rate limit reached")
            if response.status_code >= 400:
                raise XReadError(f"X API read rejected with HTTP {response.status_code}")
            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                raise XReadError("X API read returned invalid JSON") from exc
            if not isinstance(payload, Mapping):
                raise XReadError("X API read returned an invalid object")
            return payload
        raise AssertionError("unreachable X read retry loop")

    @staticmethod
    def _normalize_posts(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        rows = payload.get("data", [])
        if rows is None:
            return ()
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise XReadError("X API Post response has invalid data")

        includes = payload.get("includes", {})
        includes = includes if isinstance(includes, Mapping) else {}
        included_tweets = {
            str(item.get("id")): item
            for item in includes.get("tweets", ())
            if isinstance(item, Mapping) and item.get("id")
        }
        users = {
            str(item.get("id")): dict(item)
            for item in includes.get("users", ())
            if isinstance(item, Mapping) and item.get("id")
        }
        media = {
            str(item.get("media_key")): dict(item)
            for item in includes.get("media", ())
            if isinstance(item, Mapping) and item.get("media_key")
        }

        def original_for(row: Mapping[str, Any]) -> Mapping[str, Any]:
            references = row.get("referenced_tweets", ())
            if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
                return row
            for reference in references:
                if not isinstance(reference, Mapping):
                    continue
                if str(reference.get("type", "")) != "retweeted":
                    continue
                included = included_tweets.get(str(reference.get("id", "")))
                if included is not None:
                    return included
            return row

        normalized: list[Mapping[str, Any]] = []
        seen_ids: set[str] = set()
        for discovered_row in rows:
            if not isinstance(discovered_row, Mapping):
                continue
            row = original_for(discovered_row)
            post_id = str(row.get("id", "")).strip()
            text = str(row.get("text", "")).strip()
            if not post_id or not text or post_id in seen_ids:
                continue
            seen_ids.add(post_id)

            attachments = row.get("attachments")
            attachment_data = dict(attachments) if isinstance(attachments, Mapping) else {}
            media_keys = attachment_data.get("media_keys", ())
            if isinstance(media_keys, (str, bytes)) or not isinstance(media_keys, Sequence):
                media_keys = ()
            media_items = [media[key] for key in map(str, media_keys) if key in media]
            media_metadata: dict[str, Any] = {
                "items": media_items,
                "media_keys": [str(key) for key in media_keys],
            }
            if row is not discovered_row:
                media_metadata["discovered_via_post_id"] = str(discovered_row.get("id", ""))
                media_metadata["reference_type"] = "retweeted"

            author_id = str(row.get("author_id", "")).strip()
            author = users.get(author_id, {})
            username = str(author.get("username", "")).strip() or None
            url = (
                f"https://x.com/{username}/status/{post_id}"
                if username
                else f"https://x.com/i/web/status/{post_id}"
            )
            normalized.append(
                {
                    "id": post_id,
                    "text": text,
                    "author_id": author_id or None,
                    "username": username,
                    "created_at": row.get("created_at"),
                    "conversation_id": row.get("conversation_id"),
                    "lang": row.get("lang"),
                    "public_metrics": (
                        dict(row.get("public_metrics", {}))
                        if isinstance(row.get("public_metrics"), Mapping)
                        else {}
                    ),
                    "media_metadata": media_metadata,
                    "url": url,
                    "raw_json": {
                        "post": dict(row),
                        "discovered_post": dict(discovered_row),
                        "author": author,
                        "media": media_items,
                    },
                }
            )
        return tuple(normalized)

    def _post_params(self) -> dict[str, object]:
        return {
            "tweet.fields": self._POST_FIELDS,
            "expansions": self._EXPANSIONS,
            "media.fields": self._MEDIA_FIELDS,
            "user.fields": self._USER_FIELDS,
        }

    async def get_post(self, post_id: str) -> Mapping[str, Any]:
        payload = await self._get(
            f"/2/tweets/{post_id}",
            params=self._post_params(),
        )
        normalized = self._normalize_posts(
            {
                "data": [payload.get("data", {})],
                "includes": payload.get("includes", {}),
            }
        )
        if not normalized:
            raise XReadError("X API Post lookup returned no usable post")
        return normalized[0]

    async def get_home_timeline(
        self,
        user_id: str,
        *,
        max_results: int,
        start_time: datetime,
        since_id: str | None = None,
        exclude_replies: bool = True,
        exclude_retweets: bool = True,
    ) -> tuple[Mapping[str, Any], ...]:
        if not 1 <= max_results <= 10:
            raise ValueError("Home timeline max_results must be between 1 and 10")
        exclude: list[str] = []
        if exclude_replies:
            exclude.append("replies")
        if exclude_retweets:
            exclude.append("retweets")
        params: dict[str, object] = {
            **self._post_params(),
            "max_results": max_results,
        }
        if since_id:
            params["since_id"] = since_id
        else:
            params["start_time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        if exclude:
            params["exclude"] = ",".join(exclude)
        payload = await self._get(
            f"/2/users/{user_id}/timelines/reverse_chronological",
            params=params,
            user_context=True,
        )
        return self._normalize_posts(payload)[:max_results]

    async def get_user_by_username(self, username: str) -> Mapping[str, Any]:
        """Resolve one configured username without using Recent Search."""

        normalized = username.strip().removeprefix("@").casefold()
        if not normalized:
            raise ValueError("X username must not be empty")
        payload = await self._get(
            f"/2/users/by/username/{quote(normalized, safe='')}",
            params={"user.fields": self._USER_FIELDS},
        )
        data = payload.get("data")
        if not isinstance(data, Mapping) or not data.get("id"):
            raise XReadError(f"X account @{normalized} was not found")
        return dict(data)

    async def get_user_posts(
        self,
        user_id: str,
        *,
        max_results: int,
        start_time: datetime,
        since_id: str | None = None,
        exclude_replies: bool = True,
        exclude_retweets: bool = True,
    ) -> tuple[Mapping[str, Any], ...]:
        """Read recent posts from one selected account timeline."""

        if not 5 <= max_results <= 100:
            raise ValueError("Selected-account timeline max_results must be between 5 and 100")
        exclude: list[str] = []
        if exclude_replies:
            exclude.append("replies")
        if exclude_retweets:
            exclude.append("retweets")
        params: dict[str, object] = {
            **self._post_params(),
            "max_results": max_results,
        }
        if since_id:
            params["since_id"] = since_id
        else:
            params["start_time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        if exclude:
            params["exclude"] = ",".join(exclude)
        payload = await self._get(f"/2/users/{user_id}/tweets", params=params)
        return self._normalize_posts(payload)[:max_results]

    async def get_trends_by_woeid(
        self,
        woeid: int,
        *,
        max_trends: int,
    ) -> tuple[str, ...]:
        if not 1 <= max_trends <= 10:
            raise ValueError("Trend limit must be between 1 and 10")
        payload = await self._get(
            f"/2/trends/by/woeid/{woeid}",
            params={"max_trends": max_trends, "trend.fields": "trend_name,tweet_count"},
        )
        rows = payload.get("data", [])
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise XReadError("X trend response has invalid data")
        trends: list[str] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("trend_name", "")).strip()
            if name and name not in trends:
                trends.append(name)
        return tuple(trends[:max_trends])

    async def recent_search(
        self,
        query: str,
        *,
        max_results: int,
        start_time: datetime,
        sort_order: Literal["recency", "relevancy"] = "relevancy",
    ) -> tuple[Mapping[str, Any], ...]:
        """Fail closed: the application never uses X Recent Search."""

        del query, max_results, start_time, sort_order
        raise XReadError(
            "X Recent Search is disabled. Use home timeline, trends, Selected accounts, "
            "or a direct X post URL."
        )
