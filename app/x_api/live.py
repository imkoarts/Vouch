"""Official X API v2 user-context client for explicit manual publication only."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import mimetypes
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit

import httpx

from app.x_api.mock import AmbiguousPublicationError

_SUPPORTED_IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
_MAX_SIMPLE_IMAGE_BYTES = 5 * 1024 * 1024
_MAX_VIDEO_BYTES = 512 * 1024 * 1024
_VIDEO_CHUNK_BYTES = 4 * 1024 * 1024
_MAX_VIDEO_STATUS_POLLS = 30
MultipartValue = tuple[str | None, bytes | str] | tuple[str | None, bytes | str, str | None]


def _read_approved_image(path: Path) -> bytes:
    if not path.is_file():
        raise XApiError("The approved image file is missing")
    file_size = path.stat().st_size
    if file_size <= 0 or file_size > _MAX_SIMPLE_IMAGE_BYTES:
        raise XApiError("The image must be non-empty and no larger than 5 MB")
    return path.read_bytes()


def _approved_video_size(path: Path) -> int:
    if not path.is_file():
        raise XApiError("The approved video file is missing")
    return path.stat().st_size


class XApiError(RuntimeError):
    """A safe X API failure that exposes only allowlisted provider diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        title: str | None = None,
        detail: str | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        access_level: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.title = title
        self.detail = detail
        self.error_type = error_type
        self.error_code = error_code
        self.request_id = request_id
        self.access_level = access_level


class XPermissionError(XApiError):
    """X authenticated the request but denied the requested write capability."""


class XAuthenticationError(XApiError):
    """X rejected the configured user credential."""


class XRateLimitError(XApiError):
    def __init__(self, *, reset_at: str | None = None) -> None:
        self.reset_at = reset_at
        message = "X API rate limit reached"
        if reset_at:
            message += f"; reset header: {reset_at}"
        super().__init__(message, status_code=429)


def _safe_provider_text(value: object, *, limit: int = 280) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).replace("\x00", " ").split())
    if not normalized:
        return None
    return normalized[:limit]


def _problem_fields(response: httpx.Response) -> dict[str, str | None]:
    """Extract only RFC7807/legacy diagnostic fields; never retain the raw body."""

    title: str | None = None
    detail: str | None = None
    error_type: str | None = None
    error_code: str | None = None
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        payload = None
    if isinstance(payload, Mapping):
        title = _safe_provider_text(payload.get("title"))
        detail = _safe_provider_text(payload.get("detail") or payload.get("message"))
        error_type = _safe_provider_text(payload.get("type"))
        error_code = _safe_provider_text(payload.get("code"))
        errors = payload.get("errors")
        if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
            first = next((item for item in errors if isinstance(item, Mapping)), None)
            if isinstance(first, Mapping):
                title = title or _safe_provider_text(first.get("title"))
                detail = detail or _safe_provider_text(first.get("detail") or first.get("message"))
                error_type = error_type or _safe_provider_text(first.get("type"))
                error_code = error_code or _safe_provider_text(first.get("code"))
    request_id = _safe_provider_text(
        response.headers.get("x-request-id")
        or response.headers.get("x-transaction-id")
        or response.headers.get("trace-id")
    )
    access_level = _safe_provider_text(response.headers.get("x-access-level"))
    return {
        "title": title,
        "detail": detail,
        "error_type": error_type,
        "error_code": error_code,
        "request_id": request_id,
        "access_level": access_level,
    }


def _percent_encode(value: object) -> str:
    return quote(str(value), safe="~-._")


def _normalized_base_url(url: str) -> str:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if not scheme or not host:
        raise ValueError("X API URL must be absolute")
    port = parsed.port
    netloc = host
    if port is not None and not (
        (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    ):
        netloc = f"{host}:{port}"
    return urlunsplit((scheme, netloc, parsed.path or "/", "", ""))


def oauth1_authorization_header(
    *,
    method: str,
    url: str,
    consumer_key: str,
    consumer_secret: str,
    access_token: str,
    access_token_secret: str,
    query: Mapping[str, object] | Sequence[tuple[str, object]] | None = None,
    nonce: str | None = None,
    timestamp: int | None = None,
) -> str:
    """Create an RFC 5849 HMAC-SHA1 Authorization header for X user context."""

    oauth_parameters: dict[str, str] = {
        "oauth_consumer_key": consumer_key,
        "oauth_nonce": nonce or secrets.token_hex(16),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(timestamp if timestamp is not None else int(time.time())),
        "oauth_token": access_token,
        "oauth_version": "1.0",
    }
    parameters: list[tuple[str, str]] = [
        (key, value) for key, value in parse_qsl(urlsplit(url).query, keep_blank_values=True)
    ]
    if query is not None:
        items = query.items() if isinstance(query, Mapping) else query
        for key, value in items:
            if isinstance(value, (tuple, list)):
                parameters.extend((str(key), str(item)) for item in value)
            elif value is not None:
                parameters.append((str(key), str(value)))
    parameters.extend(oauth_parameters.items())
    encoded_parameters = sorted(
        (_percent_encode(key), _percent_encode(value)) for key, value in parameters
    )
    parameter_string = "&".join(f"{key}={value}" for key, value in encoded_parameters)
    signature_base = "&".join(
        (
            method.upper(),
            _percent_encode(_normalized_base_url(url)),
            _percent_encode(parameter_string),
        )
    )
    signing_key = f"{_percent_encode(consumer_secret)}&{_percent_encode(access_token_secret)}"
    signature = base64.b64encode(
        hmac.new(signing_key.encode(), signature_base.encode(), hashlib.sha1).digest()
    ).decode()
    oauth_parameters["oauth_signature"] = signature
    header_values = ", ".join(
        f'{_percent_encode(key)}="{_percent_encode(value)}"'
        for key, value in sorted(oauth_parameters.items())
    )
    return f"OAuth {header_values}"


class XApiClient:
    """Minimal live writer used only after Telegram approval and explicit confirmation."""

    def __init__(
        self,
        *,
        base_url: str,
        auth_mode: Literal["oauth1", "oauth2"],
        access_token: str,
        consumer_key: str = "",
        consumer_secret: str = "",
        access_token_secret: str = "",
        oauth2_scopes: frozenset[str] | None = None,
        timeout_seconds: float = 30.0,
        proxy_url: str | None = None,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        if not access_token:
            raise ValueError("X user access token is required")
        if auth_mode == "oauth1" and not all((consumer_key, consumer_secret, access_token_secret)):
            raise ValueError("OAuth 1.0a requires consumer and access-token secrets")
        self.base_url = base_url.rstrip("/")
        self.auth_mode = auth_mode
        self.access_token = access_token
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token_secret = access_token_secret
        self.oauth2_scopes = oauth2_scopes
        self.timeout_seconds = timeout_seconds
        self.last_rate_limit: dict[str, str] = {}
        self.last_access_level: str | None = None
        self.last_request_id: str | None = None
        self._source_media_ids: set[str] = set()
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
        params: Mapping[str, object] | None = None,
    ) -> str:
        if self.auth_mode == "oauth2":
            return f"Bearer {self.access_token}"
        return oauth1_authorization_header(
            method=method,
            url=url,
            consumer_key=self.consumer_key,
            consumer_secret=self.consumer_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret,
            query=params,
        )

    def _remember_response_metadata(self, response: httpx.Response) -> None:
        self.last_rate_limit = {
            name.removeprefix("x-rate-limit-"): value
            for name in ("x-rate-limit-limit", "x-rate-limit-remaining", "x-rate-limit-reset")
            if (value := response.headers.get(name)) is not None
        }
        fields = _problem_fields(response)
        if fields["access_level"]:
            self.last_access_level = fields["access_level"]
        if fields["request_id"]:
            self.last_request_id = fields["request_id"]

    def _known_read_only(self) -> bool:
        if self.last_access_level is None:
            return False
        normalized = self.last_access_level.casefold().replace("_", "-")
        return "read" in normalized and "write" not in normalized

    def _write_permission_message(self, *, status_code: int, detail: str | None) -> str:
        if self.auth_mode == "oauth1":
            action = (
                "The OAuth 1.0a user token may be read-only. In X Developer Console, confirm "
                "that the app permissions are Read and write. If you change them, regenerate "
                "the Access Token and Secret and update the local .env; older tokens keep their "
                "original permissions."
            )
        else:
            action = (
                "Re-authorize the OAuth 2.0 user token with tweet.write and users.read; "
                "media publication also requires media.write."
            )
        provider_detail = f" X detail: {detail}" if detail else ""
        return (
            f"X denied manual publication with HTTP {status_code}.{provider_detail} {action}"
        ).strip()

    def _raise_http_error(
        self,
        response: httpx.Response,
        *,
        method: str,
        path: str,
        operation: str,
    ) -> None:
        self._remember_response_metadata(response)
        fields = _problem_fields(response)
        status = response.status_code
        if status == 429:
            raise XRateLimitError(reset_at=response.headers.get("x-rate-limit-reset"))
        title = fields["title"]
        detail_text = fields["detail"]
        error_type = fields["error_type"]
        error_code = fields["error_code"]
        request_id = fields["request_id"]
        access_level = fields["access_level"] or self.last_access_level
        if status == 401:
            raise XAuthenticationError(
                "X rejected the configured user credential with HTTP 401. Recreate the user "
                "access token and update the local .env.",
                status_code=status,
                title=title,
                detail=detail_text,
                error_type=error_type,
                error_code=error_code,
                request_id=request_id,
                access_level=access_level,
            )
        is_write = method.upper() in {"POST", "PUT", "PATCH", "DELETE"} and (
            path == "/2/tweets" or path.startswith("/2/media/upload")
        )
        if status == 403 and is_write:
            raise XPermissionError(
                self._write_permission_message(status_code=status, detail=detail_text),
                status_code=status,
                title=title,
                detail=detail_text,
                error_type=error_type,
                error_code=error_code,
                request_id=request_id,
                access_level=access_level,
            )
        detail_suffix = f"; X detail: {detail_text}" if detail_text else ""
        raise XApiError(
            f"{operation} was rejected with HTTP {status}{detail_suffix}",
            status_code=status,
            title=title,
            detail=detail_text,
            error_type=error_type,
            error_code=error_code,
            request_id=request_id,
            access_level=access_level,
        )

    def _assert_known_write_access(self) -> None:
        if not self._known_read_only():
            return
        raise XPermissionError(
            self._write_permission_message(
                status_code=403,
                detail="token access level is read-only",
            ),
            status_code=403,
            detail="token access level is read-only",
            request_id=self.last_request_id,
            access_level=self.last_access_level,
        )

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
        payload: Mapping[str, Any] | None = None,
        ambiguous_on_transport_error: bool = False,
        operation: str = "X API request",
    ) -> Mapping[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": self._authorization(method, url, params),
            "Accept": "application/json",
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"
        try:
            async with self._client_factory() as client:
                response = await client.request(
                    method,
                    url,
                    params=(
                        {str(key): str(value) for key, value in params.items()}
                        if params is not None
                        else None
                    ),
                    json=dict(payload) if payload is not None else None,
                    headers=headers,
                )
        except httpx.HTTPError as exc:
            if ambiguous_on_transport_error:
                raise AmbiguousPublicationError(
                    "X request outcome is unknown; automatic retry is prohibited"
                ) from exc
            raise XApiError("X API transport request failed") from exc

        self._remember_response_metadata(response)
        if response.status_code >= 500 and ambiguous_on_transport_error:
            raise AmbiguousPublicationError(
                "X returned a server error after a write request; outcome is unknown"
            )
        if response.status_code >= 400:
            self._raise_http_error(
                response,
                method=method,
                path=path,
                operation=operation,
            )
        try:
            decoded = response.json()
        except json.JSONDecodeError as exc:
            if ambiguous_on_transport_error:
                raise AmbiguousPublicationError(
                    "X write response was not valid JSON; outcome is unknown"
                ) from exc
            raise XApiError("X API returned an invalid JSON response") from exc
        if not isinstance(decoded, Mapping):
            raise XApiError("X API returned an invalid response object")
        return decoded

    async def _request_multipart(
        self,
        method: str,
        path: str,
        *,
        data: Mapping[str, object],
        files: Mapping[str, MultipartValue] | None = None,
        operation: str = "X media upload",
    ) -> Mapping[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": self._authorization(method, url),
            "Accept": "application/json",
        }
        multipart: dict[str, MultipartValue] = {
            str(key): (None, str(value)) for key, value in data.items()
        }
        if files is not None:
            multipart.update(files)
        try:
            async with self._client_factory() as client:
                response = await client.request(
                    method,
                    url,
                    files=multipart,
                    headers=headers,
                )
        except httpx.HTTPError as exc:
            raise XApiError("X media upload transport request failed") from exc
        self._remember_response_metadata(response)
        if response.status_code >= 400:
            self._raise_http_error(
                response,
                method=method,
                path=path,
                operation=operation,
            )
        if response.status_code == 204 or not response.content:
            return {}
        try:
            decoded = response.json()
        except json.JSONDecodeError as exc:
            raise XApiError("X media upload returned invalid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise XApiError("X media upload returned an invalid response object")
        return decoded

    @staticmethod
    def _data(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise XApiError("X API response is missing data")
        return data

    async def get_me(self) -> Mapping[str, Any]:
        payload = await self._request_json(
            "GET",
            "/2/users/me",
            params={"user.fields": "username"},
        )
        data = self._data(payload)
        identifier = str(data.get("id", ""))
        if not identifier:
            raise XApiError("X API did not identify the authenticated user")
        return {"id": identifier, "username": str(data.get("username", ""))}

    async def upload_media(self, path: Path) -> str:
        """Upload an approved image or MP4 video through the official X media API."""

        self._assert_known_write_access()
        if (
            self.auth_mode == "oauth2"
            and self.oauth2_scopes is not None
            and "media.write" not in self.oauth2_scopes
        ):
            raise XApiError("OAuth2 user token is missing the media.write scope")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if media_type in _SUPPORTED_IMAGE_MEDIA_TYPES:
            media_bytes = await asyncio.to_thread(_read_approved_image, path)
            encoded = base64.b64encode(media_bytes).decode("ascii")
            payload = await self._request_json(
                "POST",
                "/2/media/upload",
                payload={
                    "media": encoded,
                    "media_category": "tweet_image",
                    "media_type": media_type,
                    "shared": False,
                },
            )
            data = self._data(payload)
            media_id = str(data.get("id") or data.get("media_id") or "")
        elif media_type == "video/mp4":
            total_bytes = await asyncio.to_thread(_approved_video_size, path)
            if total_bytes <= 0 or total_bytes > _MAX_VIDEO_BYTES:
                raise XApiError("The video must be non-empty and no larger than 512 MB")
            initialized = await self._request_json(
                "POST",
                "/2/media/upload/initialize",
                payload={
                    "media_type": "video/mp4",
                    "total_bytes": total_bytes,
                    "media_category": "tweet_video",
                    "shared": False,
                },
                operation="X media upload initialization",
            )
            media_id = str(self._data(initialized).get("id") or "")
            if not media_id:
                raise XApiError("X media initialization response is missing a media ID")
            with path.open("rb") as handle:
                segment_index = 0
                while chunk := handle.read(_VIDEO_CHUNK_BYTES):
                    await self._request_multipart(
                        "POST",
                        f"/2/media/upload/{media_id}/append",
                        data={"segment_index": segment_index},
                        files={"media": (path.name, chunk, "video/mp4")},
                        operation=f"X media chunk upload (segment {segment_index})",
                    )
                    segment_index += 1
            finalized = await self._request_json(
                "POST",
                f"/2/media/upload/{media_id}/finalize",
                operation="X media upload finalization",
            )
            processing = self._data(finalized).get("processing_info")
            polls = 0
            while isinstance(processing, Mapping):
                state = str(processing.get("state", ""))
                if state == "succeeded":
                    break
                if state == "failed":
                    raise XApiError("X video processing failed")
                if state not in {"pending", "in_progress"}:
                    raise XApiError("X video processing returned an unknown state")
                polls += 1
                if polls > _MAX_VIDEO_STATUS_POLLS:
                    raise XApiError(
                        "X video processing did not complete within the bounded poll limit"
                    )
                delay = max(0.0, min(float(processing.get("check_after_secs") or 1), 10.0))
                if delay:
                    await asyncio.sleep(delay)
                status_payload = await self._request_json(
                    "GET",
                    "/2/media/upload",
                    params={"command": "STATUS", "media_id": media_id},
                    operation="X media processing status check",
                )
                processing = self._data(status_payload).get("processing_info")
        else:
            raise XApiError(
                "Only PNG, JPEG, and WEBP images or MP4 videos are supported by the manual writer"
            )

        if not media_id:
            raise XApiError("X media upload response is missing a media ID")
        if path.name.casefold().startswith("source-"):
            self._source_media_ids.add(media_id)
        return media_id

    async def create_post(
        self,
        *,
        text: str,
        reply_to_post_id: str | None,
        media_ids: tuple[str, ...] = (),
        quote_post_id: str | None = None,
        position: int = 0,
    ) -> Mapping[str, Any]:
        del position
        self._assert_known_write_access()
        if (
            self.auth_mode == "oauth2"
            and self.oauth2_scopes is not None
            and "tweet.write" not in self.oauth2_scopes
        ):
            raise XPermissionError(
                "OAuth2 user token is missing the tweet.write scope. Re-authorize the user token.",
                status_code=403,
                detail="missing tweet.write scope",
            )
        body: dict[str, Any] = {"text": text}
        if reply_to_post_id and quote_post_id:
            raise XApiError("A post cannot be both a reply and a quote post")
        if quote_post_id and media_ids:
            raise XApiError("X quote posts cannot include uploaded media in the same request")
        if reply_to_post_id:
            body["reply"] = {"in_reply_to_tweet_id": reply_to_post_id}
        if quote_post_id:
            body["quote_tweet_id"] = quote_post_id
        if media_ids:
            body["media"] = {"media_ids": list(media_ids)}
            if not all(media_id in self._source_media_ids for media_id in media_ids):
                body["made_with_ai"] = True
        payload = await self._request_json(
            "POST",
            "/2/tweets",
            payload=body,
            ambiguous_on_transport_error=True,
        )
        data = self._data(payload)
        post_id = str(data.get("id", ""))
        if not post_id:
            raise AmbiguousPublicationError(
                "X write response is missing a post ID; automatic retry is prohibited"
            )
        result: dict[str, Any] = {"id": post_id, "mock": False}
        if self.last_rate_limit:
            result["rate_limit"] = dict(self.last_rate_limit)
        return result

    async def delete_post(self, post_id: str) -> Mapping[str, Any]:
        payload = await self._request_json(
            "DELETE",
            f"/2/tweets/{post_id}",
            ambiguous_on_transport_error=True,
        )
        data = self._data(payload)
        result: dict[str, Any] = {
            "id": post_id,
            "deleted": bool(data.get("deleted")),
            "mock": False,
        }
        if self.last_rate_limit:
            result["rate_limit"] = dict(self.last_rate_limit)
        return result
