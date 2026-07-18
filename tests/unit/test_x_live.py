from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.x_api.live import (
    XApiClient,
    XApiError,
    XPermissionError,
    XRateLimitError,
    oauth1_authorization_header,
)
from app.x_api.mock import AmbiguousPublicationError


def _client_factory(handler: Any) -> Any:
    transport = httpx.MockTransport(handler)
    return lambda: httpx.AsyncClient(transport=transport)


def test_oauth1_header_is_deterministic_and_does_not_expose_secrets() -> None:
    header = oauth1_authorization_header(
        method="GET",
        url="https://api.x.com/2/users/me",
        query={"user.fields": "username"},
        consumer_key="consumer",
        consumer_secret="consumer-secret",
        access_token="user-token",
        access_token_secret="token-secret",
        nonce="fixed-nonce",
        timestamp=1_700_000_000,
    )

    assert header.startswith("OAuth ")
    assert 'oauth_consumer_key="consumer"' in header
    assert 'oauth_nonce="fixed-nonce"' in header
    assert "oauth_signature=" in header
    assert "consumer-secret" not in header
    assert "token-secret" not in header


@pytest.mark.asyncio
async def test_oauth2_client_gets_user_uploads_image_and_creates_post(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer user-access-token"
        if request.url.path == "/2/users/me":
            return httpx.Response(200, json={"data": {"id": "450", "username": "test_account"}})
        if request.url.path == "/2/media/upload":
            body = json.loads(request.content)
            assert base64.b64decode(body["media"]) == b"synthetic-image"
            assert body["media_category"] == "tweet_image"
            return httpx.Response(200, json={"data": {"id": "media-1"}})
        if request.url.path == "/2/tweets":
            body = json.loads(request.content)
            assert body == {
                "text": "Test post",
                "media": {"media_ids": ["media-1"]},
                "made_with_ai": True,
            }
            return httpx.Response(201, json={"data": {"id": "post-1"}})
        raise AssertionError(f"Unexpected request: {request.url}")

    image = tmp_path / "image.png"
    image.write_bytes(b"synthetic-image")
    client = XApiClient(
        base_url="https://api.x.com",
        auth_mode="oauth2",
        access_token="user-access-token",
        client_factory=_client_factory(handler),
    )

    assert await client.get_me() == {"id": "450", "username": "test_account"}
    media_id = await client.upload_media(image)
    response = await client.create_post(
        text="Test post",
        reply_to_post_id=None,
        media_ids=(media_id,),
    )

    assert response == {"id": "post-1", "mock": False}
    assert [request.url.path for request in requests] == [
        "/2/users/me",
        "/2/media/upload",
        "/2/tweets",
    ]


@pytest.mark.asyncio
async def test_x_rate_limit_is_reported_without_response_body() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"x-rate-limit-reset": "1700000000"},
            text="private-provider-body",
        )

    client = XApiClient(
        base_url="https://api.x.com",
        auth_mode="oauth2",
        access_token="user-access-token",
        client_factory=_client_factory(handler),
    )

    with pytest.raises(XRateLimitError) as raised:
        await client.get_me()

    assert raised.value.reset_at == "1700000000"
    assert "private-provider-body" not in str(raised.value)


@pytest.mark.asyncio
async def test_create_post_transport_failure_is_ambiguous() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    client = XApiClient(
        base_url="https://api.x.com",
        auth_mode="oauth2",
        access_token="user-access-token",
        client_factory=_client_factory(handler),
    )

    with pytest.raises(AmbiguousPublicationError, match="automatic retry"):
        await client.create_post(text="Post", reply_to_post_id=None)


@pytest.mark.asyncio
async def test_oauth2_media_upload_requires_media_scope_before_network(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"synthetic-image")
    network_called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal network_called
        network_called = True
        return httpx.Response(500)

    client = XApiClient(
        base_url="https://api.x.com",
        auth_mode="oauth2",
        access_token="user-access-token",
        oauth2_scopes=frozenset({"tweet.read", "tweet.write", "users.read"}),
        client_factory=_client_factory(handler),
    )

    with pytest.raises(XApiError, match=r"media\.write"):
        await client.upload_media(image)

    assert network_called is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "content", "expected"),
    [
        ("image.svg", b"<svg/>", "PNG, JPEG, and WEBP"),
        ("empty.png", b"", "non-empty"),
    ],
)
async def test_simple_media_upload_rejects_unsupported_or_empty_files_before_network(
    tmp_path: Path,
    filename: str,
    content: bytes,
    expected: str,
) -> None:
    image = tmp_path / filename
    image.write_bytes(content)
    network_called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal network_called
        network_called = True
        return httpx.Response(500)

    client = XApiClient(
        base_url="https://api.x.com",
        auth_mode="oauth2",
        access_token="user-access-token",
        oauth2_scopes=frozenset({"media.write"}),
        client_factory=_client_factory(handler),
    )

    with pytest.raises(XApiError, match=expected):
        await client.upload_media(image)

    assert network_called is False


@pytest.mark.asyncio
async def test_successful_write_returns_rate_limit_headers_as_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/2/tweets"
        return httpx.Response(
            201,
            headers={
                "x-rate-limit-limit": "200",
                "x-rate-limit-remaining": "199",
                "x-rate-limit-reset": "1700000000",
            },
            json={"data": {"id": "post-1"}},
        )

    client = XApiClient(
        base_url="https://api.x.com",
        auth_mode="oauth2",
        access_token="user-access-token",
        client_factory=_client_factory(handler),
    )

    response = await client.create_post(text="Post", reply_to_post_id=None)

    assert response["rate_limit"] == {
        "limit": "200",
        "remaining": "199",
        "reset": "1700000000",
    }


@pytest.mark.asyncio
async def test_video_upload_uses_dedicated_v2_chunk_endpoints(tmp_path: Path) -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        assert request.headers["Authorization"] == "Bearer user-access-token"
        if request.url.path == "/2/media/upload/initialize":
            assert request.method == "POST"
            assert json.loads(request.content) == {
                "media_category": "tweet_video",
                "media_type": "video/mp4",
                "shared": False,
                "total_bytes": len(b"source-video"),
            }
            return httpx.Response(200, json={"data": {"id": "video-1"}})
        if request.url.path == "/2/media/upload/video-1/append":
            assert request.method == "POST"
            assert b"source-video" in request.content
            assert b"segment_index" in request.content
            return httpx.Response(200, json={"data": {"expires_at": 1}})
        if request.url.path == "/2/media/upload/video-1/finalize":
            assert request.method == "POST"
            return httpx.Response(
                200,
                json={
                    "data": {
                        "id": "video-1",
                        "processing_info": {"state": "succeeded"},
                    }
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    video = tmp_path / "source-1.mp4"
    video.write_bytes(b"source-video")
    client = XApiClient(
        base_url="https://api.x.com",
        auth_mode="oauth2",
        access_token="user-access-token",
        oauth2_scopes=frozenset({"media.write", "tweet.write"}),
        client_factory=_client_factory(handler),
    )

    assert await client.upload_media(video) == "video-1"
    assert requests == [
        ("POST", "/2/media/upload/initialize"),
        ("POST", "/2/media/upload/video-1/append"),
        ("POST", "/2/media/upload/video-1/finalize"),
    ]


@pytest.mark.asyncio
async def test_oauth1_video_upload_uses_dedicated_v2_endpoints(tmp_path: Path) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        assert request.headers["Authorization"].startswith("OAuth ")
        if request.url.path == "/2/media/upload/initialize":
            return httpx.Response(200, json={"data": {"id": "video-1"}})
        if request.url.path == "/2/media/upload/video-1/append":
            return httpx.Response(204)
        if request.url.path == "/2/media/upload/video-1/finalize":
            return httpx.Response(200, json={"data": {"id": "video-1"}})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    video = tmp_path / "source-1.mp4"
    video.write_bytes(b"source-video")
    client = XApiClient(
        base_url="https://api.x.com",
        auth_mode="oauth1",
        access_token="user-access-token",
        consumer_key="consumer",
        consumer_secret="consumer-secret",
        access_token_secret="token-secret",
        client_factory=_client_factory(handler),
    )

    assert await client.upload_media(video) == "video-1"
    assert paths == [
        "/2/media/upload/initialize",
        "/2/media/upload/video-1/append",
        "/2/media/upload/video-1/finalize",
    ]


@pytest.mark.asyncio
async def test_media_initialize_error_names_the_failed_stage(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/2/media/upload/initialize"
        return httpx.Response(
            400,
            json={
                "title": "Invalid Request",
                "detail": "One or more parameters to your request was invalid.",
            },
        )

    video = tmp_path / "source-1.mp4"
    video.write_bytes(b"source-video")
    client = XApiClient(
        base_url="https://api.x.com",
        auth_mode="oauth1",
        access_token="user-access-token",
        consumer_key="consumer",
        consumer_secret="consumer-secret",
        access_token_secret="token-secret",
        client_factory=_client_factory(handler),
    )

    with pytest.raises(XApiError) as raised:
        await client.upload_media(video)

    assert "initialization" in str(raised.value)
    assert "One or more parameters" in str(raised.value)


@pytest.mark.asyncio
async def test_media_initialize_403_gets_write_permission_guidance(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/2/media/upload/initialize"
        return httpx.Response(403, json={"detail": "Client is not permitted."})

    video = tmp_path / "source-1.mp4"
    video.write_bytes(b"source-video")
    client = XApiClient(
        base_url="https://api.x.com",
        auth_mode="oauth1",
        access_token="user-access-token",
        consumer_key="consumer",
        consumer_secret="consumer-secret",
        access_token_secret="token-secret",
        client_factory=_client_factory(handler),
    )

    with pytest.raises(XPermissionError, match="Read and write"):
        await client.upload_media(video)


@pytest.mark.asyncio
async def test_source_video_uses_chunked_upload_and_is_not_labeled_ai_generated(
    tmp_path: Path,
) -> None:
    stages: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer user-access-token"
        if request.url.path == "/2/media/upload/initialize":
            stages.append("INIT")
            body = json.loads(request.content)
            assert body["media_category"] == "tweet_video"
            assert body["shared"] is False
            return httpx.Response(200, json={"data": {"id": "video-1"}})
        if request.url.path == "/2/media/upload/video-1/append":
            stages.append("APPEND")
            assert b"source-video" in request.content
            return httpx.Response(200, json={"data": {"expires_at": 1}})
        if request.url.path == "/2/media/upload/video-1/finalize":
            stages.append("FINALIZE")
            return httpx.Response(
                200,
                json={
                    "data": {
                        "id": "video-1",
                        "processing_info": {"state": "succeeded"},
                    }
                },
            )
        if request.url.path == "/2/tweets":
            body = json.loads(request.content)
            assert body == {
                "text": "Messi had one allegedly arrogant moment.",
                "media": {"media_ids": ["video-1"]},
            }
            return httpx.Response(201, json={"data": {"id": "post-video"}})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    video = tmp_path / "source-1.mp4"
    video.write_bytes(b"source-video")
    client = XApiClient(
        base_url="https://api.x.com",
        auth_mode="oauth2",
        access_token="user-access-token",
        oauth2_scopes=frozenset({"media.write", "tweet.write"}),
        client_factory=_client_factory(handler),
    )

    media_id = await client.upload_media(video)
    result = await client.create_post(
        text="Messi had one allegedly arrogant moment.",
        reply_to_post_id=None,
        media_ids=(media_id,),
    )

    assert stages == ["INIT", "APPEND", "FINALIZE"]
    assert result["id"] == "post-video"


@pytest.mark.asyncio
async def test_create_quote_post_uses_official_quote_tweet_id_field() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/2/tweets"
        assert json.loads(request.content) == {
            "text": "the wording is doing a lot of work here",
            "quote_tweet_id": "2076668264755732487",
        }
        return httpx.Response(201, json={"data": {"id": "quote-post-1"}})

    client = XApiClient(
        base_url="https://api.x.com",
        auth_mode="oauth2",
        access_token="user-access-token",
        client_factory=_client_factory(handler),
    )

    result = await client.create_post(
        text="the wording is doing a lot of work here",
        reply_to_post_id=None,
        quote_post_id="2076668264755732487",
    )

    assert result["id"] == "quote-post-1"


@pytest.mark.asyncio
async def test_quote_post_rejects_reply_or_uploaded_media_before_network() -> None:
    network_called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal network_called
        network_called = True
        return httpx.Response(500)

    client = XApiClient(
        base_url="https://api.x.com",
        auth_mode="oauth2",
        access_token="user-access-token",
        client_factory=_client_factory(handler),
    )

    with pytest.raises(XApiError, match="both a reply and a quote"):
        await client.create_post(
            text="commentary",
            reply_to_post_id="123",
            quote_post_id="456",
        )
    with pytest.raises(XApiError, match="cannot include uploaded media"):
        await client.create_post(
            text="commentary",
            reply_to_post_id=None,
            media_ids=("media-1",),
            quote_post_id="456",
        )

    assert network_called is False


@pytest.mark.asyncio
async def test_create_post_403_surfaces_safe_write_permission_guidance() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/2/tweets"
        return httpx.Response(
            403,
            headers={"x-request-id": "req-403"},
            json={
                "title": "Forbidden",
                "detail": "You are not permitted to perform this action.",
                "type": "about:blank",
                "status": 403,
                "secret": "must-not-appear",
            },
        )

    client = XApiClient(
        base_url="https://api.x.com",
        auth_mode="oauth1",
        access_token="user-access-token",
        consumer_key="consumer",
        consumer_secret="consumer-secret",
        access_token_secret="token-secret",
        client_factory=_client_factory(handler),
    )

    with pytest.raises(XPermissionError) as raised:
        await client.create_post(text="Post", reply_to_post_id=None)

    message = str(raised.value)
    assert "Read and write" in message
    assert "regenerate" in message
    assert "You are not permitted" in message
    assert "must-not-appear" not in message
    assert raised.value.request_id == "req-403"
    assert raised.value.status_code == 403


@pytest.mark.asyncio
async def test_read_only_access_header_blocks_write_before_post_request() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/2/users/me":
            return httpx.Response(
                200,
                headers={"x-access-level": "read"},
                json={"data": {"id": "450", "username": "test_account"}},
            )
        raise AssertionError("POST should not be sent for a known read-only token")

    client = XApiClient(
        base_url="https://api.x.com",
        auth_mode="oauth1",
        access_token="user-access-token",
        consumer_key="consumer",
        consumer_secret="consumer-secret",
        access_token_secret="token-secret",
        client_factory=_client_factory(handler),
    )

    await client.get_me()
    with pytest.raises(XPermissionError, match="read-only"):
        await client.create_post(text="Post", reply_to_post_id=None)

    assert requests == ["/2/users/me"]


@pytest.mark.asyncio
async def test_oauth2_missing_tweet_write_scope_blocks_post_before_network() -> None:
    network_called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal network_called
        network_called = True
        return httpx.Response(500)

    client = XApiClient(
        base_url="https://api.x.com",
        auth_mode="oauth2",
        access_token="user-access-token",
        oauth2_scopes=frozenset({"tweet.read", "users.read"}),
        client_factory=_client_factory(handler),
    )

    with pytest.raises(XPermissionError, match=r"tweet\.write"):
        await client.create_post(text="Post", reply_to_post_id=None)

    assert network_called is False
