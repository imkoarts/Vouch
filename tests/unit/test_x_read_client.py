from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from app.x_api.read import XReadClient, XReadError


def _client_factory(handler: Any) -> Any:
    transport = httpx.MockTransport(handler)
    return lambda: httpx.AsyncClient(transport=transport)


@pytest.mark.asyncio
async def test_home_timeline_is_user_context_and_hard_capped() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer user-token"
        assert request.url.path == "/2/users/450/timelines/reverse_chronological"
        assert request.url.params["max_results"] == "5"
        assert request.url.params["exclude"] == "replies,retweets"
        assert "public_metrics" in request.url.params["tweet.fields"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": str(index),
                        "text": f"Post {index}",
                        "public_metrics": {"like_count": index},
                    }
                    for index in range(1, 8)
                ]
            },
        )

    client = XReadClient(
        base_url="https://api.x.com",
        auth_mode="oauth2",
        bearer_token="app-token",
        access_token="user-token",
        client_factory=_client_factory(handler),
    )
    rows = await client.get_home_timeline(
        "450",
        max_results=5,
        start_time=datetime(2026, 7, 11, tzinfo=UTC),
    )

    assert len(rows) == 5
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_home_timeline_uses_since_id_instead_of_replaying_lookback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["since_id"] == "2078157934983598289"
        assert "start_time" not in request.url.params
        return httpx.Response(200, json={"data": []})

    client = XReadClient(
        base_url="https://api.x.com",
        auth_mode="oauth2",
        access_token="user-token",
        client_factory=_client_factory(handler),
    )

    rows = await client.get_home_timeline(
        "450",
        max_results=5,
        start_time=datetime(2026, 7, 11, tzinfo=UTC),
        since_id="2078157934983598289",
    )

    assert rows == ()


@pytest.mark.asyncio
async def test_trends_use_app_bearer_and_recent_search_is_blocked() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        assert request.headers["Authorization"] == "Bearer app-token"
        assert request.url.path == "/2/trends/by/woeid/1"
        assert request.url.params["max_trends"] == "3"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"trend_name": "World Cup", "tweet_count": 1000},
                    {"trend_name": "Football", "tweet_count": 900},
                ]
            },
        )

    client = XReadClient(
        base_url="https://api.x.com",
        auth_mode="oauth1",
        bearer_token="app-token",
        client_factory=_client_factory(handler),
    )
    trends = await client.get_trends_by_woeid(1, max_trends=3)
    with pytest.raises(XReadError, match="Recent Search is disabled"):
        await client.recent_search(
            '"World Cup" -is:retweet -is:reply',
            max_results=10,
            start_time=datetime(2026, 7, 11, tzinfo=UTC),
        )

    assert trends == ("World Cup", "Football")
    assert paths == ["/2/trends/by/woeid/1"]


@pytest.mark.asyncio
async def test_read_limits_fail_before_network() -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    client = XReadClient(
        base_url="https://api.x.com",
        auth_mode="oauth2",
        access_token="user-token",
        client_factory=_client_factory(handler),
    )
    with pytest.raises(ValueError, match="between 1 and 10"):
        await client.get_home_timeline(
            "450",
            max_results=11,
            start_time=datetime.now(UTC),
        )
    with pytest.raises(XReadError, match="Recent Search is disabled"):
        await client.recent_search(
            "football",
            max_results=11,
            start_time=datetime.now(UTC),
        )
    assert called is False


@pytest.mark.asyncio
async def test_payment_required_is_reported_as_credit_problem() -> None:
    from app.x_api.read import XReadPaymentRequiredError

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"title": "Payment Required"})

    client = XReadClient(
        base_url="https://api.x.com",
        auth_mode="oauth1",
        bearer_token="app-token",
        client_factory=_client_factory(handler),
    )
    with pytest.raises(XReadPaymentRequiredError, match="credits"):
        await client.get_trends_by_woeid(1, max_trends=3)


@pytest.mark.asyncio
async def test_temporary_503_uses_bounded_backoff_then_succeeds() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503, headers={"Retry-After": "1"})
        return httpx.Response(200, json={"data": [{"trend_name": "AI"}]})

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    client = XReadClient(
        base_url="https://api.x.com",
        auth_mode="oauth1",
        bearer_token="app-token",
        temporary_error_max_retries=2,
        temporary_error_initial_backoff_seconds=1.0,
        temporary_error_max_backoff_seconds=4.0,
        sleep=fake_sleep,
        client_factory=_client_factory(handler),
    )

    trends = await client.get_trends_by_woeid(1, max_trends=3)

    assert trends == ("AI",)
    assert attempts == 3
    assert client.last_attempt_count == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.asyncio
async def test_temporary_503_exhaustion_raises_specific_error() -> None:
    from app.x_api.read import XReadTemporaryError

    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503)

    async def no_sleep(_seconds: float) -> None:
        return None

    client = XReadClient(
        base_url="https://api.x.com",
        auth_mode="oauth1",
        bearer_token="app-token",
        temporary_error_max_retries=2,
        temporary_error_initial_backoff_seconds=0.0,
        temporary_error_max_backoff_seconds=0.0,
        sleep=no_sleep,
        client_factory=_client_factory(handler),
    )

    with pytest.raises(XReadTemporaryError) as captured:
        await client.get_trends_by_woeid(1, max_trends=3)

    assert captured.value.status_code == 503
    assert captured.value.attempts == 3
    assert attempts == 3


@pytest.mark.asyncio
async def test_permanent_400_is_not_retried() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400)

    client = XReadClient(
        base_url="https://api.x.com",
        auth_mode="oauth1",
        bearer_token="app-token",
        client_factory=_client_factory(handler),
    )

    with pytest.raises(Exception, match="HTTP 400"):
        await client.get_trends_by_woeid(1, max_trends=3)

    assert attempts == 1


@pytest.mark.asyncio
async def test_retweet_is_resolved_to_original_post_with_video_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "referenced_tweets.id.attachments.media_keys" in request.url.params["expansions"]
        assert "variants" in request.url.params["media.fields"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "wrapper-1",
                        "text": "RT @bdll_10: original caption",
                        "author_id": "viewer-1",
                        "referenced_tweets": [{"type": "retweeted", "id": "original-1"}],
                    }
                ],
                "includes": {
                    "tweets": [
                        {
                            "id": "original-1",
                            "text": "the only time i saw messi arrogant",
                            "author_id": "author-1",
                            "attachments": {"media_keys": ["7_video"]},
                        }
                    ],
                    "users": [{"id": "author-1", "username": "bdll_10"}],
                    "media": [
                        {
                            "media_key": "7_video",
                            "type": "video",
                            "variants": [
                                {
                                    "content_type": "video/mp4",
                                    "bit_rate": 2176000,
                                    "url": "https://video.twimg.com/messi.mp4",
                                }
                            ],
                        }
                    ],
                },
            },
        )

    client = XReadClient(
        base_url="https://api.x.com",
        auth_mode="oauth2",
        access_token="user-token",
        client_factory=_client_factory(handler),
    )

    rows = await client.get_home_timeline(
        "450",
        max_results=5,
        start_time=datetime(2026, 7, 12, tzinfo=UTC),
    )

    assert len(rows) == 1
    assert rows[0]["id"] == "original-1"
    assert rows[0]["username"] == "bdll_10"
    assert rows[0]["url"] == "https://x.com/bdll_10/status/original-1"
    assert rows[0]["media_metadata"]["reference_type"] == "retweeted"
    assert rows[0]["media_metadata"]["items"][0]["type"] == "video"
