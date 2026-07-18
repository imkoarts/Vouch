from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.schemas.content import UntrustedSourceData
from app.services.source_media import download_source_media, plan_source_context


def _source(*, items: list[dict[str, object]]) -> UntrustedSourceData:
    return UntrustedSourceData(
        source_id="2076265333803913327",
        content="the only time i saw messi arrogant",
        author_username="bdll_10",
        url="https://x.com/bdll_10/status/2076265333803913327",
        media_metadata={"items": items},
    )


def test_plan_selects_highest_bitrate_source_video() -> None:
    plan = plan_source_context(
        (
            _source(
                items=[
                    {
                        "media_key": "7_1",
                        "type": "video",
                        "duration_ms": 8000,
                        "variants": [
                            {
                                "content_type": "video/mp4",
                                "bit_rate": 256000,
                                "url": "https://video.twimg.com/a-low.mp4",
                            },
                            {
                                "content_type": "video/mp4",
                                "bit_rate": 2176000,
                                "url": "https://video.twimg.com/a-high.mp4",
                            },
                        ],
                    }
                ]
            ),
        )
    )

    assert plan.strategy == "attach_source_media"
    assert len(plan.assets) == 1
    assert plan.assets[0].url.endswith("a-high.mp4")
    assert plan.assets[0].bit_rate == 2176000


def test_plan_uses_quote_context_when_media_has_no_safe_download_url() -> None:
    plan = plan_source_context(
        (
            _source(
                items=[
                    {
                        "media_key": "7_1",
                        "type": "video",
                        "variants": [
                            {
                                "content_type": "video/mp4",
                                "url": "https://example.com/video.mp4",
                            }
                        ],
                    }
                ]
            ),
        )
    )

    assert plan.strategy == "quote_post"
    assert plan.source_url == "https://x.com/bdll_10/status/2076265333803913327"


@pytest.mark.asyncio
async def test_download_materializes_video_inside_draft_bundle(tmp_path: Path) -> None:
    plan = plan_source_context(
        (
            _source(
                items=[
                    {
                        "media_key": "7_1",
                        "type": "video",
                        "width": 1280,
                        "height": 720,
                        "duration_ms": 8000,
                        "variants": [
                            {
                                "content_type": "video/mp4",
                                "bit_rate": 2176000,
                                "url": "https://video.twimg.com/source.mp4",
                            }
                        ],
                    }
                ]
            ),
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "video.twimg.com"
        return httpx.Response(200, content=b"source-video-bytes")

    plan_data = await download_source_media(
        plan,
        tmp_path,
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    assert plan_data["type"] == "video"
    assert plan_data["required_files"] == ["media/source-1.mp4"]
    assert (tmp_path / "media" / "source-1.mp4").read_bytes() == b"source-video-bytes"
    assert plan_data["metadata"]["context_strategy"] == "attach_source_media"


@pytest.mark.asyncio
async def test_download_uses_explicit_outbound_proxy_and_ignores_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = plan_source_context(
        (
            _source(
                items=[
                    {
                        "media_key": "7_1",
                        "type": "video",
                        "variants": [
                            {
                                "content_type": "video/mp4",
                                "bit_rate": 2176000,
                                "url": "https://video.twimg.com/source.mp4",
                            }
                        ],
                    }
                ]
            ),
        )
    )
    captured: dict[str, object] = {}
    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"proxied-source-video")

    def recording_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        captured.update(kwargs)
        forwarded = dict(kwargs)
        forwarded.pop("proxy", None)
        return real_async_client(
            *args,
            **forwarded,
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr("app.services.source_media.httpx.AsyncClient", recording_client)

    await download_source_media(
        plan,
        tmp_path,
        proxy_url="http://user:pass@127.0.0.1:8080",
    )

    assert captured["proxy"] == "http://user:pass@127.0.0.1:8080"
    assert captured["trust_env"] is False
    assert (tmp_path / "media" / "source-1.mp4").read_bytes() == b"proxied-source-video"


def test_video_plan_keeps_lower_bitrate_variants_as_fallbacks() -> None:
    plan = plan_source_context(
        (
            _source(
                items=[
                    {
                        "media_key": "7_1",
                        "type": "video",
                        "variants": [
                            {
                                "content_type": "video/mp4",
                                "bit_rate": 256000,
                                "url": "https://video.twimg.com/a-low.mp4",
                            },
                            {
                                "content_type": "video/mp4",
                                "bit_rate": 2176000,
                                "url": "https://video.twimg.com/a-high.mp4",
                            },
                        ],
                    }
                ]
            ),
        )
    )

    assert plan.media_context_required is True
    assert plan.assets[0].url.endswith("a-high.mp4")
    assert plan.assets[0].fallback_urls == ("https://video.twimg.com/a-low.mp4",)


@pytest.mark.asyncio
async def test_download_retries_then_uses_lower_bitrate_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = plan_source_context(
        (
            _source(
                items=[
                    {
                        "media_key": "7_1",
                        "type": "video",
                        "variants": [
                            {
                                "content_type": "video/mp4",
                                "bit_rate": 2176000,
                                "url": "https://video.twimg.com/high.mp4",
                            },
                            {
                                "content_type": "video/mp4",
                                "bit_rate": 256000,
                                "url": "https://video.twimg.com/low.mp4",
                            },
                        ],
                    }
                ]
            ),
        )
    )
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.path.endswith("high.mp4"):
            raise httpx.ConnectTimeout("synthetic timeout", request=request)
        return httpx.Response(200, content=b"lower-bitrate-video")

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("app.services.source_media.asyncio.sleep", no_sleep)
    plan_data = await download_source_media(
        plan,
        tmp_path,
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    assert requests == [
        "https://video.twimg.com/high.mp4",
        "https://video.twimg.com/high.mp4",
        "https://video.twimg.com/low.mp4",
    ]
    assert (tmp_path / "media" / "source-1.mp4").read_bytes() == b"lower-bitrate-video"
    assert (
        plan_data["metadata"]["files"]["media/source-1.mp4"]["source_url"]
        == "https://video.twimg.com/low.mp4"
    )


@pytest.mark.asyncio
async def test_media_items_without_safe_variant_require_manual_fallback(tmp_path: Path) -> None:
    from app.services.source_media import SourceMediaError

    plan = plan_source_context(
        (
            _source(
                items=[
                    {
                        "media_key": "7_1",
                        "type": "video",
                        "variants": [
                            {
                                "content_type": "video/mp4",
                                "url": "https://example.com/video.mp4",
                            }
                        ],
                    }
                ]
            ),
        )
    )

    assert plan.strategy == "quote_post"
    assert plan.media_context_required is True
    with pytest.raises(SourceMediaError, match="no safe downloadable"):
        await download_source_media(plan, tmp_path)
