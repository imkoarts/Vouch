"""Deterministic source-context planning and bounded media download.

The language model never decides whether unrelated posts should be combined or whether a
media-dependent source can be published without its context. This module inspects normalized X
metadata, selects one safe source-media asset, and stores it inside the draft bundle.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from app.schemas.content import UntrustedSourceData

SourceContextStrategy = Literal["standalone", "attach_source_media", "quote_post"]
_ALLOWED_MEDIA_HOST_SUFFIXES = (".twimg.com",)
_MAX_SOURCE_MEDIA_BYTES = 128 * 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_MEDIA_DOWNLOAD_ATTEMPTS_PER_URL = 2
_MEDIA_DOWNLOAD_RETRY_DELAY_SECONDS = 2.0
_RETRYABLE_MEDIA_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}

_LOGGER = logging.getLogger(__name__)


class SourceMediaError(RuntimeError):
    """A source needs context but its official media could not be materialized safely."""


@dataclass(frozen=True, slots=True)
class SourceMediaAsset:
    media_key: str
    media_type: str
    url: str
    content_type: str
    width: int | None = None
    height: int | None = None
    duration_ms: int | None = None
    bit_rate: int | None = None
    alt_text: str = ""
    fallback_urls: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceContextPlan:
    strategy: SourceContextStrategy
    source_post_id: str | None
    source_url: str | None
    source_username: str | None
    assets: tuple[SourceMediaAsset, ...] = ()
    media_context_required: bool = False

    @property
    def has_media(self) -> bool:
        return bool(self.assets)


def _safe_media_url(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    url = value.strip()
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    if parsed.scheme != "https" or not host.endswith(_ALLOWED_MEDIA_HOST_SUFFIXES):
        return None
    return url


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _video_asset(item: Mapping[str, Any]) -> SourceMediaAsset | None:
    variants = item.get("variants", ())
    if isinstance(variants, (str, bytes)) or not isinstance(variants, Sequence):
        variants = ()
    candidates: list[tuple[int, Mapping[str, Any], str]] = []
    for variant in variants:
        if not isinstance(variant, Mapping):
            continue
        content_type = str(variant.get("content_type", "")).casefold()
        if content_type != "video/mp4":
            continue
        url = _safe_media_url(variant.get("url"))
        if url is None:
            continue
        candidates.append((_integer(variant.get("bit_rate")) or 0, variant, url))
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda row: row[0], reverse=True)
    bit_rate, _, url = ordered[0]
    fallback_urls = tuple(dict.fromkeys(candidate_url for _, _, candidate_url in ordered[1:]))
    return SourceMediaAsset(
        media_key=str(item.get("media_key", "")),
        media_type=str(item.get("type", "video")),
        url=url,
        content_type="video/mp4",
        width=_integer(item.get("width")),
        height=_integer(item.get("height")),
        duration_ms=_integer(item.get("duration_ms")),
        bit_rate=bit_rate or None,
        alt_text=str(item.get("alt_text", "") or ""),
        fallback_urls=fallback_urls,
    )


def _image_asset(item: Mapping[str, Any]) -> SourceMediaAsset | None:
    urls = tuple(
        dict.fromkeys(
            candidate
            for candidate in (
                _safe_media_url(item.get("url")),
                _safe_media_url(item.get("preview_image_url")),
            )
            if candidate is not None
        )
    )
    if not urls:
        return None
    url = urls[0]
    path = urlsplit(url).path.casefold()
    guessed = mimetypes.guess_type(path)[0]
    content_type = guessed if guessed in {"image/jpeg", "image/png", "image/webp"} else "image/jpeg"
    return SourceMediaAsset(
        media_key=str(item.get("media_key", "")),
        media_type=str(item.get("type", "photo")),
        url=url,
        content_type=content_type,
        width=_integer(item.get("width")),
        height=_integer(item.get("height")),
        alt_text=str(item.get("alt_text", "") or ""),
        fallback_urls=urls[1:],
    )


def plan_source_context(sources: Sequence[UntrustedSourceData]) -> SourceContextPlan:
    """Choose one context strategy for the sole editorial anchor."""

    if not sources:
        return SourceContextPlan("standalone", None, None, None)
    source = sources[0]
    raw_items = source.media_metadata.get("items", ())
    items = (
        tuple(item for item in raw_items if isinstance(item, Mapping))
        if isinstance(raw_items, Sequence) and not isinstance(raw_items, (str, bytes))
        else ()
    )
    assets: list[SourceMediaAsset] = []
    for item in items:
        media_type = str(item.get("type", "")).casefold()
        asset = (
            _video_asset(item) if media_type in {"video", "animated_gif"} else _image_asset(item)
        )
        if asset is not None:
            assets.append(asset)
    if assets:
        # X allows one video/GIF or up to four images. Keep the deterministic first video or four
        # images; never mix media categories from separate source posts.
        videos = [asset for asset in assets if asset.content_type == "video/mp4"]
        selected = (videos[0],) if videos else tuple(assets[:4])
        return SourceContextPlan(
            "attach_source_media",
            source.source_id,
            source.url,
            source.author_username,
            selected,
            True,
        )
    if items:
        return SourceContextPlan(
            "quote_post",
            source.source_id,
            source.url,
            source.author_username,
            (),
            True,
        )
    return SourceContextPlan(
        "standalone",
        source.source_id,
        source.url,
        source.author_username,
    )


def _extension(asset: SourceMediaAsset) -> str:
    return {
        "video/mp4": ".mp4",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/jpeg": ".jpg",
    }.get(asset.content_type, ".bin")


async def download_source_media(
    plan: SourceContextPlan,
    directory: Path,
    *,
    timeout_seconds: float = 45.0,
    maximum_bytes: int = _MAX_SOURCE_MEDIA_BYTES,
    proxy_url: str | None = None,
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
) -> dict[str, object]:
    """Download selected official X CDN assets into a draft bundle.

    Only HTTPS ``*.twimg.com`` URLs are accepted. When an outbound proxy is configured, the
    CDN download is forced through that explicit proxy with environment proxy discovery disabled.
    Each selected URL receives one bounded retry, then lower-bitrate or preview variants are tried.
    Failed partial files are always removed.
    """

    if plan.strategy == "standalone":
        return {
            "type": "none",
            "reason": "The single source anchor is understandable without external context.",
            "required_files": [],
            "metadata": {
                "context_strategy": "standalone",
                "source_post_id": plan.source_post_id,
                "source_url": plan.source_url,
            },
        }
    if plan.strategy == "quote_post":
        if not plan.source_url:
            raise SourceMediaError("The source requires quote context but has no canonical URL")
        if plan.media_context_required:
            raise SourceMediaError(
                "The source contains media, but no safe downloadable X media variant is available"
            )
        return {
            "type": "none",
            "reason": "Use the original X post as quote context; do not summarize it in copy.",
            "required_files": [],
            "metadata": {
                "context_strategy": "quote_post",
                "quote_source_post_id": plan.source_post_id,
                "quote_source_url": plan.source_url,
            },
        }
    if not plan.assets:
        raise SourceMediaError("The source requires media context but no downloadable asset exists")

    media_directory = directory / "media"
    media_directory.mkdir(parents=True, exist_ok=True)
    factory = client_factory or (
        lambda: httpx.AsyncClient(
            proxy=proxy_url,
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 15.0)),
            follow_redirects=False,
            trust_env=False,
        )
    )
    required_files: list[str] = []
    files_metadata: dict[str, dict[str, object]] = {}
    total = 0

    async def fetch_asset(
        client: httpx.AsyncClient,
        asset: SourceMediaAsset,
        temporary: Path,
    ) -> tuple[int, str]:
        urls = tuple(dict.fromkeys((asset.url, *asset.fallback_urls)))
        last_error: Exception | None = None
        for url_index, url in enumerate(urls, start=1):
            for attempt in range(1, _MEDIA_DOWNLOAD_ATTEMPTS_PER_URL + 1):
                await asyncio.to_thread(temporary.unlink, missing_ok=True)
                written = 0
                try:
                    async with client.stream(
                        "GET", url, headers={"Accept": asset.content_type}
                    ) as response:
                        if response.status_code >= 400:
                            error = SourceMediaError(
                                f"X source media download failed with HTTP {response.status_code}"
                            )
                            if response.status_code not in _RETRYABLE_MEDIA_STATUS_CODES:
                                last_error = error
                                break
                            raise error
                        content_length = _integer(response.headers.get("content-length"))
                        if content_length is not None and total + content_length > maximum_bytes:
                            raise SourceMediaError(
                                "X source media exceeds the configured download limit"
                            )
                        with temporary.open("wb") as handle:
                            async for chunk in response.aiter_bytes(_DOWNLOAD_CHUNK_BYTES):
                                if not chunk:
                                    continue
                                written += len(chunk)
                                if total + written > maximum_bytes:
                                    raise SourceMediaError(
                                        "X source media exceeds the configured download limit"
                                    )
                                handle.write(chunk)
                    if written <= 0:
                        raise SourceMediaError("X source media download returned an empty file")
                    return written, url
                except (httpx.HTTPError, SourceMediaError) as exc:
                    last_error = exc
                    await asyncio.to_thread(temporary.unlink, missing_ok=True)
                    retryable = isinstance(exc, httpx.HTTPError) or (
                        isinstance(exc, SourceMediaError) and "HTTP " in str(exc)
                    )
                    if attempt < _MEDIA_DOWNLOAD_ATTEMPTS_PER_URL and retryable:
                        _LOGGER.warning(
                            "X media download failed for variant %s/%s attempt %s/%s; retrying",
                            url_index,
                            len(urls),
                            attempt,
                            _MEDIA_DOWNLOAD_ATTEMPTS_PER_URL,
                        )
                        await asyncio.sleep(_MEDIA_DOWNLOAD_RETRY_DELAY_SECONDS)
                        continue
                    break
        raise SourceMediaError(
            f"X source media download failed after {len(urls)} bounded variant(s)"
        ) from last_error

    try:
        async with factory() as client:
            for index, asset in enumerate(plan.assets, start=1):
                relative = f"media/source-{index}{_extension(asset)}"
                target = directory / relative
                temporary = target.with_suffix(target.suffix + ".tmp")
                written, selected_url = await fetch_asset(client, asset, temporary)
                temporary.replace(target)
                total += written
                required_files.append(relative)
                files_metadata[relative] = {
                    "width": asset.width,
                    "height": asset.height,
                    "duration_ms": asset.duration_ms,
                    "source_media_key": asset.media_key,
                    "source_url": selected_url,
                    "fallback_variant_count": len(asset.fallback_urls),
                }
    except Exception:
        for temporary in media_directory.glob("*.tmp"):
            temporary.unlink(missing_ok=True)
        for relative in required_files:
            (directory / relative).unlink(missing_ok=True)
        raise

    media_type = "video" if any(path.endswith(".mp4") for path in required_files) else "image"
    return {
        "type": media_type,
        "reason": "Source media is attached so the post keeps the context visible in the original.",
        "required_files": required_files,
        "metadata": {
            "context_strategy": "attach_source_media",
            "source_post_id": plan.source_post_id,
            "source_url": plan.source_url,
            "files": files_metadata,
        },
    }
