"""Local-first dashboard shared by browser and desktop shells."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, Select, and_, func, or_, select

from app import __version__
from app.config import Settings
from app.container import build_services
from app.domain.state_machine import DraftStateMachineError
from app.models import DashboardChatMessage, Draft, DraftVersion, IdeaSource, SourcePost
from app.models.enums import ContentType, DraftStatus
from app.services.approval_service import ApprovalGateError
from app.services.configuration import ConfigurationService
from app.services.draft_service import DraftNotFoundError, DraftValidationError
from app.services.environment_config import credential_status, update_environment_file
from app.services.idea_collector import IdeaCollector
from app.services.publishing_service import (
    AmbiguousClaimError,
    PublicationGateError,
    PublishingAccountMismatchError,
)
from app.services.tenant_context import ContextSessionFactory, ContextSettings
from app.services.trend_discovery import AutomaticDiscoveryService, build_x_read_client
from app.utils.errors import safe_exception_summary
from app.x_api.mock import MockXClient

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_EDITABLE_CONFIG_FILES = (
    "runtime.yml",
    "content_sources.yml",
    "content_profile.yml",
    "style_examples.yml",
    "cost_estimates.yml",
)
_USERNAME = re.compile(r"^[A-Za-z0-9_]{1,50}$")


class DashboardSettingsPayload(BaseModel):
    """Structured non-secret settings exposed by the local dashboard."""

    model_config = ConfigDict(extra="forbid")

    generation_provider: Literal["mock", "openai", "xai"]
    post_length_mode: Literal["short", "medium", "longread"]
    humanizer_enabled: bool
    images_enabled: bool

    automatic_discovery_enabled: bool
    interval_preset: Literal["1h", "3h", "6h", "12h", "custom"]
    custom_interval_minutes: int = Field(ge=10, le=10_080)
    max_runs_per_utc_day: int = Field(ge=1, le=144)
    lookback_hours: int = Field(ge=1, le=168)
    trends_woeid: int = Field(ge=1)
    max_trends: int = Field(ge=1, le=10)
    max_total_posts: int = Field(ge=5, le=30)
    final_candidates: int = Field(ge=1, le=10)
    generation_candidates_per_run: int = Field(ge=1, le=10)
    notify_when_no_candidate: bool

    home_timeline_enabled: bool
    home_max_posts: int = Field(ge=1, le=10)
    home_exclude_replies: bool
    home_exclude_retweets: bool
    manual_sources_enabled: bool
    evergreen_enabled: bool
    evergreen_topics: list[str] = Field(max_length=100)

    telegram_enabled: bool
    telegram_autostart: bool
    telegram_notify_on_new_draft: bool

    manual_x_publish_enabled: bool
    enterprise_quote_posts_enabled: bool

    x_request_delay_seconds: float = Field(ge=0, le=60)
    llm_minimum_interval_seconds: float = Field(ge=0, le=3600)


class SelectedAccountCreatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=200)


class SelectedAccountUpdatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class SelectedAccountsSettingsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    max_posts_per_account: int = Field(ge=5, le=100)


class CommandPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=500)


class DraftEditPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=25_000)


class DraftRewritePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback: str = Field(default="", max_length=2_000)


class DraftApprovalPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts_confirmed: bool = False


class DraftPublishPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation_phrase: str = Field(min_length=1, max_length=300)


class ConfigFilePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=1_000_000)


class CredentialSettingsPayload(BaseModel):
    """Write-only credential replacements; blank values preserve current secrets."""

    model_config = ConfigDict(extra="forbid")

    values: dict[str, str] = Field(default_factory=dict, max_length=32)
    clear: list[str] = Field(default_factory=list, max_length=32)


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value is not None else None


def _author_metadata(post: SourcePost) -> dict[str, Any]:
    raw = post.raw_json if isinstance(post.raw_json, Mapping) else {}
    author = raw.get("author", {}) if isinstance(raw, Mapping) else {}
    if not isinstance(author, Mapping):
        author = {}
    username = post.username or str(author.get("username") or "").strip() or None
    name = str(author.get("name") or "").strip() or username
    avatar_url = str(author.get("profile_image_url") or "").strip() or None
    return {
        "username": username,
        "name": name,
        "avatar_url": avatar_url,
        "verified": bool(author.get("verified", False)),
    }


def _media_items(post: SourcePost) -> list[dict[str, Any]]:
    metadata = post.media_metadata if isinstance(post.media_metadata, Mapping) else {}
    raw_items = metadata.get("items", [])
    if not isinstance(raw_items, list):
        return []
    items: list[dict[str, Any]] = []
    for raw in raw_items[:4]:
        if not isinstance(raw, Mapping):
            continue
        variants = raw.get("variants", [])
        video_url: str | None = None
        if isinstance(variants, list):
            candidates = [
                item
                for item in variants
                if isinstance(item, Mapping)
                and str(item.get("content_type", "")).startswith("video/")
                and item.get("url")
            ]
            candidates.sort(key=lambda item: int(item.get("bit_rate") or 0), reverse=True)
            if candidates:
                video_url = str(candidates[0]["url"])
        items.append(
            {
                "type": str(raw.get("type") or "photo"),
                "url": str(raw.get("url") or "").strip() or None,
                "preview_image_url": str(raw.get("preview_image_url") or "").strip() or None,
                "video_url": video_url,
                "alt_text": str(raw.get("alt_text") or "").strip() or None,
                "width": raw.get("width"),
                "height": raw.get("height"),
            }
        )
    return items


def serialize_source_post(post: SourcePost) -> dict[str, Any]:
    """Return the durable UI representation of one discovered source post."""

    return {
        "id": post.id,
        "x_post_id": post.x_post_id,
        "url": post.url,
        "text": post.text,
        "author": _author_metadata(post),
        "published_at": _iso(post.published_at),
        "fetched_at": _iso(post.fetched_at),
        "updated_at": _iso(post.updated_at),
        "language": post.language,
        "public_metrics": dict(post.public_metrics or {}),
        "media": _media_items(post),
        "source_labels": list(post.source_labels or []),
    }


def serialize_chat_message(message: DashboardChatMessage) -> dict[str, Any]:
    return {
        "id": message.id,
        "role": message.role,
        "text": message.text,
        "command": message.command,
        "metadata": dict(message.message_metadata or {}),
        "created_at": _iso(message.created_at),
    }


def _parse_cursor(raw_time: str | None) -> datetime | None:
    if not raw_time:
        return None
    try:
        value = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid cursor") from exc
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _activity_query(
    *,
    after_time: datetime | None = None,
    after_id: str | None = None,
    before_time: datetime | None = None,
    before_id: str | None = None,
    descending: bool,
    limit: int,
) -> Select[tuple[SourcePost]]:
    query = select(SourcePost)
    if after_time is not None:
        query = query.where(
            or_(
                SourcePost.updated_at > after_time,
                and_(SourcePost.updated_at == after_time, SourcePost.id > (after_id or "")),
            )
        )
    if before_time is not None:
        query = query.where(
            or_(
                SourcePost.updated_at < before_time,
                and_(SourcePost.updated_at == before_time, SourcePost.id < (before_id or "~")),
            )
        )
    ordering = (
        (SourcePost.updated_at.desc(), SourcePost.id.desc())
        if descending
        else (SourcePost.updated_at.asc(), SourcePost.id.asc())
    )
    return query.order_by(*ordering).limit(limit)


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise HTTPException(status_code=500, detail=f"Invalid configuration file: {path.name}")
    return loaded


def _write_yaml_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(dict(payload), allow_unicode=False, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _normalized_topics(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.strip().split())
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        if len(normalized) > 200:
            raise HTTPException(
                status_code=422, detail="Evergreen topics must be 200 characters or less"
            )
        seen.add(key)
        result.append(normalized)
    return result


def _dashboard_settings(settings: Settings) -> DashboardSettingsPayload:
    configuration = ConfigurationService(settings.config_dir).load()
    discovery = configuration.sources.automatic_discovery
    sources = configuration.sources.sources
    runtime = configuration.runtime
    return DashboardSettingsPayload(
        generation_provider=runtime.generation.provider,
        post_length_mode=runtime.generation.post_length_mode.value,
        humanizer_enabled=runtime.generation.humanizer_enabled,
        images_enabled=runtime.images.enabled,
        automatic_discovery_enabled=discovery.enabled,
        interval_preset=discovery.interval_preset,
        custom_interval_minutes=discovery.custom_interval_minutes,
        max_runs_per_utc_day=discovery.max_runs_per_utc_day,
        lookback_hours=discovery.lookback_hours,
        trends_woeid=discovery.trends_woeid,
        max_trends=discovery.max_trends,
        max_total_posts=discovery.max_total_posts,
        final_candidates=discovery.final_candidates,
        generation_candidates_per_run=discovery.generation_candidates_per_run,
        notify_when_no_candidate=discovery.notify_when_no_candidate,
        home_timeline_enabled=sources.home_timeline.enabled,
        home_max_posts=sources.home_timeline.max_posts,
        home_exclude_replies=sources.home_timeline.exclude_replies,
        home_exclude_retweets=sources.home_timeline.exclude_retweets,
        manual_sources_enabled=sources.manual.enabled,
        evergreen_enabled=sources.evergreen.enabled,
        evergreen_topics=list(sources.evergreen.topics),
        telegram_enabled=runtime.telegram.enabled,
        telegram_autostart=runtime.telegram.autostart,
        telegram_notify_on_new_draft=runtime.telegram.notify_on_new_draft,
        manual_x_publish_enabled=runtime.publication.manual_x_publish_enabled,
        enterprise_quote_posts_enabled=runtime.publication.enterprise_quote_posts_enabled,
        x_request_delay_seconds=runtime.request_pacing.x_request_delay_seconds,
        llm_minimum_interval_seconds=runtime.request_pacing.llm_minimum_interval_seconds,
    )


def _save_dashboard_settings(settings: Settings, payload: DashboardSettingsPayload) -> None:
    sources_path = settings.config_dir / "content_sources.yml"
    runtime_path = settings.config_dir / "runtime.yml"
    source_backup = sources_path.with_suffix(".yml.dashboard-backup")
    runtime_backup = runtime_path.with_suffix(".yml.dashboard-backup")
    shutil.copy2(sources_path, source_backup)
    shutil.copy2(runtime_path, runtime_backup)

    try:
        sources_document = _read_yaml(sources_path)
        runtime_document = _read_yaml(runtime_path)

        automatic = sources_document.setdefault("automatic_discovery", {})
        if not isinstance(automatic, dict):
            raise HTTPException(status_code=500, detail="Invalid automatic discovery section")
        automatic.pop("run_on_start", None)
        automatic.update(
            {
                "enabled": payload.automatic_discovery_enabled,
                "interval_preset": payload.interval_preset,
                "custom_interval_minutes": payload.custom_interval_minutes,
                "max_runs_per_utc_day": payload.max_runs_per_utc_day,
                "lookback_hours": payload.lookback_hours,
                "trends_woeid": payload.trends_woeid,
                "max_trends": payload.max_trends,
                "max_total_posts": payload.max_total_posts,
                "final_candidates": payload.final_candidates,
                "generation_candidates_per_run": payload.generation_candidates_per_run,
                "notify_when_no_candidate": payload.notify_when_no_candidate,
            }
        )

        source_catalog = sources_document.setdefault("sources", {})
        if not isinstance(source_catalog, dict):
            raise HTTPException(status_code=500, detail="Invalid sources section")
        recent = source_catalog.setdefault("recent_search", {})
        if not isinstance(recent, dict):
            raise HTTPException(status_code=500, detail="Invalid recent_search section")
        recent["enabled"] = False

        home = source_catalog.setdefault("home_timeline", {})
        manual = source_catalog.setdefault("manual", {})
        evergreen = source_catalog.setdefault("evergreen", {})
        if not all(isinstance(section, dict) for section in (home, manual, evergreen)):
            raise HTTPException(status_code=500, detail="Invalid source configuration section")
        home.update(
            {
                "enabled": payload.home_timeline_enabled,
                "max_posts": payload.home_max_posts,
                "exclude_replies": payload.home_exclude_replies,
                "exclude_retweets": payload.home_exclude_retweets,
            }
        )
        manual["enabled"] = payload.manual_sources_enabled
        evergreen.update(
            {
                "enabled": payload.evergreen_enabled,
                "topics": _normalized_topics(payload.evergreen_topics),
            }
        )

        generation = runtime_document.setdefault("generation", {})
        images = runtime_document.setdefault("images", {})
        telegram = runtime_document.setdefault("telegram", {})
        publication = runtime_document.setdefault("publication", {})
        pacing = runtime_document.setdefault("request_pacing", {})
        if not all(
            isinstance(section, dict)
            for section in (generation, images, telegram, publication, pacing)
        ):
            raise HTTPException(status_code=500, detail="Invalid runtime configuration section")
        generation.update(
            {
                "provider": payload.generation_provider,
                "post_length_mode": payload.post_length_mode,
                "humanizer_enabled": payload.humanizer_enabled,
            }
        )
        images["enabled"] = payload.images_enabled
        telegram.update(
            {
                "enabled": payload.telegram_enabled,
                "autostart": payload.telegram_autostart,
                "notify_on_new_draft": payload.telegram_notify_on_new_draft,
            }
        )
        publication.update(
            {
                "manual_x_publish_enabled": payload.manual_x_publish_enabled,
                "enterprise_quote_posts_enabled": payload.enterprise_quote_posts_enabled,
            }
        )
        pacing.update(
            {
                "x_request_delay_seconds": payload.x_request_delay_seconds,
                "llm_minimum_interval_seconds": payload.llm_minimum_interval_seconds,
            }
        )

        _write_yaml_atomic(sources_path, sources_document)
        _write_yaml_atomic(runtime_path, runtime_document)
        ConfigurationService(settings.config_dir).load()
    except Exception:
        shutil.copy2(source_backup, sources_path)
        shutil.copy2(runtime_backup, runtime_path)
        raise
    finally:
        source_backup.unlink(missing_ok=True)
        runtime_backup.unlink(missing_ok=True)


def _normalize_username(value: str) -> str:
    raw = value.strip()
    parsed = urlparse(raw if "://" in raw else f"https://x.com/{raw}")
    if parsed.netloc.casefold() in {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}:
        candidate = parsed.path.strip("/").split("/", 1)[0]
    else:
        candidate = raw
    candidate = candidate.removeprefix("@").strip().casefold()
    if not _USERNAME.fullmatch(candidate):
        raise HTTPException(
            status_code=422,
            detail="Enter an X username, @username, or an x.com profile URL.",
        )
    return candidate


def _selected_accounts_payload(settings: Settings) -> dict[str, Any]:
    selected = ConfigurationService(settings.config_dir).load_sources().sources.selected_accounts
    return {
        "enabled": selected.enabled,
        "max_posts_per_account": selected.max_posts_per_account,
        "items": [
            {
                "username": account.username,
                "enabled": account.enabled,
                "user_id": account.user_id,
            }
            for account in selected.tracked_accounts
        ],
    }


def _mutate_selected_accounts(
    settings: Settings,
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    path = settings.config_dir / "content_sources.yml"
    backup = path.with_suffix(".yml.dashboard-backup")
    shutil.copy2(path, backup)
    try:
        document = _read_yaml(path)
        sources = document.setdefault("sources", {})
        if not isinstance(sources, dict):
            raise HTTPException(status_code=500, detail="Invalid sources section")
        recent = sources.setdefault("recent_search", {})
        selected = sources.setdefault("selected_accounts", {})
        if not isinstance(recent, dict) or not isinstance(selected, dict):
            raise HTTPException(status_code=500, detail="Invalid source configuration")
        recent["enabled"] = False
        existing = (
            ConfigurationService(settings.config_dir).load_sources().sources.selected_accounts
        )
        selected["accounts"] = [
            {
                "username": account.username,
                "enabled": account.enabled,
                **({"user_id": account.user_id} if account.user_id else {}),
            }
            for account in existing.tracked_accounts
        ]
        selected["usernames"] = []
        selected["user_ids"] = []
        mutator(selected)
        _write_yaml_atomic(path, document)
        ConfigurationService(settings.config_dir).load_sources()
    except Exception:
        shutil.copy2(backup, path)
        raise
    finally:
        backup.unlink(missing_ok=True)


def _config_path(settings: Settings, filename: str) -> Path:
    if filename not in _EDITABLE_CONFIG_FILES:
        raise HTTPException(status_code=404, detail="Configuration file not found")
    path = settings.config_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Configuration file not found")
    return path


def create_dashboard_router(settings: Settings, engine: Engine) -> APIRouter:
    """Create dashboard routes bound to the application's active database engine."""

    router = APIRouter()
    session_factory = ContextSessionFactory(engine)
    settings = ContextSettings(settings)  # type: ignore[assignment]
    discovery_lock = asyncio.Lock()
    manual_generation_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task[None]] = set()
    configuration_lock = threading.Lock()

    def query_posts(
        *,
        after_time: datetime | None = None,
        after_id: str | None = None,
        before_time: datetime | None = None,
        before_id: str | None = None,
        descending: bool,
        limit: int,
    ) -> list[dict[str, Any]]:
        with session_factory() as session:
            rows = session.scalars(
                _activity_query(
                    after_time=after_time,
                    after_id=after_id,
                    before_time=before_time,
                    before_id=before_id,
                    descending=descending,
                    limit=limit,
                )
            ).all()
            return [serialize_source_post(row) for row in rows]

    def status_payload() -> dict[str, Any]:
        with session_factory() as session:
            total_posts = int(session.scalar(select(func.count(SourcePost.id))) or 0)
        configuration = ConfigurationService(settings.config_dir).load()
        source_settings = configuration.sources.sources
        selected_accounts = source_settings.selected_accounts
        discovery = configuration.sources.automatic_discovery
        return {
            "version": __version__,
            "mode": "mock" if settings.mock_mode else "live",
            "host": settings.app_host,
            "port": settings.app_port,
            "total_posts": total_posts,
            "automatic_discovery": discovery.enabled,
            "discovery_interval_minutes": discovery.effective_interval_minutes,
            "startup_discovery": discovery.enabled,
            "home_timeline": source_settings.home_timeline.enabled,
            "trends": discovery.max_trends > 0,
            "telegram": configuration.runtime.telegram.enabled,
            "recent_search": False,
            "selected_accounts": sum(
                1 for account in selected_accounts.tracked_accounts if account.enabled
            ),
        }

    def save_chat_message(
        role: Literal["user", "assistant"],
        text: str,
        *,
        command: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> DashboardChatMessage:
        with session_factory.begin() as session:
            message = DashboardChatMessage(
                role=role,
                text=text,
                command=command,
                message_metadata=dict(metadata or {}),
            )
            session.add(message)
        return message

    def save_chat_pair(
        user_text: str,
        response_text: str,
        *,
        command: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[DashboardChatMessage, DashboardChatMessage]:
        created_at = datetime.now(UTC)
        with session_factory.begin() as session:
            user_message = DashboardChatMessage(
                role="user",
                text=user_text,
                command=command,
                message_metadata={},
                created_at=created_at,
                updated_at=created_at,
            )
            assistant_message = DashboardChatMessage(
                role="assistant",
                text=response_text,
                command=command,
                message_metadata=dict(metadata or {}),
                created_at=created_at + timedelta(microseconds=1),
                updated_at=created_at + timedelta(microseconds=1),
            )
            session.add_all((user_message, assistant_message))
        return user_message, assistant_message

    def draft_preview(draft_id: str | None) -> str | None:
        if draft_id is None:
            return None
        with session_factory() as session:
            return session.scalar(
                select(DraftVersion.rendered_text)
                .join(Draft, Draft.id == DraftVersion.draft_id)
                .where(
                    Draft.id == draft_id,
                    DraftVersion.version_number == Draft.current_version_number,
                )
            )

    def serialize_draft(draft: Draft, version: DraftVersion) -> dict[str, Any]:
        raw_parts = version.content.get("parts", [])
        parts = [str(part) for part in raw_parts] if isinstance(raw_parts, list) else []
        status = draft.status
        return {
            "id": draft.id,
            "status": status.value,
            "content_type": draft.content_type.value,
            "text": version.rendered_text,
            "parts": parts,
            "updated_at": _iso(draft.updated_at),
            "fact_check_status": draft.fact_check_status.value,
            "blocking_flags": list(draft.blocking_safety_flags),
            "hash_prefix": str(draft.current_content_hash or "")[:12],
            "can_edit": status in {
                DraftStatus.NEEDS_REVIEW,
                DraftStatus.APPROVED,
                DraftStatus.REJECTED,
                DraftStatus.PUBLISH_FAILED,
            },
            "can_rewrite": status in {
                DraftStatus.NEEDS_REVIEW,
                DraftStatus.APPROVED,
                DraftStatus.REJECTED,
                DraftStatus.PUBLISH_FAILED,
            },
            "can_approve": status is DraftStatus.NEEDS_REVIEW,
            "can_reject": status in {DraftStatus.NEEDS_REVIEW, DraftStatus.APPROVED},
            "can_publish": status is DraftStatus.APPROVED,
        }

    def current_draft_payload(session: Any, draft_id: str) -> dict[str, Any]:
        row = session.execute(
            select(Draft, DraftVersion)
            .join(
                DraftVersion,
                (DraftVersion.draft_id == Draft.id)
                & (DraftVersion.version_number == Draft.current_version_number),
            )
            .where(Draft.id == draft_id, Draft.deleted_at.is_(None))
        ).one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="Draft was not found")
        return serialize_draft(row[0], row[1])

    def draft_source_url(session: Any, draft_id: str) -> str | None:
        row = session.execute(
            select(Draft, DraftVersion)
            .join(
                DraftVersion,
                (DraftVersion.draft_id == Draft.id)
                & (DraftVersion.version_number == Draft.current_version_number),
            )
            .where(Draft.id == draft_id)
        ).one_or_none()
        if row is None:
            return None
        draft, version = row
        source_context = version.generation_metadata.get("source_context", {})
        candidates: list[str] = []
        if isinstance(source_context, Mapping):
            candidates.append(str(source_context.get("source_url") or ""))
        if draft.idea_id:
            stored_url = session.scalar(
                select(SourcePost.url)
                .join(IdeaSource, IdeaSource.source_post_id == SourcePost.id)
                .where(IdeaSource.idea_id == draft.idea_id, SourcePost.url.is_not(None))
                .order_by(IdeaSource.position)
                .limit(1)
            )
            candidates.append(str(stored_url or ""))
        for candidate in candidates:
            parsed = urlparse(candidate.strip())
            host = (parsed.hostname or "").casefold()
            allowed_hosts = {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}
            if parsed.scheme == "https" and host in allowed_hosts:
                return candidate.strip()
        return None

    def draft_action_error(exc: Exception) -> HTTPException:
        if isinstance(exc, DraftNotFoundError):
            return HTTPException(status_code=404, detail="Draft was not found")
        if isinstance(exc, DraftValidationError):
            return HTTPException(status_code=422, detail=str(exc))
        if isinstance(
            exc,
            (
                ApprovalGateError,
                PublicationGateError,
                AmbiguousClaimError,
                DraftStateMachineError,
            ),
        ):
            return HTTPException(status_code=409, detail=str(exc))
        return HTTPException(status_code=500, detail="The draft action failed safely")

    async def run_discovery() -> None:
        async with discovery_lock:
            try:
                outcome = await AutomaticDiscoveryService(settings).run_once(actor="dashboard")
                preview = draft_preview(outcome.draft_id)
                suffix = f"\n\nDraft for review:\n{preview}" if preview else ""
                save_chat_message(
                    "assistant",
                    f"Discovery finished: {outcome.message}{suffix}",
                    command="/discover",
                    metadata={"status": outcome.status, "draft_id": outcome.draft_id},
                )
            except Exception as exc:
                save_chat_message(
                    "assistant",
                    "Discovery stopped safely: " + safe_exception_summary(exc),
                    command="/discover",
                    metadata={"status": "error", "error": type(exc).__name__},
                )

    async def run_manual_generation(url: str) -> None:
        async with manual_generation_lock:
            try:
                configuration = ConfigurationService(settings.config_dir).load()
                if not configuration.sources.sources.manual.enabled:
                    raise RuntimeError("Manual sources are disabled in Settings.")
                reader = (
                    MockXClient()
                    if settings.mock_mode
                    else build_x_read_client(
                        settings,
                        request_pacing=configuration.runtime.request_pacing,
                    )
                )
                with session_factory.begin() as session:
                    collector = IdeaCollector(
                        session,
                        source_configuration=configuration.sources,
                        similarity_threshold=settings.similarity_threshold,
                    )
                    idea_ids = await collector.import_url(url, reader)
                if not idea_ids:
                    raise RuntimeError("The source tweet was not imported.")
                with session_factory.begin() as session:
                    services = build_services(session, settings)
                    draft = await services.drafts.generate_from_idea(
                        idea_ids[0], actor="dashboard_manual_url"
                    )
                    draft_id = draft.id if draft is not None else None
                preview = draft_preview(draft_id)
                if draft_id is None:
                    message = (
                        "The tweet was imported into Activity, but the editorial workflow decided "
                        "that no grounded post should be created."
                    )
                else:
                    message = (
                        "Generation completed from the supplied tweet. "
                        f"Draft ID: {draft_id}. The source tweet remains in Activity."
                        + (f"\n\nDraft for review:\n{preview}" if preview else "")
                    )
                save_chat_message(
                    "assistant",
                    message,
                    command="/new",
                    metadata={"draft_id": draft_id, "source_url": url},
                )
            except Exception as exc:
                save_chat_message(
                    "assistant",
                    "Could not generate from that tweet: " + safe_exception_summary(exc),
                    command="/new",
                    metadata={"error": type(exc).__name__, "source_url": url},
                )

    def schedule(coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    @router.get("/", include_in_schema=False)
    def dashboard_index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @router.get("/login", include_in_schema=False)
    def login_index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "login.html")

    @router.get("/assets/{filename}", include_in_schema=False)
    def dashboard_asset(filename: str) -> FileResponse:
        if filename not in {
            "app.css",
            "app.js",
            "auth.css",
            "auth.js",
            "design.css",
            "favicon.svg",
            "vouch-mark.svg",
        }:
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(_STATIC_DIR / filename)

    @router.get("/api/dashboard/status")
    def dashboard_status() -> dict[str, Any]:
        return status_payload()

    @router.get("/api/activity/feed")
    def activity_feed(
        limit: int = 30,
        before: str | None = None,
        before_id: str | None = None,
    ) -> dict[str, Any]:
        safe_limit = max(1, min(limit, 100))
        items = query_posts(
            before_time=_parse_cursor(before),
            before_id=before_id,
            descending=True,
            limit=safe_limit + 1,
        )
        has_more = len(items) > safe_limit
        page = items[:safe_limit]
        next_cursor = None
        if has_more and page:
            last = page[-1]
            next_cursor = {"before": last["updated_at"], "before_id": last["id"]}
        return {"items": page, "count": len(page), "next_cursor": next_cursor}

    @router.get("/api/activity/stream", include_in_schema=False)
    async def activity_stream(request: Request) -> StreamingResponse:
        cursor_time = _parse_cursor(request.query_params.get("after"))
        cursor_id = request.query_params.get("after_id")

        async def events() -> AsyncIterator[str]:
            nonlocal cursor_time, cursor_id
            yield "retry: 1500\n\n"
            while not await request.is_disconnected():
                items = await asyncio.to_thread(
                    query_posts,
                    after_time=cursor_time,
                    after_id=cursor_id,
                    descending=False,
                    limit=100,
                )
                for item in items:
                    cursor_time = _parse_cursor(item["updated_at"])
                    cursor_id = item["id"]
                    yield "event: source_post\n"
                    yield "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"
                yield ": keep-alive\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/api/dashboard/settings")
    def read_dashboard_settings() -> DashboardSettingsPayload:
        return _dashboard_settings(settings)

    @router.put("/api/dashboard/settings")
    def save_dashboard_settings(payload: DashboardSettingsPayload) -> JSONResponse:
        try:
            with configuration_lock:
                _save_dashboard_settings(settings, payload)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse(
            {
                "saved": True,
                "restart_required": True,
                "message": "Settings were saved. Restart the server for background-task changes.",
            }
        )

    @router.get("/api/dashboard/credentials")
    def read_credential_settings() -> dict[str, Any]:
        return {
            "configured": credential_status(settings),
            "public": {},
            "restart_required": False,
            "values_returned": False,
        }

    @router.put("/api/dashboard/credentials")
    def save_credential_settings(payload: CredentialSettingsPayload) -> dict[str, Any]:
        try:
            update_environment_file(
                settings.data_dir.parent / ".env",
                values=payload.values,
                clear=set(payload.clear),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "saved": True,
            "restart_required": True,
            "values_returned": False,
        }

    @router.get("/api/selected-accounts")
    def selected_accounts() -> dict[str, Any]:
        return _selected_accounts_payload(settings)

    @router.put("/api/selected-accounts")
    def save_selected_accounts_settings(
        payload: SelectedAccountsSettingsPayload,
    ) -> dict[str, Any]:
        with configuration_lock:
            _mutate_selected_accounts(
                settings,
                lambda section: section.update(
                    {
                        "enabled": payload.enabled,
                        "max_posts_per_account": payload.max_posts_per_account,
                    }
                ),
            )
        return _selected_accounts_payload(settings)

    @router.post("/api/selected-accounts", status_code=201)
    def add_selected_account(payload: SelectedAccountCreatePayload) -> dict[str, Any]:
        username = _normalize_username(payload.username)

        def add(section: dict[str, Any]) -> None:
            accounts = section.setdefault("accounts", [])
            if not isinstance(accounts, list):
                raise HTTPException(status_code=500, detail="Invalid selected accounts section")
            if any(
                isinstance(item, Mapping) and str(item.get("username", "")).casefold() == username
                for item in accounts
            ):
                raise HTTPException(status_code=409, detail=f"@{username} is already tracked")
            accounts.append({"username": username, "enabled": True})

        with configuration_lock:
            _mutate_selected_accounts(settings, add)
        return _selected_accounts_payload(settings)

    @router.patch("/api/selected-accounts/{username}")
    def update_selected_account(
        username: str,
        payload: SelectedAccountUpdatePayload,
    ) -> dict[str, Any]:
        normalized = _normalize_username(username)

        def update(section: dict[str, Any]) -> None:
            accounts = section.setdefault("accounts", [])
            if not isinstance(accounts, list):
                raise HTTPException(status_code=500, detail="Invalid selected accounts section")
            for item in accounts:
                if (
                    isinstance(item, dict)
                    and str(item.get("username", "")).casefold() == normalized
                ):
                    item["enabled"] = payload.enabled
                    return
            raise HTTPException(status_code=404, detail=f"@{normalized} is not tracked")

        with configuration_lock:
            _mutate_selected_accounts(settings, update)
        return _selected_accounts_payload(settings)

    @router.delete("/api/selected-accounts/{username}")
    def delete_selected_account(username: str) -> dict[str, Any]:
        normalized = _normalize_username(username)

        def delete(section: dict[str, Any]) -> None:
            accounts = section.setdefault("accounts", [])
            if not isinstance(accounts, list):
                raise HTTPException(status_code=500, detail="Invalid selected accounts section")
            remaining = [
                item
                for item in accounts
                if not (
                    isinstance(item, Mapping)
                    and str(item.get("username", "")).casefold() == normalized
                )
            ]
            if len(remaining) == len(accounts):
                raise HTTPException(status_code=404, detail=f"@{normalized} is not tracked")
            section["accounts"] = remaining

        with configuration_lock:
            _mutate_selected_accounts(settings, delete)
        return _selected_accounts_payload(settings)

    # Retained as a compatibility API for advanced users. The normal UI uses structured forms.
    @router.get("/api/config/files")
    def config_files() -> dict[str, Any]:
        items = []
        for filename in _EDITABLE_CONFIG_FILES:
            path = settings.config_dir / filename
            if path.is_file():
                stat = path.stat()
                items.append(
                    {
                        "name": filename,
                        "size": stat.st_size,
                        "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                    }
                )
        return {"items": items}

    @router.get("/api/config/files/{filename}")
    def read_config_file(filename: str) -> dict[str, Any]:
        path = _config_path(settings, filename)
        stat = path.stat()
        return {
            "name": filename,
            "content": path.read_text(encoding="utf-8"),
            "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        }

    @router.put("/api/config/files/{filename}")
    def save_config_file(filename: str, payload: ConfigFilePayload) -> JSONResponse:
        path = _config_path(settings, filename)
        try:
            parsed = yaml.safe_load(payload.content)
        except yaml.YAMLError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid YAML: {exc}") from exc
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=422, detail="The YAML root must be a mapping")
        if filename == "content_sources.yml":
            sources = parsed.setdefault("sources", {})
            automatic = parsed.setdefault("automatic_discovery", {})
            if isinstance(sources, dict):
                recent = sources.setdefault("recent_search", {})
                if isinstance(recent, dict):
                    recent["enabled"] = False
            if isinstance(automatic, dict):
                automatic.pop("run_on_start", None)
            payload = ConfigFilePayload(
                content=yaml.safe_dump(parsed, allow_unicode=False, sort_keys=False)
            )
        backup = path.with_suffix(path.suffix + ".dashboard-backup")
        temporary = path.with_suffix(path.suffix + ".tmp")
        shutil.copy2(path, backup)
        try:
            temporary.write_text(payload.content, encoding="utf-8")
            temporary.replace(path)
            ConfigurationService(settings.config_dir).load()
        except Exception as exc:
            shutil.copy2(backup, path)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            temporary.unlink(missing_ok=True)
            backup.unlink(missing_ok=True)
        return JSONResponse(
            {
                "saved": True,
                "restart_required": True,
                "message": f"{filename} was saved. Restart the server to apply runtime changes.",
            }
        )

    @router.get("/api/chat/history")
    def chat_history(limit: int = 100) -> dict[str, Any]:
        safe_limit = max(1, min(limit, 250))
        with session_factory() as session:
            rows = list(
                session.scalars(
                    select(DashboardChatMessage)
                    .order_by(
                        DashboardChatMessage.created_at.desc(),
                        DashboardChatMessage.id.desc(),
                    )
                    .limit(safe_limit)
                ).all()
            )
        rows.reverse()
        return {"items": [serialize_chat_message(row) for row in rows]}

    @router.get("/api/drafts/recent")
    async def recent_drafts(limit: int = 5) -> dict[str, Any]:
        bounded_limit = min(max(limit, 1), 20)
        with session_factory() as session:
            rows = session.execute(
                select(Draft, DraftVersion)
                .join(
                    DraftVersion,
                    (DraftVersion.draft_id == Draft.id)
                    & (DraftVersion.version_number == Draft.current_version_number),
                )
                .where(Draft.deleted_at.is_(None))
                .order_by(Draft.updated_at.desc())
                .limit(bounded_limit)
            ).all()
            return {
                "items": [serialize_draft(draft, version) for draft, version in rows]
            }

    @router.put("/api/drafts/{draft_id}")
    def edit_draft(draft_id: str, payload: DraftEditPayload) -> dict[str, Any]:
        with session_factory.begin() as session:
            services = build_services(session, settings)
            try:
                draft = services.drafts.get(draft_id)
                parts = (
                    tuple(part.strip() for part in payload.text.split("\n\n---\n\n"))
                    if draft.content_type is ContentType.THREAD
                    else (payload.text.strip(),)
                )
                services.drafts.edit_current(draft_id, parts=parts, actor="dashboard")
                result = current_draft_payload(session, draft_id)
            except Exception as exc:
                raise draft_action_error(exc) from exc
        return {"draft": result, "message": "Edits saved. Approval was reset."}

    @router.post("/api/drafts/{draft_id}/rewrite")
    async def rewrite_draft(draft_id: str, payload: DraftRewritePayload) -> dict[str, Any]:
        with session_factory.begin() as session:
            services = build_services(session, settings)
            try:
                await services.drafts.regenerate(
                    draft_id,
                    feedback=payload.feedback.strip() or None,
                    actor="dashboard",
                )
                result = current_draft_payload(session, draft_id)
            except Exception as exc:
                raise draft_action_error(exc) from exc
        return {"draft": result, "message": "A new draft version was generated."}

    @router.post("/api/drafts/{draft_id}/approve")
    def approve_draft(draft_id: str, payload: DraftApprovalPayload) -> dict[str, Any]:
        with session_factory.begin() as session:
            services = build_services(session, settings)
            try:
                services.approvals.approve(
                    draft_id,
                    actor="dashboard",
                    facts_confirmed=payload.facts_confirmed,
                )
                result = current_draft_payload(session, draft_id)
            except Exception as exc:
                raise draft_action_error(exc) from exc
        return {"draft": result, "message": "Draft approved for manual publication."}

    @router.post("/api/drafts/{draft_id}/reject")
    def reject_draft(draft_id: str) -> dict[str, Any]:
        with session_factory.begin() as session:
            services = build_services(session, settings)
            try:
                services.drafts.quarantine(draft_id, actor="dashboard")
            except Exception as exc:
                raise draft_action_error(exc) from exc
        return {"deleted": True, "message": "Draft rejected and moved to local quarantine."}

    @router.post("/api/drafts/{draft_id}/publish-preview")
    async def publish_preview(draft_id: str) -> dict[str, Any]:
        with session_factory() as session:
            services = build_services(session, settings)
            try:
                preview = await services.publishing.preview(draft_id)
            except Exception as exc:
                if isinstance(exc, PublicationGateError) and "requires an X quote post" in str(exc):
                    source_url = draft_source_url(session, draft_id)
                    if source_url is not None:
                        save_chat_message(
                            "assistant",
                            (
                                "This draft needs the original X post for context. "
                                "Open the source tweet here and publish it as a quote post: "
                                f"{source_url}"
                            ),
                            command="/publish",
                            metadata={
                                "status": "quote_context_required",
                                "draft_id": draft_id,
                                "source_url": source_url,
                            },
                        )
                elif isinstance(exc, PublishingAccountMismatchError):
                    save_chat_message(
                        "assistant",
                        str(exc),
                        command="/publish",
                        metadata={
                            "status": "publishing_account_mismatch",
                            "draft_id": draft_id,
                            "expected_account_id": exc.expected_id,
                            "authenticated_account_id": exc.actual_id,
                            "authenticated_username": exc.username,
                        },
                    )
                raise draft_action_error(exc) from exc
        return {
            "draft_id": preview.draft_id,
            "confirmation_phrase": preview.confirmation_phrase,
            "parts": list(preview.parts),
            "weighted_lengths": list(preview.weighted_lengths),
            "facts": preview.facts,
            "action": preview.action,
            "account_username": preview.account_username,
            "estimated_cost_usd": (
                str(preview.estimated_cost_usd)
                if preview.estimated_cost_usd is not None
                else None
            ),
        }

    @router.post("/api/drafts/{draft_id}/publish")
    async def publish_draft(draft_id: str, payload: DraftPublishPayload) -> dict[str, Any]:
        with session_factory() as session:
            services = build_services(session, settings)
            try:
                publications = await services.publishing.publish(
                    draft_id,
                    confirmation_phrase=payload.confirmation_phrase,
                    actor="dashboard",
                )
                session.commit()
            except Exception as exc:
                session.rollback()
                raise draft_action_error(exc) from exc
        return {
            "published": True,
            "items": [
                {
                    "position": item.thread_position,
                    "status": item.status.value,
                    "x_post_id": item.x_post_id,
                }
                for item in publications
            ],
        }

    @router.post("/api/chat/command")
    async def chat_command(payload: CommandPayload) -> dict[str, Any]:
        text = payload.text.strip()
        parts = text.split(maxsplit=1)
        command = parts[0].casefold()
        argument = parts[1].strip() if len(parts) > 1 else ""
        action: str | None = None
        extra: dict[str, Any] = {}
        if command in {"/help", "help"}:
            message = (
                "Available commands: /status, /discover, /new <X post URL>, /feed, "
                "/settings, /accounts, /help. Commands are allow-listed and never executed "
                "as shell code."
            )
        elif command in {"/status", "status"}:
            status = status_payload()
            message = (
                f"Bot {status['version']} is online in {status['mode']} mode. "
                f"Stored source posts: {status['total_posts']}. "
                f"Automatic discovery: {'on' if status['automatic_discovery'] else 'off'}. "
                f"Selected accounts enabled: {status['selected_accounts']}. "
                "Recent Search: disabled."
            )
            extra["status"] = status
        elif command in {"/feed", "feed"}:
            message = "Activity feed is open on the right."
            action = "activity"
        elif command in {"/settings", "settings", "/config", "config"}:
            message = "Opening structured settings."
            action = "settings"
        elif command in {"/accounts", "accounts"}:
            message = "Opening Selected accounts."
            action = "accounts"
        elif command in {"/discover", "discover"}:
            if discovery_lock.locked():
                message = "Discovery is already running."
            else:
                schedule(run_discovery())
                message = (
                    "Discovery started. Home timeline, official Trends, and enabled Selected "
                    "accounts will be read. Recent Search is not used."
                )
        elif command == "/new":
            if not argument:
                message = "Add an X post URL. Example: /new https://x.com/user/status/1234567890"
                extra["error"] = "missing_url"
            elif manual_generation_lock.locked():
                message = "Another /new generation is already running."
            else:
                # Validate before scheduling, while IdeaCollector performs the authoritative check.
                parsed = urlparse(argument)
                if parsed.scheme not in {"http", "https"} or parsed.netloc.casefold() not in {
                    "x.com",
                    "www.x.com",
                    "twitter.com",
                    "www.twitter.com",
                }:
                    message = "Use a direct x.com or twitter.com post URL after /new."
                    extra["error"] = "invalid_url"
                else:
                    schedule(run_manual_generation(argument))
                    message = (
                        "The source tweet is being imported and generation has started. "
                        "The final result will appear here, and the tweet will remain in Activity."
                    )
                    action = "activity"
        else:
            message = "Unknown command. Use /help to see the safe command list."
            extra["error"] = "unknown_command"

        user_message, assistant_message = save_chat_pair(
            text,
            message,
            command=command,
            metadata={"action": action} if action else {},
        )
        response: dict[str, Any] = {
            "message": message,
            "action": action,
            "messages": [
                serialize_chat_message(user_message),
                serialize_chat_message(assistant_message),
            ],
        }
        response.update(extra)
        return response

    return router
