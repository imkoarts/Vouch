"""Safe X Activity subscription planning, event ingestion, and optional reply drafting."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import ConfigurationError, Settings
from app.domain.enums import ContentSourceKind, ContentType, ReplyCandidateStatus
from app.models import Idea, OptOutUser, ReplyCandidate, SourcePost, XActivityEvent
from app.models.base import utc_now
from app.repositories.content import ContentRepository
from app.schemas.configuration import (
    ContentSourcesConfiguration,
    XActivityRuntimeConfiguration,
    XActivitySubscriptionConfiguration,
)
from app.services.draft_service import DraftService
from app.services.idea_collector import IdeaCollector
from app.x_api.activity import (
    XActivityApiError,
    XActivityClient,
    XActivitySubscription,
    subscription_keys,
)

_PRIVATE_EVENT_TYPES = frozenset({"post.mention.create"})
_SUPPORTED_EVENT_TYPES = frozenset({"post.create", "post.delete", "post.mention.create"})


@dataclass(frozen=True, slots=True)
class XActivityPlan:
    desired: tuple[XActivitySubscription, ...]
    existing: tuple[XActivitySubscription, ...]
    missing: tuple[XActivitySubscription, ...]
    unmanaged_existing: tuple[XActivitySubscription, ...]


@dataclass(frozen=True, slots=True)
class XActivityProcessingResult:
    event_uuid: str
    event_type: str
    duplicate: bool
    status: str
    source_post_id: str | None = None
    idea_id: str | None = None
    reply_candidate_id: str | None = None
    draft_id: str | None = None
    notification: str | None = None


def activity_requires_private_auth(configuration: XActivityRuntimeConfiguration) -> bool:
    return any(item.event_type in _PRIVATE_EVENT_TYPES for item in configuration.subscriptions)


def resolve_activity_user_id(value: str, settings: Settings) -> str:
    if value != "self":
        return value
    if not settings.x_user_id:
        raise ConfigurationError("X_USER_ID is required for x_activity user_id=self")
    return settings.x_user_id


def desired_activity_subscriptions(
    settings: Settings,
    configuration: XActivityRuntimeConfiguration,
) -> tuple[XActivitySubscription, ...]:
    desired: list[XActivitySubscription] = []
    for item in configuration.subscriptions:
        desired.append(
            XActivitySubscription(
                event_type=item.event_type,
                user_id=resolve_activity_user_id(item.user_id, settings),
                tag=item.tag,
            )
        )
    return tuple(desired)


def build_activity_client(
    settings: Settings,
    configuration: XActivityRuntimeConfiguration,
) -> XActivityClient:
    private = activity_requires_private_auth(configuration)
    missing = settings.missing_x_activity_credentials(private=private)
    if missing:
        raise ConfigurationError("X Activity credentials are incomplete: " + ", ".join(missing))
    token = settings.x_activity_token(private=private)
    if token is None:
        raise ConfigurationError("No usable X Activity bearer credential is configured")
    return XActivityClient(
        token=token,
        base_url=settings.x_api_base_url,
        timeout_seconds=settings.request_timeout_seconds,
        proxy_url=settings.outbound_proxy_url,
    )


class XActivitySubscriptionManager:
    """Plan remote subscriptions and create missing entries only after explicit confirmation."""

    def __init__(
        self,
        client: XActivityClient,
        *,
        settings: Settings,
        configuration: XActivityRuntimeConfiguration,
    ) -> None:
        self.client = client
        self.settings = settings
        self.configuration = configuration

    async def plan(self) -> XActivityPlan:
        desired = desired_activity_subscriptions(self.settings, self.configuration)
        existing = await self.client.list_subscriptions()
        desired_keys = subscription_keys(desired)
        existing_keys = subscription_keys(existing)
        missing = tuple(
            item for item in desired if (item.event_type, item.user_id) not in existing_keys
        )
        unmanaged = tuple(
            item for item in existing if (item.event_type, item.user_id) not in desired_keys
        )
        return XActivityPlan(
            desired=desired,
            existing=existing,
            missing=missing,
            unmanaged_existing=unmanaged,
        )

    async def create_missing(
        self, *, confirmation_phrase: str
    ) -> tuple[XActivitySubscription, ...]:
        if confirmation_phrase != "SYNC X ACTIVITY SUBSCRIPTIONS":
            raise PermissionError("Exact confirmation phrase is required")
        plan = await self.plan()
        created: list[XActivitySubscription] = []
        for subscription in plan.missing:
            created.append(await self.client.create_subscription(subscription))
        return tuple(created)


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _list_of_mappings(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _iso_datetime(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC).isoformat()
    except ValueError:
        try:
            return parsedate_to_datetime(text).astimezone(UTC).isoformat()
        except (TypeError, ValueError):
            return None


def _extract_post(data: Mapping[str, Any]) -> dict[str, Any] | None:
    includes = _mapping(data.get("includes"))
    tweets = _list_of_mappings(includes.get("tweets")) if includes is not None else ()
    posts = _list_of_mappings(includes.get("posts")) if includes is not None else ()
    payload = _mapping(data.get("payload"))
    candidates = [*tweets, *posts]
    if payload is not None:
        embedded = _mapping(payload.get("tweet")) or _mapping(payload.get("post"))
        if embedded is not None:
            candidates.append(embedded)
        if payload.get("id") and payload.get("text"):
            candidates.append(payload)
    if not candidates:
        return None
    post = dict(candidates[0])
    users = _list_of_mappings(includes.get("users")) if includes is not None else ()
    author_id = str(post.get("author_id") or "")
    matching_user = next((user for user in users if str(user.get("id") or "") == author_id), None)
    if matching_user is not None and not post.get("username"):
        post["username"] = matching_user.get("username")
    created_at = _iso_datetime(post.get("created_at"))
    if created_at is not None:
        post["created_at"] = created_at
    post["raw_json"] = dict(data)
    return post if str(post.get("id") or "") and str(post.get("text") or "").strip() else None


def _delete_post_id(data: Mapping[str, Any]) -> str | None:
    payload = _mapping(data.get("payload"))
    if payload is not None:
        for key in ("id", "post_id", "tweet_id"):
            value = payload.get(key)
            if value is not None and str(value):
                return str(value)
    post = _extract_post(data)
    return str(post["id"]) if post is not None else None


class XActivityProcessor:
    """Persist each event once and convert supported post events into reviewable local data."""

    def __init__(
        self,
        session: Session,
        *,
        settings: Settings,
        runtime: XActivityRuntimeConfiguration,
        sources: ContentSourcesConfiguration,
        drafts: DraftService,
    ) -> None:
        self.session = session
        self.settings = settings
        self.runtime = runtime
        self.sources = sources
        self.drafts = drafts
        self.repository = ContentRepository(session)

    def _configured_subscription(
        self, event_type: str, filtered_user_id: str | None
    ) -> XActivitySubscriptionConfiguration | None:
        for item in self.runtime.subscriptions:
            resolved = resolve_activity_user_id(item.user_id, self.settings)
            if item.event_type == event_type and resolved == (filtered_user_id or ""):
                return item
        return None

    def _existing_event(self, event_uuid: str) -> XActivityEvent | None:
        return self.session.scalar(
            select(XActivityEvent).where(XActivityEvent.event_uuid == event_uuid)
        )

    def _create_reply_candidate(self, source_post: SourcePost) -> ReplyCandidate:
        existing = self.session.scalar(
            select(ReplyCandidate).where(ReplyCandidate.source_post_id == source_post.id)
        )
        if existing is not None:
            return existing
        author_id = source_post.author_id or ""
        opted_out = bool(
            author_id
            and self.session.scalar(select(OptOutUser.id).where(OptOutUser.x_user_id == author_id))
        )
        if opted_out:
            status = ReplyCandidateStatus.OPTED_OUT
            reason = "author opted out"
        elif not author_id:
            status = ReplyCandidateStatus.INELIGIBLE
            reason = "source post has no author ID"
        else:
            status = ReplyCandidateStatus.ELIGIBLE
            reason = "direct X mention received through X Activity"
        candidate = ReplyCandidate(
            source_post_id=source_post.id,
            conversation_id=source_post.conversation_id,
            author_id=author_id or "unknown",
            username=source_post.username,
            status=status,
            direct_mention=True,
            reply_to_owned_post=False,
            allowlisted=False,
            eligibility_reason=reason,
            conversation_context={"activity_event": True},
        )
        self.session.add(candidate)
        self.session.flush()
        return candidate

    def _store_post_snapshot(
        self,
        post: Mapping[str, Any],
        *,
        event_type: str,
        filtered_user_id: str,
        create_idea: bool,
    ) -> tuple[SourcePost, Idea | None]:
        """Persist a provider snapshot and optionally rank it as an idea.

        X Activity is also useful as a durable event feed.  A configured
        subscription may therefore retain the source post without promoting it
        into the content-idea queue.  Reply drafting still needs an idea, so the
        caller can explicitly request one even when ``ingest_as_idea`` is false.
        """

        # Activity events have their own provenance and enablement.  In particular,
        # mention ingestion must never depend on the unrelated Recent Search toggle.
        source_kind = ContentSourceKind.X_ACTIVITY
        if create_idea:
            collector = IdeaCollector(
                self.session,
                source_configuration=self.sources,
                similarity_threshold=self.settings.similarity_threshold,
            )
            idea_ids = collector.ingest(
                [post],
                source_kind=source_kind,
                source_name=f"X Activity {event_type}",
                source_identifier=filtered_user_id,
                persistence_kind=ContentSourceKind.SELECTED_ACCOUNT,
                source_metadata={
                    "ingestion_channel": ContentSourceKind.X_ACTIVITY.value,
                    "event_type": event_type,
                },
            )
            source_post = self.repository.find_by_x_post_id(str(post["id"]))
            if source_post is None:  # pragma: no cover - repository invariant
                raise XActivityApiError("X Activity source post was not persisted")
            idea = self.session.get(Idea, idea_ids[0]) if idea_ids else None
            return source_post, idea

        configured_source = self.sources.source_for_kind(source_kind)
        source = self.repository.get_or_create_source(
            kind=ContentSourceKind.SELECTED_ACCOUNT,
            name=f"X Activity {event_type}",
            identifier=filtered_user_id,
            weight=configured_source.weight,
            configuration={
                "ingestion_channel": ContentSourceKind.X_ACTIVITY.value,
                "event_type": event_type,
            },
        )
        source_post, _ = self.repository.import_source_post(payload=post, source=source)
        return source_post, None

    async def process(self, envelope: Mapping[str, Any]) -> XActivityProcessingResult:
        data = _mapping(envelope.get("data"))
        if data is None:
            raise XActivityApiError("X Activity event is missing data")
        event_uuid = str(data.get("event_uuid") or "").strip()
        event_type = str(data.get("event_type") or "").strip()
        if not event_uuid or event_type not in _SUPPORTED_EVENT_TYPES:
            raise XActivityApiError("X Activity event type or UUID is unsupported")
        filter_value = _mapping(data.get("filter"))
        filtered_user_id = (
            str(filter_value.get("user_id") or "") if filter_value is not None else ""
        )
        existing = self._existing_event(event_uuid)
        if existing is not None:
            return XActivityProcessingResult(
                event_uuid=event_uuid,
                event_type=event_type,
                duplicate=True,
                status=existing.status,
                source_post_id=existing.source_post_id,
                idea_id=existing.idea_id,
                reply_candidate_id=existing.reply_candidate_id,
            )
        record = XActivityEvent(
            event_uuid=event_uuid,
            event_type=event_type,
            filtered_user_id=filtered_user_id or None,
            tag=str(data.get("tag")) if data.get("tag") is not None else None,
            status="received",
            raw_event=dict(envelope),
        )
        self.session.add(record)
        self.session.flush()
        configured = self._configured_subscription(event_type, filtered_user_id)
        if configured is None:
            record.status = "ignored"
            record.processed_at = utc_now()
            record.result_metadata = {"reason": "event is not in local desired subscriptions"}
            return XActivityProcessingResult(
                event_uuid=event_uuid,
                event_type=event_type,
                duplicate=False,
                status=record.status,
                notification=f"X Activity: ignored unmanaged event {event_type}.",
            )

        if event_type == "post.delete":
            x_post_id = _delete_post_id(data)
            source_post = self.repository.find_by_x_post_id(x_post_id) if x_post_id else None
            if source_post is not None:
                source_post.text = "[deleted on X]"
                source_post.public_metrics = {}
                source_post.media_metadata = {}
                source_post.raw_json = {"deleted": True, "x_post_id": x_post_id}
            record.status = "processed"
            record.source_post_id = source_post.id if source_post is not None else None
            record.processed_at = utc_now()
            record.result_metadata = {
                "x_post_id": x_post_id,
                "local_source_found": bool(source_post),
            }
            return XActivityProcessingResult(
                event_uuid=event_uuid,
                event_type=event_type,
                duplicate=False,
                status=record.status,
                source_post_id=record.source_post_id,
                notification=f"X Activity: post.delete for {x_post_id or 'unknown'}.",
            )

        post = _extract_post(data)
        if post is None:
            record.status = "ignored"
            record.processed_at = utc_now()
            record.result_metadata = {"reason": "event did not include a complete post"}
            return XActivityProcessingResult(
                event_uuid=event_uuid,
                event_type=event_type,
                duplicate=False,
                status=record.status,
                notification=f"X Activity: {event_type} without full post text.",
            )

        source_post, idea = self._store_post_snapshot(
            post,
            event_type=event_type,
            filtered_user_id=filtered_user_id,
            create_idea=(configured.ingest_as_idea or configured.generate_reply_draft),
        )
        candidate: ReplyCandidate | None = None
        draft_id: str | None = None
        if event_type == "post.mention.create" and source_post is not None:
            candidate = self._create_reply_candidate(source_post)
            if idea is not None:
                idea.recommended_format = ContentType.REPLY
                idea.title = f"Reply to @{source_post.username or 'unknown'}"
                idea.explanation = {
                    **idea.explanation,
                    "activity_event": "direct mention",
                    "reply_requires_manual_review": True,
                }
            if (
                configured.generate_reply_draft
                and candidate.status is ReplyCandidateStatus.ELIGIBLE
                and idea is not None
            ):
                draft = await self.drafts.generate_from_idea(idea.id, actor="x_activity")
                if draft is not None:
                    draft.reply_candidate_id = candidate.id
                    candidate.status = ReplyCandidateStatus.DRAFT_CREATED
                    draft_id = draft.id

        record.status = "processed"
        record.source_post_id = source_post.id if source_post is not None else None
        record.idea_id = idea.id if idea is not None else None
        record.reply_candidate_id = candidate.id if candidate is not None else None
        record.processed_at = utc_now()
        record.result_metadata = {
            "draft_id": draft_id,
            "ingest_as_idea": configured.ingest_as_idea,
            "generate_reply_draft": configured.generate_reply_draft,
        }
        username = str(post.get("username") or "unknown")
        text = str(post.get("text") or "")
        notification = (
            f"X Activity — {event_type}\n@{username}:\n{text}"
            if self.runtime.notify_telegram and configured.notify_telegram
            else None
        )
        return XActivityProcessingResult(
            event_uuid=event_uuid,
            event_type=event_type,
            duplicate=False,
            status=record.status,
            source_post_id=record.source_post_id,
            idea_id=record.idea_id,
            reply_candidate_id=record.reply_candidate_id,
            draft_id=draft_id,
            notification=notification,
        )


class XActivityRunner:
    """Reconnect a persistent stream and dispatch events through an injected handler."""

    def __init__(
        self,
        client: XActivityClient,
        *,
        configuration: XActivityRuntimeConfiguration,
        handler: Callable[[Mapping[str, Any]], Awaitable[None]],
        error_handler: Callable[[Exception], Awaitable[None]] | None = None,
    ) -> None:
        self.client = client
        self.configuration = configuration
        self.handler = handler
        self.error_handler = error_handler

    async def run_forever(self, stop_event: asyncio.Event | None = None) -> None:
        delay = self.configuration.reconnect_initial_seconds
        while stop_event is None or not stop_event.is_set():
            try:
                async for event in self.client.stream_events(
                    backfill_minutes=self.configuration.backfill_minutes,
                    stop_event=stop_event,
                ):
                    await self.handler(event)
                    delay = self.configuration.reconnect_initial_seconds
                if stop_event is not None and stop_event.is_set():
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self.error_handler is not None:
                    await self.error_handler(exc)
                if stop_event is not None:
                    with suppress(TimeoutError):
                        await asyncio.wait_for(stop_event.wait(), timeout=delay)
                else:
                    await asyncio.sleep(delay)
                delay = min(delay * 2, self.configuration.reconnect_max_seconds)
