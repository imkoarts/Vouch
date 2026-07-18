"""X Activity stays optional, deduplicated, and review-only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.container import build_mock_services
from app.database import build_engine
from app.models import Base, Idea, ReplyCandidate, SourcePost, XActivityEvent
from app.models.enums import DraftStatus
from app.schemas.configuration import (
    XActivityRuntimeConfiguration,
    XActivitySubscriptionConfiguration,
)
from app.services.configuration import ConfigurationService
from app.services.x_activity import (
    XActivityProcessor,
    XActivitySubscriptionManager,
    activity_requires_private_auth,
    desired_activity_subscriptions,
)
from app.x_api.activity import XActivityClient, XActivitySubscription

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "app_env": "test",
        "mock_mode": True,
        "database_url": f"sqlite:///{(tmp_path / 'activity.db').as_posix()}",
        "data_dir": tmp_path / "data",
        "drafts_dir": tmp_path / "drafts",
        "config_dir": PROJECT_ROOT / "config",
        "x_user_id": "123456789",
    }
    values.update(overrides)
    return Settings(**values)


def _runtime(*subscriptions: XActivitySubscriptionConfiguration) -> XActivityRuntimeConfiguration:
    return XActivityRuntimeConfiguration(
        enabled=True,
        subscriptions=subscriptions,
    )


def _envelope(
    *,
    event_uuid: str,
    event_type: str,
    filtered_user_id: str = "123456789",
    post_id: str = "900000000000000099",
    author_id: str = "700000000000000099",
    text: str = "Synthetic activity event for a manual review workflow.",
) -> dict[str, Any]:
    return {
        "data": {
            "event_uuid": event_uuid,
            "event_type": event_type,
            "filter": {"user_id": filtered_user_id},
            "tag": "test-subscription",
            "payload": {
                "id": post_id,
                "text": text,
                "author_id": author_id,
                "username": "synthetic_author",
                "conversation_id": post_id,
                "created_at": "2026-07-11T12:00:00Z",
                "public_metrics": {"like_count": 1},
            },
        }
    }


def _processor(
    tmp_path: Path,
    runtime: XActivityRuntimeConfiguration,
) -> tuple[Session, XActivityProcessor]:
    settings = _settings(tmp_path)
    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    services = build_mock_services(session, settings)
    processor = XActivityProcessor(
        session,
        settings=settings,
        runtime=runtime,
        sources=services.configuration.sources,
        drafts=services.drafts,
    )
    return session, processor


def _http_client_factory(handler: Any) -> Any:
    transport = httpx.MockTransport(handler)

    def factory(*, streaming: bool = False) -> httpx.AsyncClient:
        del streaming
        return httpx.AsyncClient(transport=transport)

    return factory


def test_checked_in_activity_configuration_is_disabled_and_private() -> None:
    activity = ConfigurationService(PROJECT_ROOT / "config").load_runtime().x_activity

    assert activity.enabled is False
    assert activity.autostart is False
    assert activity_requires_private_auth(activity) is True
    assert activity.subscriptions[0].event_type == "post.mention.create"
    assert activity.subscriptions[0].generate_reply_draft is False


def test_activity_credentials_distinguish_public_and_private_events() -> None:
    app_only = Settings(
        _env_file=None,
        mock_mode=False,
        x_auth_mode="oauth1",
        x_bearer_token="synthetic-app-token",
    )
    private_user = Settings(
        _env_file=None,
        mock_mode=False,
        x_auth_mode="oauth2",
        x_access_token="synthetic-user-token",
        x_oauth2_scopes="tweet.read users.read",
    )
    oauth1_user = Settings(
        _env_file=None,
        mock_mode=False,
        x_auth_mode="oauth1",
        x_consumer_key="consumer",
        x_consumer_secret="consumer-secret",
        x_access_token="user-token",
        x_access_token_secret="token-secret",
    )

    assert app_only.missing_x_activity_credentials(private=False) == []
    assert app_only.missing_x_activity_credentials(private=True)
    assert oauth1_user.missing_x_activity_credentials(private=False)
    assert private_user.missing_x_activity_credentials(private=True) == []
    assert private_user.x_activity_token(private=True) == "synthetic-user-token"


def test_desired_subscriptions_resolve_self_without_network(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    runtime = _runtime(
        XActivitySubscriptionConfiguration(
            event_type="post.create",
            user_id="self",
            tag="own-posts",
        )
    )

    desired = desired_activity_subscriptions(settings, runtime)

    assert desired == (
        XActivitySubscription(
            event_type="post.create",
            user_id="123456789",
            tag="own-posts",
        ),
    )


@pytest.mark.asyncio
async def test_activity_client_lists_creates_and_streams_events() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer synthetic-token"
        if request.method == "GET" and request.url.path == "/2/activity/subscriptions":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "subscription_id": "sub-1",
                            "event_type": "post.create",
                            "filter": {"user_id": "123456789"},
                            "tag": "own-posts",
                        }
                    ]
                },
            )
        if request.method == "POST" and request.url.path == "/2/activity/subscriptions":
            body = json.loads(request.content)
            assert body == {
                "event_type": "post.mention.create",
                "filter": {"user_id": "123456789"},
                "tag": "mentions",
            }
            return httpx.Response(
                200,
                json={
                    "data": {
                        "subscription": {"subscription_id": "sub-2", **body},
                        "total_subscriptions_for_instance_id": 2,
                    }
                },
            )
        if request.method == "GET" and request.url.path == "/2/activity/stream":
            assert request.url.params["backfill_minutes"] == "1"
            return httpx.Response(
                200,
                content=(b'\n{"data":{"event_uuid":"event-1","event_type":"post.create"}}\n'),
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    client = XActivityClient(
        token="synthetic-token",
        client_factory=_http_client_factory(handler),
    )

    existing = await client.list_subscriptions()
    created = await client.create_subscription(
        XActivitySubscription(
            event_type="post.mention.create",
            user_id="123456789",
            tag="mentions",
        )
    )
    streamed = [event async for event in client.stream_events(backfill_minutes=1)]

    assert existing[0].subscription_id == "sub-1"
    assert created.subscription_id == "sub-2"
    assert streamed[0]["data"]["event_uuid"] == "event-1"
    assert [request.url.path for request in requests] == [
        "/2/activity/subscriptions",
        "/2/activity/subscriptions",
        "/2/activity/stream",
    ]


@pytest.mark.asyncio
async def test_activity_client_paginates_subscription_listing() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.params["max_results"] == "1000"
        token = request.url.params.get("pagination_token")
        if token is None:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "subscription_id": "sub-1",
                            "event_type": "post.create",
                            "filter": {"user_id": "123456789"},
                            "tag": "own-posts",
                        }
                    ],
                    "meta": {"next_token": "page-2"},
                },
            )
        assert token == "page-2"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "subscription_id": "sub-2",
                        "event_type": "post.delete",
                        "filter": {"user_id": "123456789"},
                        "tag": "own-deletes",
                    }
                ],
                "meta": {},
            },
        )

    client = XActivityClient(
        token="synthetic-token",
        client_factory=_http_client_factory(handler),
    )

    subscriptions = await client.list_subscriptions()

    assert [item.subscription_id for item in subscriptions] == ["sub-1", "sub-2"]
    assert len(requests) == 2


class _SubscriptionClient:
    def __init__(self) -> None:
        self.created: list[XActivitySubscription] = []

    async def list_subscriptions(self) -> tuple[XActivitySubscription, ...]:
        return (
            XActivitySubscription(
                event_type="post.create",
                user_id="999",
                tag="unmanaged",
                subscription_id="remote-1",
            ),
        )

    async def create_subscription(
        self, subscription: XActivitySubscription
    ) -> XActivitySubscription:
        self.created.append(subscription)
        return XActivitySubscription(
            event_type=subscription.event_type,
            user_id=subscription.user_id,
            tag=subscription.tag,
            subscription_id="created-1",
        )


@pytest.mark.asyncio
async def test_subscription_sync_is_explicit_and_never_deletes_unmanaged_entries(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    runtime = _runtime(
        XActivitySubscriptionConfiguration(
            event_type="post.create",
            user_id="self",
            tag="own-posts",
        )
    )
    client = _SubscriptionClient()
    manager = XActivitySubscriptionManager(  # type: ignore[arg-type]
        client,
        settings=settings,
        configuration=runtime,
    )

    plan = await manager.plan()
    assert len(plan.missing) == 1
    assert len(plan.unmanaged_existing) == 1
    with pytest.raises(PermissionError, match="Exact confirmation"):
        await manager.create_missing(confirmation_phrase="yes")

    created = await manager.create_missing(confirmation_phrase="SYNC X ACTIVITY SUBSCRIPTIONS")

    assert created[0].subscription_id == "created-1"
    assert client.created == list(plan.missing)


@pytest.mark.asyncio
async def test_post_event_can_be_stored_without_creating_an_idea(tmp_path: Path) -> None:
    runtime = _runtime(
        XActivitySubscriptionConfiguration(
            event_type="post.create",
            user_id="self",
            ingest_as_idea=False,
            notify_telegram=False,
        )
    )
    session, processor = _processor(tmp_path, runtime)
    try:
        result = await processor.process(
            _envelope(event_uuid="event-store-only", event_type="post.create")
        )
        session.commit()

        assert result.status == "processed"
        assert result.source_post_id is not None
        assert result.idea_id is None
        assert session.scalar(select(Idea)) is None
        assert session.scalar(select(SourcePost)) is not None
    finally:
        session.close()


@pytest.mark.asyncio
async def test_activity_event_is_deduplicated_and_mention_stays_manual_review_only(
    tmp_path: Path,
) -> None:
    runtime = _runtime(
        XActivitySubscriptionConfiguration(
            event_type="post.mention.create",
            user_id="self",
            ingest_as_idea=True,
            generate_reply_draft=False,
        )
    )
    session, processor = _processor(tmp_path, runtime)
    try:
        envelope = _envelope(event_uuid="event-mention", event_type="post.mention.create")
        first = await processor.process(envelope)
        duplicate = await processor.process(envelope)
        session.commit()

        assert first.status == "processed"
        assert first.reply_candidate_id is not None
        assert first.draft_id is None
        assert duplicate.duplicate is True
        assert len(session.scalars(select(XActivityEvent)).all()) == 1
        assert len(session.scalars(select(ReplyCandidate)).all()) == 1
    finally:
        session.close()


@pytest.mark.asyncio
async def test_mention_reply_draft_is_generated_but_never_published(tmp_path: Path) -> None:
    runtime = _runtime(
        XActivitySubscriptionConfiguration(
            event_type="post.mention.create",
            user_id="self",
            ingest_as_idea=False,
            generate_reply_draft=True,
        )
    )
    session, processor = _processor(tmp_path, runtime)
    try:
        result = await processor.process(
            _envelope(event_uuid="event-reply-draft", event_type="post.mention.create")
        )
        session.commit()

        candidate = session.get(ReplyCandidate, result.reply_candidate_id)
        assert result.draft_id is not None
        assert candidate is not None
        assert candidate.draft is not None
        assert candidate.draft.status is DraftStatus.NEEDS_REVIEW
        assert candidate.draft.publications == []
        source = session.get(SourcePost, result.source_post_id)
        assert source is not None
        assert (
            source.occurrences[0].content_source.configuration["ingestion_channel"] == "x_activity"
        )
        assert (
            ConfigurationService(PROJECT_ROOT / "config")
            .load_sources()
            .sources.recent_search.enabled
            is False
        )
    finally:
        session.close()


@pytest.mark.asyncio
async def test_delete_event_clears_retained_provider_snapshot(tmp_path: Path) -> None:
    runtime = _runtime(
        XActivitySubscriptionConfiguration(
            event_type="post.create",
            user_id="self",
            ingest_as_idea=False,
        ),
        XActivitySubscriptionConfiguration(
            event_type="post.delete",
            user_id="self",
            ingest_as_idea=False,
        ),
    )
    session, processor = _processor(tmp_path, runtime)
    try:
        await processor.process(_envelope(event_uuid="event-create", event_type="post.create"))
        deleted = await processor.process(
            _envelope(event_uuid="event-delete", event_type="post.delete")
        )
        session.commit()

        source = session.get(SourcePost, deleted.source_post_id)
        assert source is not None
        assert source.text == "[deleted on X]"
        assert source.raw_json["deleted"] is True
    finally:
        session.close()
