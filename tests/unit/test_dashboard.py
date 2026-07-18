from __future__ import annotations

import hashlib
import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import Settings
from app.database import build_engine
from app.models import Base, DashboardChatMessage, Idea, SourcePost
from app.models.enums import ContentType
from app.services.configuration import ConfigurationService
from app.services.idea_collector import IdeaCollector
from app.web.dashboard import create_dashboard_router
from desktop import EmbeddedServer


def _settings(tmp_path: Path) -> Settings:
    config_dir = tmp_path / "config"
    shutil.copytree(Path("config"), config_dir)
    return Settings(
        _env_file=None,
        mock_mode=True,
        x_user_id=None,
        database_url=f"sqlite:///{(tmp_path / 'app.db').as_posix()}",
        config_dir=config_dir,
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        logs_dir=tmp_path / "logs",
    )


def _client(tmp_path: Path) -> tuple[TestClient, object]:
    settings = _settings(tmp_path)
    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)
    app = FastAPI()
    app.include_router(create_dashboard_router(settings, engine))
    return TestClient(app), engine


def _post(index: int, published_at: datetime) -> SourcePost:
    text = f"Source post {index}"
    return SourcePost(
        x_post_id=str(1000 + index),
        url=f"https://x.com/example/status/{1000 + index}",
        author_id=f"author-{index}",
        username=f"user{index}",
        text=text,
        language="en",
        published_at=published_at,
        public_metrics={"like_count": index},
        media_metadata={
            "items": [
                {
                    "type": "photo",
                    "url": f"https://example.invalid/{index}.jpg",
                    "alt_text": "Example image",
                }
            ]
        },
        source_labels=["selected_account"],
        fetched_at=published_at,
        raw_json={
            "author": {
                "name": f"User {index}",
                "username": f"user{index}",
                "verified": index % 2 == 0,
            }
        },
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
        created_at=published_at,
        updated_at=published_at,
    )


def test_dashboard_serves_shared_static_interface(tmp_path: Path) -> None:
    client, engine = _client(tmp_path)
    try:
        response = client.get("/")
        assert response.status_code == 200
        assert "Vouch" in response.text
        assert client.get("/assets/app.css").status_code == 200
        assert client.get("/assets/design.css").status_code == 200
        assert client.get("/assets/app.js").status_code == 200
        javascript = client.get("/assets/app.js").text
        assert "sortFeedNewestFirst" in javascript
        assert "openDraftAction" in javascript
        assert "Publish to X" in javascript
        assert 'link.target = "_blank"' in javascript
        assert "publishPreviewDraftId" in javascript
        assert 'id="draft-action-dialog"' in response.text
        login = client.get("/login")
        assert login.status_code == 200
        assert "Sign in with a one-time code sent to your email" in login.text
        assert client.get("/assets/auth.css").status_code == 200
        assert client.get("/assets/auth.js").status_code == 200
        assert "Email sign-in delivery" not in response.text
        assert "Server online" not in response.text
        assert '<div class="chat-orb" aria-hidden="true"><img' in response.text
    finally:
        engine.dispose()


def test_dashboard_credentials_are_write_only_and_restart_bounded(tmp_path: Path) -> None:
    client, engine = _client(tmp_path)
    try:
        saved = client.put(
            "/api/dashboard/credentials",
            json={
                "values": {
                    "openai_api_key": "synthetic-dashboard-key",
                },
                "clear": [],
            },
        )
        assert saved.status_code == 200
        assert saved.json()["restart_required"] is True
        assert "synthetic-dashboard-key" not in saved.text
        status = client.get("/api/dashboard/credentials")
        assert status.status_code == 200
        assert status.json()["configured"]["openai_api_key"] is False
        assert saved.json()["restart_required"] is True
        assert status.json()["values_returned"] is False
        assert "synthetic-dashboard-key" not in status.text
        assert "OPENAI_API_KEY=synthetic-dashboard-key" in (tmp_path / ".env").read_text(
            encoding="utf-8"
        )
    finally:
        engine.dispose()


def test_activity_feed_is_persistent_and_paginated(tmp_path: Path) -> None:
    client, engine = _client(tmp_path)
    now = datetime.now(UTC)
    try:
        with Session(engine) as session:
            session.add_all(_post(index, now + timedelta(minutes=index)) for index in range(3))
            session.commit()

        first = client.get("/api/activity/feed", params={"limit": 2}).json()
        assert [item["text"] for item in first["items"]] == ["Source post 2", "Source post 1"]
        assert first["next_cursor"] is not None
        assert first["items"][0]["author"]["name"] == "User 2"
        assert first["items"][0]["media"][0]["type"] == "photo"

        second = client.get("/api/activity/feed", params=first["next_cursor"]).json()
        assert [item["text"] for item in second["items"]] == ["Source post 0"]
        assert second["next_cursor"] is None
    finally:
        engine.dispose()


def test_chat_commands_are_allow_listed_and_persisted(tmp_path: Path) -> None:
    client, engine = _client(tmp_path)
    try:
        response = client.post("/api/chat/command", json={"text": "/status"})
        assert response.status_code == 200
        assert "online" in response.json()["message"]

        history = client.get("/api/chat/history").json()["items"]
        assert [item["role"] for item in history] == ["user", "assistant"]
        assert history[0]["text"] == "/status"
        assert "Recent Search: disabled" in history[1]["text"]
        with Session(engine) as session:
            assert session.query(DashboardChatMessage).count() == 2
    finally:
        engine.dispose()


def test_config_editor_never_exposes_dotenv(tmp_path: Path) -> None:
    client, engine = _client(tmp_path)
    try:
        assert client.get("/api/config/files/.env").status_code == 404
        response = client.put(
            "/api/config/files/.env",
            json={"content": "TOKEN=value"},
        )
        assert response.status_code == 404
    finally:
        engine.dispose()


def test_embedded_server_shutdown_requests_in_process_exit() -> None:
    server = EmbeddedServer("127.0.0.1", 8000)
    server.owned = True
    server.server = SimpleNamespace(should_exit=False, force_exit=False)
    server.thread = SimpleNamespace(is_alive=lambda: False, join=lambda timeout: None)

    server.stop()

    assert server.server.should_exit is True
    assert server.owned is False


def test_dashboard_settings_are_structured_and_force_recent_search_off(tmp_path: Path) -> None:
    client, engine = _client(tmp_path)
    try:
        response = client.get("/api/dashboard/settings")
        assert response.status_code == 200
        payload = response.json()
        assert "run_on_start" not in payload
        assert "evergreen_topics" in payload

        status = client.get("/api/dashboard/status").json()
        assert status["startup_discovery"] is True
        assert status["home_timeline"] is True
        assert status["recent_search"] is False

        content_sources = client.get("/api/config/files/content_sources.yml").json()["content"]
        assert "recent_search:" in content_sources
        assert "enabled: false" in content_sources
        assert "run_on_start" not in content_sources
    finally:
        engine.dispose()


def test_selected_accounts_can_be_added_paused_and_deleted(tmp_path: Path) -> None:
    client, engine = _client(tmp_path)
    try:
        initial = client.get("/api/selected-accounts")
        assert initial.status_code == 200
        assert initial.json()["items"] == []

        added = client.post(
            "/api/selected-accounts",
            json={"username": "https://x.com/OpenAI"},
        )
        assert added.status_code == 201
        assert added.json()["items"] == [{"username": "openai", "enabled": True, "user_id": None}]

        paused = client.patch(
            "/api/selected-accounts/openai",
            json={"enabled": False},
        )
        assert paused.status_code == 200
        assert paused.json()["items"][0]["enabled"] is False

        removed = client.delete("/api/selected-accounts/openai")
        assert removed.status_code == 200
        assert removed.json()["items"] == []
    finally:
        engine.dispose()


def test_new_command_requires_a_direct_x_post_url(tmp_path: Path) -> None:
    client, engine = _client(tmp_path)
    try:
        missing = client.post("/api/chat/command", json={"text": "/new"})
        assert missing.status_code == 200
        assert missing.json()["error"] == "missing_url"

        invalid = client.post(
            "/api/chat/command",
            json={"text": "/new https://example.com/post/123"},
        )
        assert invalid.status_code == 200
        assert invalid.json()["error"] == "invalid_url"
    finally:
        engine.dispose()


def test_new_command_imports_direct_tweet_into_persistent_activity(tmp_path: Path) -> None:
    client, engine = _client(tmp_path)
    try:
        with client:
            response = client.post(
                "/api/chat/command",
                json={"text": "/new https://x.com/example/status/1234567890"},
            )
            assert response.status_code == 200
            assert response.json()["action"] == "activity"

            completed = False
            for _ in range(100):
                history = client.get("/api/chat/history?limit=20").json()["items"]
                completed = any(
                    item["role"] == "assistant"
                    and item["command"] == "/new"
                    and "Generation completed" in item["text"]
                    for item in history
                )
                if completed:
                    break
                time.sleep(0.02)

            assert completed is True
            drafts = client.get("/api/drafts/recent").json()["items"]
            assert len(drafts) == 1
            assert drafts[0]["status"] == "needs_review"
            assert drafts[0]["text"]
            feed = client.get("/api/activity/feed").json()["items"]
            assert len(feed) == 1
            assert feed[0]["x_post_id"] == "1234567890"
            assert feed[0]["source_labels"] == ["manual_url"]
    finally:
        engine.dispose()


def test_dashboard_draft_review_actions_preserve_manual_publish_gate(tmp_path: Path) -> None:
    client, engine = _client(tmp_path)
    try:
        with client:
            started = client.post(
                "/api/chat/command",
                json={"text": "/new https://x.com/example/status/1234567890"},
            )
            assert started.status_code == 200

            draft: dict[str, object] | None = None
            for _ in range(100):
                items = client.get("/api/drafts/recent").json()["items"]
                if items:
                    draft = items[0]
                    break
                time.sleep(0.02)
            assert draft is not None
            draft_id = str(draft["id"])

            approved = client.post(
                f"/api/drafts/{draft_id}/approve",
                json={"facts_confirmed": True},
            )
            assert approved.status_code == 200
            assert approved.json()["draft"]["status"] == "approved"

            edited_text = str(draft["text"]) + " Still worth noticing."
            edited = client.put(
                f"/api/drafts/{draft_id}",
                json={"text": edited_text},
            )
            assert edited.status_code == 200
            edited_draft = edited.json()["draft"]
            assert edited_draft["text"] == edited_text
            assert edited_draft["status"] == "needs_review"
            assert edited_draft["can_approve"] is True

            reapproved = client.post(
                f"/api/drafts/{draft_id}/approve",
                json={"facts_confirmed": True},
            )
            assert reapproved.status_code == 200
            preview = client.post(f"/api/drafts/{draft_id}/publish-preview")
            assert preview.status_code == 409
            assert "quote post" in preview.json()["detail"]
            chat = client.get("/api/chat/history?limit=20").json()["items"]
            source_notice = next(item for item in chat if item["command"] == "/publish")
            assert "https://x.com/example/status/1234567890" in source_notice["text"]
            assert source_notice["metadata"]["status"] == "quote_context_required"
            current = reapproved.json()["draft"]
            phrase = f"PUBLISH {draft_id} {current['hash_prefix']}"
            assert phrase.startswith(f"PUBLISH {draft_id} ")

            disabled_publish = client.post(
                f"/api/drafts/{draft_id}/publish",
                json={"confirmation_phrase": phrase},
            )
            assert disabled_publish.status_code == 409
            assert disabled_publish.json()["detail"] == "PUBLISH_ENABLED is false"
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_manual_url_is_an_explicit_quote_reaction_request(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)

    class Reader:
        async def get_post(self, post_id: str) -> dict[str, object]:
            return {
                "id": post_id,
                "text": "A model added a low-verbosity setting.",
                "author_id": "author-1",
                "username": "example",
                "url": f"https://x.com/example/status/{post_id}",
                "public_metrics": {},
            }

    try:
        with Session(engine) as session:
            collector = IdeaCollector(
                session,
                source_configuration=ConfigurationService(settings.config_dir).load_sources(),
            )
            idea_ids = await collector.import_url(
                "https://x.com/example/status/1234567890",
                Reader(),
            )
            idea = session.get(Idea, idea_ids[0])

            assert idea is not None
            assert idea.recommended_format is ContentType.QUOTE_COMMENTARY
            assert idea.explanation["origin"] == "manual_url"
            assert idea.explanation["editorial_intent"] == "quote_reaction"
            assert idea.explanation["operator_requested_generation"] is True
    finally:
        engine.dispose()
