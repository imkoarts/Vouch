from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import Settings
from app.domain.enums import ContentType
from app.models.enums import DraftStatus, FactCheckStatus
from app.services.draft_files import DraftArtifactError
from app.telegram.bot import TelegramReviewBot, notify_draft_once

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeTelegramApi:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.events.append(
            ("message", {"chat_id": chat_id, "text": text, "reply_markup": reply_markup})
        )
        return {"message_id": len(self.events)}

    async def send_photo(
        self, chat_id: int, path: Path, *, caption: str | None = None
    ) -> dict[str, Any]:
        self.events.append(("photo", {"chat_id": chat_id, "path": path, "caption": caption}))
        return {"message_id": len(self.events)}

    async def send_video(
        self, chat_id: int, path: Path, *, caption: str | None = None
    ) -> dict[str, Any]:
        self.events.append(("video", {"chat_id": chat_id, "path": path, "caption": caption}))
        return {"message_id": len(self.events)}

    async def answer_callback_query(self, callback_query_id: str, text: str = "") -> None:
        self.events.append(("answer", {"id": callback_query_id, "text": text}))


def _settings(tmp_path: Path) -> Settings:
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    for name in ("runtime.yml", "content_profile.yml", "content_sources.yml", "cost_estimates.yml"):
        content = (PROJECT_ROOT / "config" / name).read_text(encoding="utf-8")
        if name == "runtime.yml":
            content = content.replace("operator_user_id: null", "operator_user_id: 123456789")
        (config_dir / name).write_text(content, encoding="utf-8")
    return Settings(
        _env_file=None,
        config_dir=config_dir,
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        mock_mode=True,
        publish_enabled=False,
    )


@pytest.mark.asyncio
async def test_thread_is_sent_in_blocks_before_review_keyboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeTelegramApi()
    bot = TelegramReviewBot(_settings(tmp_path), api)  # type: ignore[arg-type]
    artifact = tmp_path / "draft"
    artifact.mkdir()
    draft = SimpleNamespace(
        id="draft-123",
        media_plan={"type": "none", "required_files": []},
        artifact_path=str(artifact),
        status=DraftStatus.NEEDS_REVIEW,
        current_content_hash="a" * 64,
        content_type=ContentType.THREAD,
    )
    version = SimpleNamespace(
        content={"parts": ["First part", "Second part"]},
        rendered_text="First part\n\nSecond part",
    )
    artifacts = SimpleNamespace(validated_media_files=lambda *_args, **_kwargs: ())
    drafts = SimpleNamespace(
        reconcile=lambda *_args, **_kwargs: None,
        get=lambda _draft_id: draft,
        current_version=lambda _draft: version,
        artifacts=artifacts,
    )

    @contextmanager
    def fake_session_scope():
        yield object()

    monkeypatch.setattr("app.telegram.bot.session_scope", fake_session_scope)
    monkeypatch.setattr(
        "app.telegram.bot.build_services",
        lambda *_args, **_kwargs: SimpleNamespace(drafts=drafts),
    )

    await bot.notify_draft("draft-123")

    messages = [payload for kind, payload in api.events if kind == "message"]
    assert "Part 1/2" in messages[1]["text"]
    assert "First part" in messages[1]["text"]
    assert "Part 2/2" in messages[2]["text"]
    assert "Second part" in messages[2]["text"]
    assert messages[-1]["text"] == "Choose the next step:"
    labels = [
        button["text"] for row in messages[-1]["reply_markup"]["inline_keyboard"] for button in row
    ]
    assert labels == [
        "💾 Save draft",
        "🚀 Publish",
        "🔄 New from X",
        "🗑 Reject",
        "✍️ Custom topic",
    ]


@pytest.mark.asyncio
async def test_invalid_media_manifest_never_sends_an_unverified_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeTelegramApi()
    bot = TelegramReviewBot(_settings(tmp_path), api)  # type: ignore[arg-type]
    artifact = tmp_path / "draft"
    artifact.mkdir()
    draft = SimpleNamespace(
        id="draft-media",
        media_plan={"type": "image", "required_files": ["../../secret.png"]},
        artifact_path=str(artifact),
        status=DraftStatus.NEEDS_REVIEW,
        current_content_hash="b" * 64,
        content_type=ContentType.SHORT_POST,
    )
    version = SimpleNamespace(
        content={"parts": ["Safe text"]},
        rendered_text="Safe text",
    )

    def reject_media(*_args: Any, **_kwargs: Any) -> tuple[Path, ...]:
        raise DraftArtifactError("synthetic unsafe media")

    drafts = SimpleNamespace(
        reconcile=lambda *_args, **_kwargs: None,
        get=lambda _draft_id: draft,
        current_version=lambda _draft: version,
        artifacts=SimpleNamespace(validated_media_files=reject_media),
    )

    @contextmanager
    def fake_session_scope():
        yield object()

    monkeypatch.setattr("app.telegram.bot.session_scope", fake_session_scope)
    monkeypatch.setattr(
        "app.telegram.bot.build_services",
        lambda *_args, **_kwargs: SimpleNamespace(drafts=drafts),
    )

    await bot.notify_draft("draft-media")

    assert not any(kind == "photo" for kind, _payload in api.events)
    messages = [payload["text"] for kind, payload in api.events if kind == "message"]
    assert "Safe text" in messages
    assert any("failed validation" in message for message in messages)


@pytest.mark.asyncio
async def test_prod_button_is_fail_closed_when_publication_flags_are_off(tmp_path: Path) -> None:
    api = FakeTelegramApi()
    bot = TelegramReviewBot(_settings(tmp_path), api)  # type: ignore[arg-type]
    query = {
        "id": "callback-1",
        "from": {"id": 123456789},
        "message": {"chat": {"id": 123456789}},
        "data": "prod|draft-123",
    }

    await bot._handle_callback(query)

    messages = [payload["text"] for kind, payload in api.events if kind == "message"]
    assert messages == [
        "Manual X publishing is disabled in config/runtime.yml and .env. "
        "The post was not published."
    ]


@pytest.mark.asyncio
async def test_thread_command_requests_thread_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeTelegramApi()
    bot = TelegramReviewBot(_settings(tmp_path), api)  # type: ignore[arg-type]
    calls: list[tuple[int, str, ContentType]] = []

    async def fake_generate(
        chat_id: int,
        topic: str,
        *,
        content_type: ContentType = ContentType.SHORT_POST,
    ) -> str:
        calls.append((chat_id, topic, content_type))
        return "draft-thread"

    monkeypatch.setattr(bot, "_generate_topic", fake_generate)
    await bot._handle_message(
        {
            "from": {"id": 123456789},
            "chat": {"id": 123456789},
            "text": "/thread Automation failures",
        }
    )

    assert calls == [(123456789, "Automation failures", ContentType.THREAD)]
    messages = [payload["text"] for kind, payload in api.events if kind == "message"]
    assert messages == ["Generating a thread..."]


@pytest.mark.asyncio
async def test_other_topic_creates_a_separate_draft_with_same_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeTelegramApi()
    bot = TelegramReviewBot(_settings(tmp_path), api)  # type: ignore[arg-type]
    bot.state.set_pending(123456789, action="new_topic", draft_id="old-draft")
    generated: list[tuple[int, str, ContentType]] = []
    previous = SimpleNamespace(content_type=ContentType.THREAD)

    @contextmanager
    def fake_session_scope():
        yield object()

    monkeypatch.setattr("app.telegram.bot.session_scope", fake_session_scope)
    monkeypatch.setattr(
        "app.telegram.bot.build_services",
        lambda *_args, **_kwargs: SimpleNamespace(
            drafts=SimpleNamespace(get=lambda _draft_id: previous)
        ),
    )

    async def fake_generate(
        chat_id: int,
        topic: str,
        *,
        content_type: ContentType = ContentType.SHORT_POST,
    ) -> str:
        generated.append((chat_id, topic, content_type))
        return "new-draft"

    monkeypatch.setattr(bot, "_generate_topic", fake_generate)

    await bot._handle_message(
        {
            "from": {"id": 123456789},
            "chat": {"id": 123456789},
            "text": "A completely different topic",
        }
    )

    assert generated == [(123456789, "A completely different topic", ContentType.THREAD)]
    assert bot.state.pop_pending(123456789) is None
    messages = [payload["text"] for kind, payload in api.events if kind == "message"]
    assert messages[0] == "Generating a separate draft for the custom topic..."
    assert "previous draft was not published" in messages[-1]
    assert "new-draft" in messages[-1]


def test_image_controls_follow_feature_flag_and_key(tmp_path: Path) -> None:
    api = FakeTelegramApi()
    without_key = TelegramReviewBot(_settings(tmp_path), api)  # type: ignore[arg-type]
    draft = SimpleNamespace(id="draft-image", media_plan={"type": "image"})

    labels_without_key = [
        button["text"]
        for row in without_key._review_keyboard(draft, has_image=True)["inline_keyboard"]
        for button in row
    ]

    assert "🖼 Keep image" in labels_without_key
    assert "🚫 Remove image" in labels_without_key
    assert "♻️ Regenerate image" not in labels_without_key


@pytest.mark.asyncio
async def test_automatic_notification_honors_generation_feature_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot_settings = _settings(tmp_path).model_copy(update={"telegram_bot_token": "synthetic-token"})
    bot = TelegramReviewBot(bot_settings, FakeTelegramApi())  # type: ignore[arg-type]
    disabled_generation = bot.configuration.runtime.generation.model_copy(
        update={"notify_telegram": False}
    )
    disabled_runtime = bot.configuration.runtime.model_copy(
        update={"generation": disabled_generation}
    )
    disabled_configuration = bot.configuration.__class__(
        sources=bot.configuration.sources,
        profile=bot.configuration.profile,
        costs=bot.configuration.costs,
        runtime=disabled_runtime,
    )

    monkeypatch.setattr(
        "app.telegram.bot.ConfigurationService.load",
        lambda _service: disabled_configuration,
    )

    class UnexpectedApi:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("Telegram API must not be constructed when notifications are off")

    monkeypatch.setattr("app.telegram.bot.TelegramBotApi", UnexpectedApi)

    await notify_draft_once(bot_settings, "draft-disabled")


@pytest.mark.asyncio
async def test_long_command_is_fail_closed_when_premium_profile_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeTelegramApi()
    bot = TelegramReviewBot(_settings(tmp_path), api)  # type: ignore[arg-type]
    calls: list[tuple[int, str, ContentType]] = []

    async def fake_generate(
        chat_id: int,
        topic: str,
        *,
        content_type: ContentType = ContentType.SHORT_POST,
    ) -> str:
        calls.append((chat_id, topic, content_type))
        return "draft-long"

    monkeypatch.setattr(bot, "_generate_topic", fake_generate)
    await bot._handle_message(
        {
            "from": {"id": 123456789},
            "chat": {"id": 123456789},
            "text": "/long Reviewing long posts",
        }
    )

    assert calls == []
    messages = [payload["text"] for kind, payload in api.events if kind == "message"]
    assert messages == ["Premium long posts are disabled in config/content_profile.yml."]


def test_image_generation_button_requires_live_mode_feature_flag_and_key(tmp_path: Path) -> None:
    live_settings = _settings(tmp_path).model_copy(
        update={"mock_mode": False, "openai_api_key": "synthetic-openai-key"}
    )
    bot = TelegramReviewBot(live_settings, FakeTelegramApi())  # type: ignore[arg-type]
    enabled_images = bot.configuration.runtime.images.model_copy(update={"enabled": True})
    enabled_runtime = bot.configuration.runtime.model_copy(update={"images": enabled_images})
    bot.configuration = bot.configuration.__class__(
        sources=bot.configuration.sources,
        profile=bot.configuration.profile,
        costs=bot.configuration.costs,
        runtime=enabled_runtime,
    )
    draft = SimpleNamespace(id="draft-image-live", media_plan={"type": "none"})

    labels = [
        button["text"]
        for row in bot._review_keyboard(draft, has_image=False)["inline_keyboard"]
        for button in row
    ]

    assert "🎨 Generate image" in labels

    mock_bot = TelegramReviewBot(
        live_settings.model_copy(update={"mock_mode": True}), FakeTelegramApi()
    )  # type: ignore[arg-type]
    mock_bot.configuration = bot.configuration
    mock_labels = [
        button["text"]
        for row in mock_bot._review_keyboard(draft, has_image=False)["inline_keyboard"]
        for button in row
    ]
    assert "🎨 Generate image" not in mock_labels


@pytest.mark.asyncio
async def test_unauthorized_telegram_message_is_ignored(tmp_path: Path) -> None:
    api = FakeTelegramApi()
    bot = TelegramReviewBot(_settings(tmp_path), api)  # type: ignore[arg-type]

    await bot._handle_message(
        {
            "from": {"id": 999999999},
            "chat": {"id": 999999999},
            "text": "/new Prohibited topic",
        }
    )

    assert api.events == []


@pytest.mark.asyncio
async def test_unauthorized_callback_is_rejected_without_running_action(tmp_path: Path) -> None:
    api = FakeTelegramApi()
    bot = TelegramReviewBot(_settings(tmp_path), api)  # type: ignore[arg-type]

    await bot._handle_callback(
        {
            "id": "unauthorized-callback",
            "from": {"id": 999999999},
            "message": {"chat": {"id": 999999999}},
            "data": "save|draft-123",
        }
    )

    assert api.events == [("answer", {"id": "unauthorized-callback", "text": "Access denied"})]


@pytest.mark.asyncio
async def test_manual_prod_requires_second_hash_bound_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeTelegramApi()
    settings = _settings(tmp_path).model_copy(update={"publish_enabled": True})
    bot = TelegramReviewBot(settings, api)  # type: ignore[arg-type]
    publication = bot.configuration.runtime.publication.model_copy(
        update={"manual_x_publish_enabled": True}
    )
    bot.configuration = bot.configuration.__class__(
        sources=bot.configuration.sources,
        profile=bot.configuration.profile,
        costs=bot.configuration.costs,
        runtime=bot.configuration.runtime.model_copy(update={"publication": publication}),
    )
    draft = SimpleNamespace(
        id="draft-123",
        status=DraftStatus.NEEDS_REVIEW,
        current_content_hash="a" * 64,
    )
    approval_calls: list[str] = []
    publish_calls: list[tuple[str, str]] = []
    preview = SimpleNamespace(
        content_hash="a" * 64,
        confirmation_phrase=f"PUBLISH draft-123 {'a' * 12}",
    )

    class FakePublishing:
        async def preview(self, draft_id: str) -> Any:
            assert draft_id == "draft-123"
            return preview

        async def publish(self, draft_id: str, *, confirmation_phrase: str) -> list[Any]:
            publish_calls.append((draft_id, confirmation_phrase))
            return [SimpleNamespace(id="publication-1")]

    @contextmanager
    def fake_session_scope():
        yield object()

    monkeypatch.setattr("app.telegram.bot.session_scope", fake_session_scope)
    monkeypatch.setattr(
        "app.telegram.bot.build_services",
        lambda *_args, **_kwargs: SimpleNamespace(
            drafts=SimpleNamespace(get=lambda _draft_id: draft),
            approvals=SimpleNamespace(
                approve=lambda draft_id, actor: approval_calls.append(f"{draft_id}:{actor}")
            ),
            publishing=FakePublishing(),
        ),
    )

    await bot._handle_callback(
        {
            "id": "callback-prod",
            "from": {"id": 123456789},
            "message": {"chat": {"id": 123456789}},
            "data": "prod|draft-123",
        }
    )

    messages = [payload for kind, payload in api.events if kind == "message"]
    confirmation_button = messages[-1]["reply_markup"]["inline_keyboard"][0][0]
    assert confirmation_button["callback_data"] == "confirm|draft-123|aaaaaaaaaaaa"
    assert approval_calls == ["draft-123:telegram"]
    assert publish_calls == []

    await bot._handle_callback(
        {
            "id": "callback-confirm",
            "from": {"id": 123456789},
            "message": {"chat": {"id": 123456789}},
            "data": confirmation_button["callback_data"],
        }
    )

    assert publish_calls == [("draft-123", preview.confirmation_phrase)]
    assert any(
        kind == "message" and "Mock publication completed" in payload["text"]
        for kind, payload in api.events
    )


@pytest.mark.asyncio
async def test_required_fact_check_uses_explicit_hash_bound_confirmation_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeTelegramApi()
    settings = _settings(tmp_path).model_copy(update={"publish_enabled": True})
    bot = TelegramReviewBot(settings, api)  # type: ignore[arg-type]
    publication = bot.configuration.runtime.publication.model_copy(
        update={"manual_x_publish_enabled": True}
    )
    bot.configuration = bot.configuration.__class__(
        sources=bot.configuration.sources,
        profile=bot.configuration.profile,
        costs=bot.configuration.costs,
        runtime=bot.configuration.runtime.model_copy(update={"publication": publication}),
    )
    artifact = tmp_path / "draft-fact"
    artifact.mkdir()
    (artifact / "sources.json").write_text(
        '[{"url":"https://x.com/source/status/123"}]',
        encoding="utf-8",
    )
    draft = SimpleNamespace(
        id="draft-123",
        status=DraftStatus.NEEDS_REVIEW,
        fact_check_status=FactCheckStatus.REQUIRED,
        current_content_hash="a" * 64,
        artifact_path=str(artifact),
        media_plan={"type": "none", "metadata": {}},
        blocking_safety_flags=[],
    )
    approval_calls: list[tuple[str, str, bool]] = []
    preview = SimpleNamespace(
        content_hash="a" * 64,
        confirmation_phrase=f"PUBLISH draft-123 {'a' * 12}",
    )

    class FakeApprovals:
        def verify_facts(
            self,
            draft_id: str,
            *,
            actor: str,
            expected_hash_prefix: str | None = None,
        ) -> Any:
            assert draft_id == "draft-123"
            assert actor == "telegram"
            assert expected_hash_prefix == "aaaaaaaaaaaa"
            draft.fact_check_status = FactCheckStatus.VERIFIED
            return draft

        def approve(
            self,
            draft_id: str,
            *,
            actor: str,
            facts_confirmed: bool = False,
        ) -> Any:
            approval_calls.append((draft_id, actor, facts_confirmed))
            draft.status = DraftStatus.APPROVED
            return SimpleNamespace(content_hash=draft.current_content_hash)

    class FakePublishing:
        async def preview(self, draft_id: str) -> Any:
            assert draft_id == "draft-123"
            assert draft.fact_check_status is FactCheckStatus.VERIFIED
            assert draft.status is DraftStatus.APPROVED
            return preview

    @contextmanager
    def fake_session_scope():
        yield object()

    monkeypatch.setattr("app.telegram.bot.session_scope", fake_session_scope)
    monkeypatch.setattr(
        "app.telegram.bot.build_services",
        lambda *_args, **_kwargs: SimpleNamespace(
            drafts=SimpleNamespace(get=lambda _draft_id: draft),
            approvals=FakeApprovals(),
            publishing=FakePublishing(),
        ),
    )

    await bot._handle_callback(
        {
            "id": "callback-prod",
            "from": {"id": 123456789},
            "message": {"chat": {"id": 123456789}},
            "data": "prod|draft-123",
        }
    )

    assert approval_calls == []
    messages = [payload for kind, payload in api.events if kind == "message"]
    assert "Fact-check required before approval" in messages[-1]["text"]
    assert "https://x.com/source/status/123" in messages[-1]["text"]
    fact_button = messages[-1]["reply_markup"]["inline_keyboard"][0][0]
    assert fact_button["callback_data"] == "factprod|draft-123|aaaaaaaaaaaa"

    await bot._handle_callback(
        {
            "id": "callback-facts",
            "from": {"id": 123456789},
            "message": {"chat": {"id": 123456789}},
            "data": fact_button["callback_data"],
        }
    )

    assert approval_calls == [("draft-123", "telegram", False)]
    messages = [payload for kind, payload in api.events if kind == "message"]
    assert any("Facts marked verified" in payload["text"] for payload in messages)
    confirmation_button = messages[-1]["reply_markup"]["inline_keyboard"][0][0]
    assert confirmation_button["callback_data"] == "confirm|draft-123|aaaaaaaaaaaa"


@pytest.mark.asyncio
async def test_required_fact_check_save_does_not_raise_processing_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeTelegramApi()
    bot = TelegramReviewBot(_settings(tmp_path), api)  # type: ignore[arg-type]
    draft = SimpleNamespace(
        id="draft-save",
        status=DraftStatus.NEEDS_REVIEW,
        fact_check_status=FactCheckStatus.REQUIRED,
        current_content_hash="b" * 64,
        artifact_path=str(tmp_path / "missing-artifact"),
        media_plan={"type": "none", "metadata": {}},
        blocking_safety_flags=[],
    )

    @contextmanager
    def fake_session_scope():
        yield object()

    monkeypatch.setattr("app.telegram.bot.session_scope", fake_session_scope)
    monkeypatch.setattr(
        "app.telegram.bot.build_services",
        lambda *_args, **_kwargs: SimpleNamespace(
            drafts=SimpleNamespace(get=lambda _draft_id: draft)
        ),
    )

    await bot._handle_callback(
        {
            "id": "callback-save",
            "from": {"id": 123456789},
            "message": {"chat": {"id": 123456789}},
            "data": "save|draft-save",
        }
    )

    messages = [payload for kind, payload in api.events if kind == "message"]
    assert "Fact-check required before approval" in messages[-1]["text"]
    assert messages[-1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == (
        "factsave|draft-save|bbbbbbbbbbbb"
    )


@pytest.mark.asyncio
async def test_fact_verification_persists_before_similarity_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeTelegramApi()
    bot = TelegramReviewBot(_settings(tmp_path), api)  # type: ignore[arg-type]
    draft = SimpleNamespace(
        id="draft-blocked",
        status=DraftStatus.NEEDS_REVIEW,
        fact_check_status=FactCheckStatus.REQUIRED,
        current_content_hash="c" * 64,
        artifact_path=str(tmp_path),
        media_plan={"type": "none", "metadata": {}},
        blocking_safety_flags=["similarity_threshold_exceeded"],
    )
    approve_called = False

    class FakeApprovals:
        def verify_facts(
            self,
            draft_id: str,
            *,
            actor: str,
            expected_hash_prefix: str | None = None,
        ) -> Any:
            assert draft_id == "draft-blocked"
            assert actor == "telegram"
            assert expected_hash_prefix == "cccccccccccc"
            draft.fact_check_status = FactCheckStatus.VERIFIED
            return draft

        def approve(self, *_args: Any, **_kwargs: Any) -> Any:
            nonlocal approve_called
            approve_called = True
            raise AssertionError("blocked draft must not be approved")

    @contextmanager
    def fake_session_scope():
        yield object()

    monkeypatch.setattr("app.telegram.bot.session_scope", fake_session_scope)
    monkeypatch.setattr(
        "app.telegram.bot.build_services",
        lambda *_args, **_kwargs: SimpleNamespace(
            approvals=FakeApprovals(),
            drafts=SimpleNamespace(get=lambda _draft_id: draft),
        ),
    )

    await bot._handle_callback(
        {
            "id": "callback-facts-blocked",
            "from": {"id": 123456789},
            "message": {"chat": {"id": 123456789}},
            "data": "factsave|draft-blocked|cccccccccccc",
        }
    )

    assert draft.fact_check_status is FactCheckStatus.VERIFIED
    assert not approve_called
    messages = [payload for kind, payload in api.events if kind == "message"]
    assert "Facts were recorded as verified" in messages[-1]["text"]
    button = messages[-1]["reply_markup"]["inline_keyboard"][0][0]
    assert button["callback_data"] == "safetyfix|draft-blocked|cccccccccccc"


@pytest.mark.asyncio
async def test_stale_telegram_confirmation_cannot_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeTelegramApi()
    settings = _settings(tmp_path).model_copy(update={"publish_enabled": True})
    bot = TelegramReviewBot(settings, api)  # type: ignore[arg-type]
    publish_called = False

    class FakePublishing:
        async def preview(self, _draft_id: str) -> Any:
            return SimpleNamespace(
                content_hash="b" * 64,
                confirmation_phrase=f"PUBLISH draft-123 {'b' * 12}",
            )

        async def publish(self, _draft_id: str, *, confirmation_phrase: str) -> list[Any]:
            del confirmation_phrase
            nonlocal publish_called
            publish_called = True
            return []

    @contextmanager
    def fake_session_scope():
        yield object()

    monkeypatch.setattr("app.telegram.bot.session_scope", fake_session_scope)
    monkeypatch.setattr(
        "app.telegram.bot.build_services",
        lambda *_args, **_kwargs: SimpleNamespace(publishing=FakePublishing()),
    )

    await bot._handle_callback(
        {
            "id": "callback-stale",
            "from": {"id": 123456789},
            "message": {"chat": {"id": 123456789}},
            "data": "confirm|draft-123|aaaaaaaaaaaa",
        }
    )

    assert publish_called is False
    assert any(
        kind == "message" and "draft changed after confirmation" in payload["text"].lower()
        for kind, payload in api.events
    )


@pytest.mark.asyncio
async def test_source_video_and_quote_context_are_sent_for_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeTelegramApi()
    bot = TelegramReviewBot(_settings(tmp_path), api)  # type: ignore[arg-type]
    artifact = tmp_path / "draft-video"
    artifact.mkdir()
    video = artifact / "media" / "source-1.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    draft = SimpleNamespace(
        id="draft-video",
        media_plan={
            "type": "video",
            "required_files": ["media/source-1.mp4"],
            "metadata": {
                "context_strategy": "attach_source_media",
                "source_url": "https://x.com/bdll_10/status/1",
            },
        },
        artifact_path=str(artifact),
        status=DraftStatus.NEEDS_REVIEW,
        current_content_hash="c" * 64,
        content_type=ContentType.SHORT_POST,
    )
    version = SimpleNamespace(
        content={"parts": ["A source-specific thought."]},
        rendered_text="A source-specific thought.",
        generation_metadata={},
    )
    drafts = SimpleNamespace(
        reconcile=lambda *_args, **_kwargs: None,
        get=lambda _draft_id: draft,
        current_version=lambda _draft: version,
        artifacts=SimpleNamespace(validated_media_files=lambda *_args, **_kwargs: (video,)),
    )

    @contextmanager
    def fake_session_scope():
        yield object()

    monkeypatch.setattr("app.telegram.bot.session_scope", fake_session_scope)
    monkeypatch.setattr(
        "app.telegram.bot.build_services",
        lambda *_args, **_kwargs: SimpleNamespace(drafts=drafts),
    )

    await bot.notify_draft("draft-video")

    videos = [payload for kind, payload in api.events if kind == "video"]
    assert videos == [
        {
            "chat_id": 123456789,
            "path": video,
            "caption": "Source video attached for context",
        }
    ]


@pytest.mark.asyncio
async def test_x_permission_failure_is_reported_without_processing_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.x_api.live import XPermissionError

    api = FakeTelegramApi()
    settings = _settings(tmp_path).model_copy(update={"publish_enabled": True})
    bot = TelegramReviewBot(settings, api)  # type: ignore[arg-type]

    class FakePublishing:
        async def preview(self, _draft_id: str) -> Any:
            return SimpleNamespace(
                content_hash="a" * 64,
                confirmation_phrase=f"PUBLISH draft-123 {'a' * 12}",
            )

        async def publish(self, _draft_id: str, *, confirmation_phrase: str) -> list[Any]:
            del confirmation_phrase
            raise XPermissionError(
                "X denied manual publication with HTTP 403. Set the app to Read and write "
                "and regenerate the token.",
                status_code=403,
                request_id="req-403",
            )

    @contextmanager
    def fake_session_scope():
        yield object()

    monkeypatch.setattr("app.telegram.bot.session_scope", fake_session_scope)
    monkeypatch.setattr(
        "app.telegram.bot.build_services",
        lambda *_args, **_kwargs: SimpleNamespace(publishing=FakePublishing()),
    )

    await bot._handle_callback(
        {
            "id": "callback-confirm",
            "from": {"id": 123456789},
            "message": {"chat": {"id": 123456789}},
            "data": "confirm|draft-123|aaaaaaaaaaaa",
        }
    )

    messages = [payload["text"] for kind, payload in api.events if kind == "message"]
    assert any("No post was published" in message for message in messages)
    assert any("CONFIGURE_VOUCH.bat" in message for message in messages)
    assert all("Processing error" not in message for message in messages)
