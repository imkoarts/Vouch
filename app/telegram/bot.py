"""Long-polling Telegram review bot with human-only actions."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from app.config import Settings
from app.container import build_services
from app.database import session_scope
from app.domain.enums import PostLengthMode
from app.domain.security import FactCheckBlockedError
from app.domain.x_text import weighted_length
from app.models import Draft
from app.models.enums import ContentType, DraftStatus, FactCheckStatus
from app.services.approval_service import ApprovalGateError
from app.services.configuration import ConfigurationService
from app.services.cost_service import BudgetExceededError
from app.services.draft_files import DraftArtifactError
from app.services.draft_service import DraftValidationError
from app.services.manual_generation import (
    TopicResearchError,
    create_researched_topic_idea,
)
from app.services.post_length import selection_from_metadata
from app.services.publishing_service import PublicationGateError
from app.services.trend_discovery import AutomaticDiscoveryService
from app.telegram.api import TelegramApiError, TelegramBotApi
from app.telegram.state import TelegramStateStore
from app.utils.errors import (
    new_error_id,
    operator_message,
    safe_exception_summary,
    safe_traceback,
)
from app.x_api.live import XApiError

_LOGGER = logging.getLogger(__name__)
_MAX_MESSAGE = 3900


def _split_text(text: str, limit: int = _MAX_MESSAGE) -> tuple[str, ...]:
    normalized = text.strip()
    if not normalized:
        return ("(empty text)",)
    chunks: list[str] = []
    remaining = normalized
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at < limit // 2:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return tuple(chunks)


def _button(text: str, data: str) -> dict[str, str]:
    if len(data.encode("utf-8")) > 64:
        raise ValueError("Telegram callback_data exceeds 64 bytes")
    return {"text": text, "callback_data": data}


class TelegramReviewBot:
    def __init__(self, settings: Settings, api: TelegramBotApi) -> None:
        self.settings = settings
        self.api = api
        self.configuration = ConfigurationService(settings.config_dir).load()
        self.runtime = self.configuration.runtime.telegram
        self.state = TelegramStateStore(settings.data_dir / "telegram_state.json")
        self.allowed_user_ids = frozenset(self.runtime.effective_allowed_user_ids)
        self.review_chat_id = self.runtime.effective_review_chat_id

    def _authorized(self, user_id: int | None, chat_id: int | None) -> bool:
        if user_id is None or user_id not in self.allowed_user_ids:
            return False
        return self.review_chat_id is None or chat_id == self.review_chat_id

    @staticmethod
    def _fact_check_status(draft: Draft) -> FactCheckStatus:
        raw_status = getattr(draft, "fact_check_status", FactCheckStatus.NOT_REQUIRED)
        if isinstance(raw_status, FactCheckStatus):
            return raw_status
        try:
            return FactCheckStatus(str(getattr(raw_status, "value", raw_status)))
        except ValueError:
            return FactCheckStatus.REQUIRED

    @staticmethod
    def _fact_check_source_urls(draft: Draft) -> tuple[str, ...]:
        """Return safe X source links from the projected artifact bundle.

        The projection is local application output, but URL values still come from
        external source metadata.  Only canonical X/Twitter HTTPS links are shown.
        """

        urls: list[str] = []
        artifact_path = getattr(draft, "artifact_path", None)
        if isinstance(artifact_path, str) and artifact_path:
            source_file = Path(artifact_path) / "sources.json"
            try:
                payload = json.loads(source_file.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                payload = []
            if isinstance(payload, list):
                for item in payload:
                    if not isinstance(item, Mapping):
                        continue
                    raw_url = item.get("url")
                    if isinstance(raw_url, str):
                        urls.append(raw_url.strip())

        media_plan = getattr(draft, "media_plan", {})
        if isinstance(media_plan, Mapping):
            metadata = media_plan.get("metadata", {})
            if isinstance(metadata, Mapping):
                for key in ("quote_source_url", "source_url"):
                    raw_url = metadata.get(key)
                    if isinstance(raw_url, str):
                        urls.append(raw_url.strip())

        safe: list[str] = []
        allowed_hosts = {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}
        for url in urls:
            if any(character.isspace() for character in url):
                continue
            try:
                parsed = urlsplit(url)
            except ValueError:
                continue
            if (
                parsed.scheme.casefold() != "https"
                or (parsed.hostname or "").casefold() not in allowed_hosts
                or parsed.username is not None
                or parsed.password is not None
            ):
                continue
            if url not in safe:
                safe.append(url)
            if len(safe) >= 3:
                break
        return tuple(safe)

    async def _request_fact_confirmation(
        self,
        chat_id: int,
        draft: Draft,
        *,
        next_action: str,
    ) -> None:
        content_hash = str(getattr(draft, "current_content_hash", "") or "")
        if len(content_hash) < 12:
            await self.api.send_message(
                chat_id,
                "The draft has no stable content hash. Reopen the draft before continuing.",
            )
            return
        links = self._fact_check_source_urls(draft)
        source_block = ""
        if links:
            source_block = "\n\nSources to review:\n" + "\n".join(links)
        action_label = "continue to publication" if next_action == "prod" else "approve the draft"
        callback_action = "factprod" if next_action == "prod" else "factsave"
        await self.api.send_message(
            chat_id,
            (
                "Fact-check required before approval. Review the source material and confirm "
                "only if you personally verified the factual claims. Confirming records this "
                f"exact draft hash as verified and will {action_label}."
                f"{source_block}"
            ),
            reply_markup={
                "inline_keyboard": [
                    [
                        _button(
                            "✅ I verified the facts",
                            f"{callback_action}|{draft.id}|{content_hash[:12]}",
                        )
                    ],
                    [_button("Cancel", f"cancel|{draft.id}")],
                ]
            },
        )

    async def _send_publish_confirmation(
        self,
        chat_id: int,
        draft_id: str,
        *,
        content_hash: str,
    ) -> None:
        await self.api.send_message(
            chat_id,
            "Confirm the X publication with the separate button below.",
            reply_markup={
                "inline_keyboard": [
                    [
                        _button(
                            "Confirm publish",
                            f"confirm|{draft_id}|{content_hash[:12]}",
                        )
                    ]
                ]
            },
        )

    @staticmethod
    def _message_identity(message: Mapping[str, Any]) -> tuple[int | None, int | None]:
        sender = message.get("from")
        chat = message.get("chat")
        user_id = sender.get("id") if isinstance(sender, Mapping) else None
        chat_id = chat.get("id") if isinstance(chat, Mapping) else None
        return (
            int(user_id) if isinstance(user_id, int) else None,
            int(chat_id) if isinstance(chat_id, int) else None,
        )

    async def _send_chunks(self, chat_id: int, text: str, *, prefix: str = "") -> None:
        chunks = _split_text(text)
        for index, chunk in enumerate(chunks, start=1):
            label = f"{prefix} [{index}/{len(chunks)}]\n\n" if len(chunks) > 1 else prefix
            await self.api.send_message(chat_id, f"{label}{chunk}".strip())

    def _review_keyboard(self, draft: Draft, *, has_image: bool = False) -> dict[str, Any]:
        fact_check_required = self._fact_check_status(draft) is FactCheckStatus.REQUIRED
        save_label = "🔎 Verify & save" if fact_check_required else "💾 Save draft"
        publish_label = "🔎 Verify & publish" if fact_check_required else "🚀 Publish"
        rows: list[list[dict[str, str]]] = [
            [
                _button(save_label, f"save|{draft.id}"),
                _button(publish_label, f"prod|{draft.id}"),
            ],
            [
                _button("🔄 New from X", f"discover|{draft.id}"),
                _button("🗑 Reject", f"reject|{draft.id}"),
            ],
            [_button("✍️ Custom topic", f"custom|{draft.id}")],
        ]
        blocking_flags = tuple(getattr(draft, "blocking_safety_flags", ()) or ())
        content_hash = str(getattr(draft, "current_content_hash", "") or "")
        if blocking_flags and len(content_hash) >= 12:
            rows.insert(
                0,
                [
                    _button(
                        "🛠 Fix safety flags",
                        f"safetyfix|{draft.id}|{content_hash[:12]}",
                    )
                ],
            )
        media_type = str(draft.media_plan.get("type", "none"))
        image_generation_available = (
            not self.settings.mock_mode
            and self.configuration.runtime.images.enabled
            and self.configuration.runtime.providers.openai.enabled
            and self.settings.openai_api_key is not None
        )
        if has_image:
            image_buttons = [_button("🖼 Keep image", f"img_keep|{draft.id}")]
            if image_generation_available:
                image_buttons.append(_button("♻️ Regenerate image", f"img_regen|{draft.id}"))
            image_buttons.append(_button("🚫 Remove image", f"img_drop|{draft.id}"))
            rows.append(image_buttons)
        elif image_generation_available:
            rows.append([_button("🎨 Generate image", f"img_regen|{draft.id}")])
        elif media_type == "image":
            rows.append([_button("🚫 Remove image", f"img_drop|{draft.id}")])
        return {"inline_keyboard": rows}

    async def notify_draft(self, draft_id: str, *, chat_id: int | None = None) -> None:
        target_chat = chat_id or self.review_chat_id
        if target_chat is None:
            raise TelegramApiError("Telegram operator_user_id/review_chat_id is not configured")
        with session_scope() as session:
            services = build_services(session, self.settings)
            services.drafts.reconcile(draft_id, actor="telegram")
            draft = services.drafts.get(draft_id)
            version = services.drafts.current_version(draft)
            parts_raw = version.content.get("parts", [])
            parts = tuple(part for part in parts_raw if isinstance(part, str))
            artifact_path = Path(draft.artifact_path)
            media_warning = False
            try:
                media_files = services.drafts.artifacts.validated_media_files(
                    artifact_path,
                    draft.media_plan,
                )
            except DraftArtifactError:
                media_files = ()
                media_warning = True
            status = draft.status.value
            content_hash = draft.current_content_hash or ""
            has_image = any(
                path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"} for path in media_files
            )
            keyboard = self._review_keyboard(draft, has_image=has_image)
            content_type = draft.content_type
            generation_metadata = getattr(version, "generation_metadata", {})
            length_selection = selection_from_metadata(
                generation_metadata.get("post_length")
                if isinstance(generation_metadata, dict)
                else None
            )
            actual_length = (
                weighted_length(parts[0])
                if parts and content_type is not ContentType.THREAD
                else None
            )
            media_metadata = (
                draft.media_plan.get("metadata", {}) if isinstance(draft.media_plan, dict) else {}
            )
            context_strategy = (
                str(media_metadata.get("context_strategy", ""))
                if isinstance(media_metadata, dict)
                else ""
            )
            quote_source_url = (
                str(media_metadata.get("quote_source_url", ""))
                if isinstance(media_metadata, dict)
                else ""
            )
            publication_context = (
                media_metadata.get("publication_context", {})
                if isinstance(media_metadata, dict)
                else {}
            )
            publication_format = (
                str(publication_context.get("recommended_format", ""))
                if isinstance(publication_context, dict)
                else ""
            )
            source_dependency = (
                publication_context.get("source_dependency")
                if isinstance(publication_context, dict)
                else None
            )
            standalone_clarity = (
                publication_context.get("standalone_clarity")
                if isinstance(publication_context, dict)
                else None
            )
            raw_fact_check_status = getattr(draft, "fact_check_status", "not_required")
            fact_check_status = getattr(raw_fact_check_status, "value", raw_fact_check_status)

        length_line = ""
        if publication_format == "quote_post":
            length_line = f"\nLength: quote commentary (<=280), actual {actual_length}"
        elif length_selection is not None:
            length_line = (
                f"\nLength: {length_selection.label} "
                f"({length_selection.minimum}-{length_selection.maximum}), "
                f"actual {actual_length}"
            )
        publication_line = f"\nPublication: {publication_format}" if publication_format else ""
        dependency_line = ""
        if isinstance(source_dependency, (int, float)) and isinstance(
            standalone_clarity, (int, float)
        ):
            dependency_line = (
                f"\nSource dependency: {float(source_dependency):.2f}; "
                f"standalone clarity: {float(standalone_clarity):.2f}"
            )
        await self.api.send_message(
            target_chat,
            (
                f"New draft\n"
                f"ID: {draft_id}\n"
                f"Status: {status}\n"
                f"Hash: {content_hash[:12]}\n"
                f"Format: {content_type.value}"
                f"{publication_line}"
                f"\nFact check: {fact_check_status}"
                f"{dependency_line}"
                f"{length_line}"
            ),
        )
        if content_type is ContentType.THREAD:
            for index, part in enumerate(parts, start=1):
                await self._send_chunks(
                    target_chat,
                    part,
                    prefix=f"Part {index}/{len(parts)}\n\n",
                )
        else:
            await self._send_chunks(target_chat, parts[0] if parts else version.rendered_text)
        for media_file in media_files:
            suffix = media_file.suffix.casefold()
            if media_file.is_file() and suffix in {".png", ".jpg", ".jpeg", ".webp"}:
                caption = (
                    "Source image attached for context"
                    if context_strategy == "attach_source_media"
                    else "Draft image"
                )
                await self.api.send_photo(target_chat, media_file, caption=caption)
            elif media_file.is_file() and suffix == ".mp4":
                await self.api.send_video(
                    target_chat,
                    media_file,
                    caption="Source video attached for context",
                )
        if context_strategy == "quote_post" and quote_source_url:
            await self.api.send_message(
                target_chat,
                "Quote-post context (open this post when publishing manually):\n"
                + quote_source_url,
            )
        if media_warning:
            await self.api.send_message(
                target_chat,
                "Media failed validation and was not sent.",
            )
        await self.api.send_message(
            target_chat,
            "Choose the next step:",
            reply_markup=keyboard,
        )

    async def _help(self, chat_id: int) -> None:
        await self.api.send_message(
            chat_id,
            (
                "Commands:\n"
                "/discover — choose a topic automatically from X signals\n"
                "/new <topic> — create a post about a topic you provide\n"
                "/long <topic> — create a Premium long post\n"
                "/thread <topic> — create a thread\n"
                "/latest — show the latest draft\n"
                "/draft <id> — show a draft\n"
                "/cancel — cancel custom-topic input\n\n"
                "Automatic discovery is the primary workflow. /new is an optional "
                "manual-topic command. Publishing always requires a separate human confirmation."
            ),
        )

    async def _generate_topic(
        self,
        chat_id: int,
        topic: str,
        *,
        content_type: ContentType = ContentType.SHORT_POST,
        post_length_mode: PostLengthMode | None = None,
    ) -> str:
        await self.api.send_message(
            chat_id,
            "Researching a small, budget-limited X sample before drafting...",
        )
        idea_id = await create_researched_topic_idea(
            self.settings,
            topic_text=topic,
            content_type=content_type,
            actor="telegram_custom_topic",
        )

        for attempt in range(2):
            try:
                with session_scope() as session:
                    services = build_services(session, self.settings)
                    draft = await services.drafts.generate_from_idea(
                        idea_id,
                        actor="telegram",
                        post_length_mode=post_length_mode,
                    )
                    draft_id = None if draft is None else draft.id
                break
            except OperationalError as exc:
                if attempt or "locked" not in str(exc).casefold():
                    raise
                await asyncio.sleep(0.75)
        if draft_id is None:
            await self.api.send_message(
                chat_id,
                "Editorial decision: no post needed. The source already covers the available "
                "point, or no grounded contribution survived review.",
            )
            return ""
        await self.notify_draft(draft_id, chat_id=chat_id)
        return draft_id

    async def _generate_topic_safely(
        self,
        chat_id: int,
        topic: str,
        *,
        content_type: ContentType = ContentType.SHORT_POST,
        post_length_mode: PostLengthMode | None = None,
    ) -> str | None:
        try:
            if post_length_mode is None:
                return await self._generate_topic(
                    chat_id,
                    topic,
                    content_type=content_type,
                )
            return await self._generate_topic(
                chat_id,
                topic,
                content_type=content_type,
                post_length_mode=post_length_mode,
            )
        except TopicResearchError as exc:
            await self.api.send_message(
                chat_id,
                f"{exc} Try a narrower topic or run /discover.",
            )
            return None

    async def _handle_message(self, message: Mapping[str, Any]) -> None:
        user_id, chat_id = self._message_identity(message)
        if not self._authorized(user_id, chat_id) or user_id is None or chat_id is None:
            return
        text = message.get("text")
        if not isinstance(text, str):
            return
        text = text.strip()
        if text == "/start" or text == "/help":
            await self._help(chat_id)
            return
        if text == "/cancel":
            cleared = self.state.clear_pending(user_id)
            await self.api.send_message(
                chat_id, "Pending input cancelled." if cleared else "There is no pending input."
            )
            return
        if text == "/discover":
            await self.api.send_message(
                chat_id,
                (
                    "Reviewing a budget-limited sample of home timeline and trend signals "
                    "from the last 24 hours..."
                ),
            )
            outcome = await AutomaticDiscoveryService(self.settings).run_once(
                actor="telegram_discover"
            )
            if outcome.draft_id is not None:
                await self.notify_draft(outcome.draft_id, chat_id=chat_id)
                await self.api.send_message(
                    chat_id,
                    (
                        f"Sources: home={outcome.home_posts}, "
                        f"selected accounts={outcome.search_posts}. "
                        f"Maximum estimated X read cost: ${outcome.estimated_x_cost_usd}."
                    ),
                )
            else:
                await self.api.send_message(chat_id, outcome.message)
            return

        if text.startswith("/new "):
            topic = text.removeprefix("/new ").strip()
            if not topic:
                await self.api.send_message(
                    chat_id,
                    "Add a topic after /new. Example: /new prediction markets after a token launch",
                )
                return
            await self.api.send_message(chat_id, "Generating a draft...")
            await self._generate_topic_safely(chat_id, topic)
            return
        if text.startswith("/long "):
            topic = text.removeprefix("/long ").strip()
            if not topic:
                await self.api.send_message(chat_id, "Add a topic after /long.")
                return
            if not self.configuration.profile.account.premium_long_posts_enabled:
                await self.api.send_message(
                    chat_id,
                    "Premium long posts are disabled in config/content_profile.yml.",
                )
                return
            await self.api.send_message(chat_id, "Generating a Premium long post...")
            await self._generate_topic_safely(
                chat_id,
                topic,
                content_type=ContentType.LONG_POST,
            )
            return
        if text.startswith("/thread "):
            topic = text.removeprefix("/thread ").strip()
            if not topic:
                await self.api.send_message(chat_id, "Add a topic after /thread.")
                return
            await self.api.send_message(chat_id, "Generating a thread...")
            await self._generate_topic_safely(chat_id, topic, content_type=ContentType.THREAD)
            return
        if text == "/latest":
            with session_scope() as session:
                draft_id = session.scalar(
                    select(Draft.id).order_by(Draft.created_at.desc()).limit(1)
                )
            if draft_id is None:
                await self.api.send_message(chat_id, "No drafts yet.")
            else:
                await self.notify_draft(draft_id, chat_id=chat_id)
            return
        if text.startswith("/draft "):
            await self.notify_draft(text.removeprefix("/draft ").strip(), chat_id=chat_id)
            return

        pending = self.state.pop_pending(user_id)
        if pending is None:
            await self.api.send_message(chat_id, "Unknown command. Use /help.")
            return
        if pending["action"] != "new_topic":
            await self.api.send_message(chat_id, "Unknown pending action was cancelled.")
            return
        try:
            with session_scope() as session:
                services = build_services(session, self.settings)
                previous = services.drafts.get(pending["draft_id"])
                content_type = previous.content_type
                current_version = getattr(services.drafts, "current_version", None)
                previous_length = None
                if callable(current_version):
                    previous_version = current_version(previous)
                    generation_metadata = getattr(previous_version, "generation_metadata", {})
                    if isinstance(generation_metadata, dict):
                        previous_length = selection_from_metadata(
                            generation_metadata.get("post_length")
                        )
            await self.api.send_message(
                chat_id, "Generating a separate draft for the custom topic..."
            )
            if previous_length is None:
                new_draft_id = await self._generate_topic_safely(
                    chat_id,
                    text,
                    content_type=content_type,
                )
            else:
                new_draft_id = await self._generate_topic_safely(
                    chat_id,
                    text,
                    content_type=content_type,
                    post_length_mode=previous_length.resolved_mode,
                )
        except Exception:
            self.state.set_pending(user_id, action="new_topic", draft_id=pending["draft_id"])
            raise
        if new_draft_id is not None:
            await self.api.send_message(
                chat_id,
                (
                    "A new draft was created. The previous draft was not published and "
                    "remains in history. "
                    f"New ID: {new_draft_id}"
                ),
            )

    async def _handle_callback(self, query: Mapping[str, Any]) -> None:
        query_id = query.get("id")
        sender = query.get("from")
        message = query.get("message")
        data = query.get("data")
        user_id = sender.get("id") if isinstance(sender, Mapping) else None
        chat = message.get("chat") if isinstance(message, Mapping) else None
        chat_id = chat.get("id") if isinstance(chat, Mapping) else None
        if not isinstance(query_id, str):
            return
        if not self._authorized(
            int(user_id) if isinstance(user_id, int) else None,
            int(chat_id) if isinstance(chat_id, int) else None,
        ):
            await self.api.answer_callback_query(query_id, "Access denied")
            return
        if (
            not isinstance(data, str)
            or not isinstance(chat_id, int)
            or not isinstance(user_id, int)
        ):
            await self.api.answer_callback_query(query_id, "Invalid command")
            return
        await self.api.answer_callback_query(query_id, "Accepted")
        parts = data.split("|")
        if len(parts) < 2:
            await self.api.send_message(chat_id, "Invalid button.")
            return
        action, draft_id = parts[0], parts[1]

        if action == "cancel":
            await self.api.send_message(
                chat_id,
                "Cancelled. Nothing was approved or published.",
            )
            return

        if action == "save":
            try:
                with session_scope() as session:
                    services = build_services(session, self.settings)
                    draft = services.drafts.get(draft_id)
                    fact_status = self._fact_check_status(draft)
                    if fact_status is FactCheckStatus.REQUIRED:
                        fact_prompt_draft = draft
                        approval_hash = None
                    elif fact_status is FactCheckStatus.FAILED:
                        fact_prompt_draft = None
                        approval_hash = None
                    elif draft.status is DraftStatus.APPROVED:
                        fact_prompt_draft = None
                        approval_hash = draft.current_content_hash
                    else:
                        fact_prompt_draft = None
                        approval = services.approvals.approve(draft_id, actor="telegram")
                        approval_hash = approval.content_hash
            except (ApprovalGateError, FactCheckBlockedError) as exc:
                await self.api.send_message(chat_id, str(exc))
                return
            if fact_status is FactCheckStatus.FAILED:
                await self.api.send_message(
                    chat_id,
                    "Fact checking failed for this draft. Regenerate or reject it before approval.",
                )
                return
            if fact_prompt_draft is not None:
                await self._request_fact_confirmation(
                    chat_id,
                    fact_prompt_draft,
                    next_action="save",
                )
                return
            message_text = (
                "Draft saved and approved for hash "
                f"{str(approval_hash)[:12]}. Nothing was published."
            )
            await self.api.send_message(chat_id, message_text)
            return

        if action == "prod":
            runtime_publication = self.configuration.runtime.publication
            if (
                not self.settings.publish_enabled
                or not runtime_publication.manual_x_publish_enabled
            ):
                await self.api.send_message(
                    chat_id,
                    (
                        "Manual X publishing is disabled in config/runtime.yml and .env. "
                        "The post was not published."
                    ),
                )
                return
            try:
                with session_scope() as session:
                    services = build_services(session, self.settings)
                    draft = services.drafts.get(draft_id)
                    fact_status = self._fact_check_status(draft)
                    if fact_status is FactCheckStatus.REQUIRED:
                        fact_prompt_draft = draft
                        preview = None
                    elif fact_status is FactCheckStatus.FAILED:
                        fact_prompt_draft = None
                        preview = None
                    else:
                        fact_prompt_draft = None
                        if draft.status is DraftStatus.NEEDS_REVIEW:
                            services.approvals.approve(draft_id, actor="telegram")
                        preview = await services.publishing.preview(draft_id)
            except (PublicationGateError, ApprovalGateError, FactCheckBlockedError) as exc:
                await self.api.send_message(chat_id, str(exc))
                return
            if fact_status is FactCheckStatus.FAILED:
                await self.api.send_message(
                    chat_id,
                    "Fact checking failed for this draft. It cannot be published.",
                )
                return
            if fact_prompt_draft is not None:
                await self._request_fact_confirmation(
                    chat_id,
                    fact_prompt_draft,
                    next_action="prod",
                )
                return
            if preview is None:
                await self.api.send_message(chat_id, "Publication preview is unavailable.")
                return
            await self._send_publish_confirmation(
                chat_id,
                draft_id,
                content_hash=preview.content_hash,
            )
            return

        if action in {"factsave", "factprod"} and len(parts) == 3:
            expected_hash = parts[2]
            if action == "factprod":
                runtime_publication = self.configuration.runtime.publication
                if (
                    not self.settings.publish_enabled
                    or not runtime_publication.manual_x_publish_enabled
                ):
                    await self.api.send_message(
                        chat_id,
                        "Manual X publishing is disabled. Nothing was approved or published.",
                    )
                    return
            try:
                with session_scope() as session:
                    services = build_services(session, self.settings)
                    draft = services.approvals.verify_facts(
                        draft_id,
                        actor="telegram",
                        expected_hash_prefix=expected_hash,
                    )
                    current_hash = str(draft.current_content_hash or "")
                    blocking_flags = tuple(draft.blocking_safety_flags or ())
                    if blocking_flags:
                        approval_hash = None
                        preview = None
                    elif draft.status is DraftStatus.NEEDS_REVIEW:
                        approval = services.approvals.approve(
                            draft_id,
                            actor="telegram",
                        )
                        approval_hash = approval.content_hash
                        preview = (
                            await services.publishing.preview(draft_id)
                            if action == "factprod"
                            else None
                        )
                    elif draft.status is DraftStatus.APPROVED:
                        approval_hash = current_hash
                        preview = (
                            await services.publishing.preview(draft_id)
                            if action == "factprod"
                            else None
                        )
                    else:
                        raise ApprovalGateError(
                            "Only a needs_review or approved draft can continue."
                        )
            except (PublicationGateError, ApprovalGateError, FactCheckBlockedError) as exc:
                await self.api.send_message(chat_id, str(exc))
                return

            if blocking_flags:
                await self.api.send_message(
                    chat_id,
                    (
                        "Facts were recorded as verified for this exact draft hash, but the "
                        "draft still has a separate wording-similarity safety block. It was not "
                        "approved or published. Use the button below to rewrite it independently "
                        "from the source, then review the new version."
                    ),
                    reply_markup={
                        "inline_keyboard": [
                            [
                                _button(
                                    "🛠 Fix draft and re-review",
                                    f"safetyfix|{draft_id}|{current_hash[:12]}",
                                )
                            ],
                            [_button("Cancel", f"cancel|{draft_id}")],
                        ]
                    },
                )
                return

            if action == "factsave":
                await self.api.send_message(
                    chat_id,
                    (
                        "Facts marked verified. Draft saved and approved for hash "
                        f"{str(approval_hash)[:12]}. Nothing was published."
                    ),
                )
                return
            if preview is None:
                await self.api.send_message(chat_id, "Publication preview is unavailable.")
                return
            await self.api.send_message(
                chat_id,
                "Facts marked verified for this exact draft version.",
            )
            await self._send_publish_confirmation(
                chat_id,
                draft_id,
                content_hash=preview.content_hash,
            )
            return

        if action == "safetyfix" and len(parts) == 3:
            expected_hash = parts[2]
            try:
                with session_scope() as session:
                    services = build_services(session, self.settings)
                    draft = services.drafts.get(draft_id)
                    current_hash = str(draft.current_content_hash or "")
                    if current_hash[:12] != expected_hash:
                        await self.api.send_message(
                            chat_id,
                            "The draft changed after this repair button was created. Reopen the "
                            "current version before continuing.",
                        )
                        return
                    await services.drafts.regenerate(
                        draft_id,
                        feedback=(
                            "Fix SOURCE_ECHO only. Preserve the verified event and selected "
                            "angle, but remove the source sentence structure and close paraphrase. "
                            "Write an independent observation in the account voice. Do not add "
                            "facts, certainty, newsroom attribution, or a new topic."
                        ),
                        actor="telegram",
                    )
            except (DraftValidationError, PublicationGateError, ApprovalGateError) as exc:
                await self.api.send_message(chat_id, str(exc))
                return
            await self.api.send_message(
                chat_id,
                "The safety-blocked draft was rewritten. Review and verify the new hash before "
                "approval or publication.",
            )
            await self.notify_draft(draft_id, chat_id=chat_id)
            return

        if action == "confirm" and len(parts) == 3:
            try:
                with session_scope() as session:
                    services = build_services(session, self.settings)
                    preview = await services.publishing.preview(draft_id)
                    if preview.content_hash[:12] != parts[2]:
                        await self.api.send_message(
                            chat_id,
                            "The draft changed after confirmation. Reopen the current draft; "
                            "nothing was published.",
                        )
                        return
                    records = await services.publishing.publish(
                        draft_id,
                        confirmation_phrase=preview.confirmation_phrase,
                    )
            except (PublicationGateError, ApprovalGateError, FactCheckBlockedError) as exc:
                await self.api.send_message(chat_id, str(exc))
                return
            except BudgetExceededError as exc:
                estimated = (
                    f"; next post estimate ${exc.estimated_next}"
                    if exc.estimated_next is not None
                    else ""
                )
                await self.api.send_message(
                    chat_id,
                    (
                        "The bot's local X write budget blocked publication. "
                        f"Used today: ${exc.spent}; limit: ${exc.limit}{estimated}. "
                        "Open CONFIGURE_VOUCH.bat and increase the write budget or enable "
                        "Use the same local budget as X read. No post was published."
                    ),
                )
                return
            except XApiError as exc:
                error_id = new_error_id()
                _LOGGER.warning(
                    "Manual X publication rejected [%s] %s: %s",
                    error_id,
                    type(exc).__name__,
                    safe_exception_summary(exc),
                )
                await self.api.send_message(chat_id, operator_message(exc, error_id))
                return
            mode = "Mock publication" if self.settings.mock_mode else "X publication"
            post_ids = [
                str(post_id)
                for record in records
                if (post_id := getattr(record, "x_post_id", None))
            ]
            links = "\n".join(f"https://x.com/i/web/status/{post_id}" for post_id in post_ids)
            suffix = f"\n{links}" if links else ""
            await self.api.send_message(
                chat_id,
                f"{mode} completed. Local records: {len(records)}.{suffix}",
            )
            return

        if action == "discover":
            await self.api.send_message(
                chat_id,
                "Finding a fresh topic from the current X source sample...",
            )
            outcome = await AutomaticDiscoveryService(self.settings).run_once(
                actor="telegram_discover_button"
            )
            if outcome.draft_id is None:
                await self.api.send_message(chat_id, outcome.message)
            else:
                await self.notify_draft(outcome.draft_id, chat_id=chat_id)
            return

        if action == "custom":
            self.state.set_pending(user_id, action="new_topic", draft_id=draft_id)
            await self.api.send_message(
                chat_id,
                (
                    "Send the custom topic as a plain message, or use /new <topic>. "
                    "Example: /new prediction markets after a token launch. "
                    "The current draft will not be published. Use /cancel to stop."
                ),
            )
            return

        if action == "reject":
            with session_scope() as session:
                services = build_services(session, self.settings)
                services.drafts.quarantine(draft_id, actor="telegram")
            await self.api.send_message(chat_id, "Draft rejected and moved to quarantine.")
            return

        if action == "img_keep":
            await self.api.send_message(chat_id, "The current image remains attached to the draft.")
            return

        if action == "img_drop":
            with session_scope() as session:
                services = build_services(session, self.settings)
                services.drafts.remove_image(draft_id, actor="telegram")
            await self.api.send_message(chat_id, "Image removed. The draft must be reviewed again.")
            await self.notify_draft(draft_id, chat_id=chat_id)
            return

        if action == "img_regen":
            await self.api.send_message(chat_id, "Generating a new image...")
            with session_scope() as session:
                services = build_services(session, self.settings)
                await services.images.generate_for_draft(draft_id)
            await self.notify_draft(draft_id, chat_id=chat_id)
            return

        await self.api.send_message(chat_id, "Unsupported action.")

    async def _process_update(self, update: Mapping[str, Any]) -> None:
        callback = update.get("callback_query")
        if isinstance(callback, Mapping):
            await self._handle_callback(callback)
            return
        message = update.get("message")
        if isinstance(message, Mapping):
            await self._handle_message(message)

    def _authorized_chat_for_update(self, update: Mapping[str, Any]) -> int | None:
        message = update.get("message")
        if isinstance(message, Mapping):
            user_id, chat_id = self._message_identity(message)
            return chat_id if self._authorized(user_id, chat_id) else None
        callback = update.get("callback_query")
        if not isinstance(callback, Mapping):
            return None
        sender = callback.get("from")
        callback_message = callback.get("message")
        chat = callback_message.get("chat") if isinstance(callback_message, Mapping) else None
        user_id = sender.get("id") if isinstance(sender, Mapping) else None
        chat_id = chat.get("id") if isinstance(chat, Mapping) else None
        resolved_user = int(user_id) if isinstance(user_id, int) else None
        resolved_chat = int(chat_id) if isinstance(chat_id, int) else None
        return resolved_chat if self._authorized(resolved_user, resolved_chat) else None

    @staticmethod
    async def _sleep_or_stop(stop_event: asyncio.Event | None, seconds: float) -> None:
        if stop_event is None:
            await asyncio.sleep(seconds)
            return
        with suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=seconds)

    async def run_forever(self, stop_event: asyncio.Event | None = None) -> None:
        offset = self.state.offset()
        initialized = False
        while stop_event is None or not stop_event.is_set():
            try:
                if not initialized:
                    await self.api.get_me()
                    await self.api.delete_webhook(
                        drop_pending_updates=self.runtime.drop_pending_updates_on_start
                    )
                    initialized = True
                updates = await self.api.get_updates(
                    offset=offset,
                    poll_timeout=self.runtime.long_poll_timeout_seconds,
                )
                for update in updates:
                    update_id = update.get("update_id")
                    if not isinstance(update_id, int):
                        continue
                    try:
                        await self._process_update(update)
                    except Exception as exc:
                        error_id = new_error_id()
                        _LOGGER.error(
                            "Telegram update failed [%s] %s: %s\n%s",
                            error_id,
                            type(exc).__name__,
                            safe_exception_summary(exc),
                            safe_traceback(exc),
                        )
                        chat_id = self._authorized_chat_for_update(update)
                        if chat_id is not None:
                            await self.api.send_message(
                                chat_id,
                                operator_message(exc, error_id),
                            )
                    offset = update_id + 1
                    self.state.set_offset(offset)
            except TelegramApiError as exc:
                _LOGGER.warning("Telegram polling error: %s", type(exc).__name__)
                initialized = False
                await self._sleep_or_stop(stop_event, 3)


async def notify_draft_once(settings: Settings, draft_id: str, *, force: bool = False) -> None:
    configuration = ConfigurationService(settings.config_dir).load()
    runtime = configuration.runtime.telegram
    if not runtime.enabled:
        return
    if not force and (
        not runtime.notify_on_new_draft or not configuration.runtime.generation.notify_telegram
    ):
        return
    if settings.telegram_bot_token is None:
        raise TelegramApiError("TELEGRAM_BOT_TOKEN is required when Telegram is enabled")
    api = TelegramBotApi(
        settings.telegram_bot_token.get_secret_value(),
        timeout_seconds=runtime.request_timeout_seconds,
        proxy_url=settings.outbound_proxy_url,
    )
    try:
        await TelegramReviewBot(settings, api).notify_draft(draft_id)
    finally:
        await api.close()
