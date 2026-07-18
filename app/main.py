"""Local FastAPI application with optional Telegram and X Activity background tasks."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select

from app import __version__
from app.config import Settings, get_settings
from app.container import build_services
from app.database import build_engine, session_scope
from app.health import database_readiness, live_payload, not_ready_payload, ready_payload
from app.models import UserAccount, VoiceProfile
from app.schemas.configuration import VoiceProfileConfiguration
from app.services.auth import AuthSessionService
from app.services.configuration import ConfigurationService
from app.services.tenant_context import bind_tenant, reset_tenant
from app.services.tenant_storage import TenantWorkspaceRegistry
from app.services.trend_discovery import (
    DiscoveryOutcome,
    automatic_discovery_loop,
)
from app.services.x_activity import (
    XActivityProcessor,
    XActivityRunner,
    build_activity_client,
)
from app.telegram.api import TelegramBotApi
from app.telegram.bot import TelegramReviewBot
from app.web.auth import SESSION_COOKIE, create_auth_router
from app.web.dashboard import create_dashboard_router

_LOGGER = logging.getLogger(__name__)


def _tenant_discovery_ready(settings: Settings) -> bool:
    """Start a tenant loop only when it is enabled and has an official X read capability."""

    configuration = ConfigurationService(settings.config_dir).load()
    discovery = configuration.sources.automatic_discovery
    if not discovery.enabled or settings.mock_mode:
        return False
    x_ready = not (
        settings.missing_x_home_timeline_credentials() and settings.missing_x_read_credentials()
    )
    provider = configuration.runtime.generation.provider
    ai_ready = (
        settings.openai_api_key is not None
        if provider == "openai"
        else settings.xai_api_key is not None
        if provider == "xai"
        else False
    )
    return x_ready and ai_ready


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime = settings or get_settings()
    runtime.ensure_directories()
    engine = build_engine(runtime.database_url)
    tenant_workspaces = TenantWorkspaceRegistry(runtime)
    tenant_discovery_tasks: dict[str, asyncio.Task[None]] = {}
    tenant_discovery_stops: dict[str, asyncio.Event] = {}
    tenant_discovery_lock = asyncio.Lock()

    async def ensure_tenant_discovery(storage_key: str, settings: Settings) -> None:
        if not _tenant_discovery_ready(settings):
            return
        existing = tenant_discovery_tasks.get(storage_key)
        if existing is not None and not existing.done():
            return
        async with tenant_discovery_lock:
            existing = tenant_discovery_tasks.get(storage_key)
            if existing is not None and not existing.done():
                return
            stop = asyncio.Event()
            tenant_discovery_stops[storage_key] = stop
            tenant_discovery_tasks[storage_key] = asyncio.create_task(
                automatic_discovery_loop(settings, stop_event=stop, on_outcome=None)
            )
            _LOGGER.info("Account-bound automatic discovery started")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        del application
        readiness_failure = database_readiness(engine)
        if readiness_failure is not None:
            _LOGGER.error(
                "Database is not ready (%s); Telegram, discovery and X Activity "
                "were not started. Run START_VOUCH.bat to apply migrations.",
                readiness_failure,
            )
            try:
                yield
            finally:
                engine.dispose()
            return

        telegram_api: TelegramBotApi | None = None
        telegram_bot: TelegramReviewBot | None = None
        telegram_task: asyncio.Task[None] | None = None
        telegram_stop = asyncio.Event()
        activity_task: asyncio.Task[None] | None = None
        activity_stop = asyncio.Event()
        discovery_task: asyncio.Task[None] | None = None
        discovery_stop = asyncio.Event()
        configuration = ConfigurationService(runtime.config_dir).load()
        telegram = configuration.runtime.telegram
        if telegram.enabled and telegram.autostart:
            if runtime.telegram_bot_token is None:
                _LOGGER.warning(
                    "Telegram is enabled but TELEGRAM_BOT_TOKEN is missing; polling was not started"
                )
            else:
                telegram_api = TelegramBotApi(
                    runtime.telegram_bot_token.get_secret_value(),
                    timeout_seconds=telegram.request_timeout_seconds,
                    proxy_url=runtime.outbound_proxy_url,
                )
                telegram_bot = TelegramReviewBot(runtime, telegram_api)
                telegram_task = asyncio.create_task(telegram_bot.run_forever(telegram_stop))

        discovery = configuration.sources.automatic_discovery
        if discovery.enabled:

            async def handle_discovery(outcome: DiscoveryOutcome) -> None:
                if telegram_api is not None and telegram.effective_review_chat_id is not None:
                    for notice in outcome.operator_notices:
                        await telegram_api.send_message(
                            telegram.effective_review_chat_id,
                            notice,
                        )
                if telegram_bot is not None and outcome.draft_id is not None:
                    await telegram_bot.notify_draft(outcome.draft_id)
                    if telegram_api is not None and telegram.effective_review_chat_id is not None:
                        await telegram_api.send_message(
                            telegram.effective_review_chat_id,
                            (
                                f"Automatic selection: home={outcome.home_posts}, "
                                f"selected accounts={outcome.search_posts}; "
                                f"maximum estimated X read cost=${outcome.estimated_x_cost_usd}."
                            ),
                        )
                    return
                if (
                    telegram_api is not None
                    and telegram.effective_review_chat_id is not None
                    and (outcome.draft_id is not None or discovery.notify_when_no_candidate)
                ):
                    await telegram_api.send_message(
                        telegram.effective_review_chat_id,
                        outcome.message,
                    )

            discovery_task = asyncio.create_task(
                automatic_discovery_loop(
                    runtime,
                    stop_event=discovery_stop,
                    on_outcome=handle_discovery,
                )
            )

        activity = configuration.runtime.x_activity
        if activity.enabled and activity.autostart:
            if runtime.mock_mode:
                _LOGGER.warning("X Activity autostart is disabled while MOCK_MODE=true")
            else:
                try:
                    activity_client = build_activity_client(runtime, activity)
                except Exception as exc:
                    _LOGGER.warning("X Activity was not started: %s", type(exc).__name__)
                else:

                    async def handle_activity(envelope: Mapping[str, Any]) -> None:
                        with session_scope() as session:
                            services = build_services(session, runtime)
                            processor = XActivityProcessor(
                                session,
                                settings=runtime,
                                runtime=activity,
                                sources=services.configuration.sources,
                                drafts=services.drafts,
                            )
                            result = await processor.process(envelope)
                            draft_id = result.draft_id
                            notification = result.notification
                        if telegram_bot is not None and draft_id and notification is not None:
                            await telegram_bot.notify_draft(draft_id)
                        elif (
                            telegram_api is not None
                            and telegram.effective_review_chat_id is not None
                            and notification is not None
                        ):
                            await telegram_api.send_message(
                                telegram.effective_review_chat_id,
                                notification,
                            )

                    async def report_activity_error(exc: Exception) -> None:
                        _LOGGER.warning("X Activity stream error: %s", type(exc).__name__)

                    runner = XActivityRunner(
                        activity_client,
                        configuration=activity,
                        handler=handle_activity,
                        error_handler=report_activity_error,
                    )
                    activity_task = asyncio.create_task(runner.run_forever(activity_stop))
        try:
            yield
        finally:
            discovery_stop.set()
            if discovery_task is not None:
                discovery_task.cancel()
                with suppress(asyncio.CancelledError):
                    await discovery_task
            activity_stop.set()
            if activity_task is not None:
                activity_task.cancel()
                with suppress(asyncio.CancelledError):
                    await activity_task
            telegram_stop.set()
            if telegram_task is not None:
                telegram_task.cancel()
                with suppress(asyncio.CancelledError):
                    await telegram_task
            if telegram_api is not None:
                await telegram_api.close()
            for stop in tenant_discovery_stops.values():
                stop.set()
            for task in tenant_discovery_tasks.values():
                task.cancel()
            if tenant_discovery_tasks:
                await asyncio.gather(
                    *tenant_discovery_tasks.values(),
                    return_exceptions=True,
                )
            tenant_discovery_tasks.clear()
            tenant_discovery_stops.clear()
            tenant_workspaces.dispose()
            engine.dispose()

    app = FastAPI(title="Vouch", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def tenant_workspace(request: Request, call_next: Any) -> Any:
        """Authenticate shared API access and bind an isolated account workspace."""

        protected = request.url.path.startswith("/api/") and not request.url.path.startswith(
            "/api/auth/"
        )
        if not protected:
            return await call_next(request)
        from sqlalchemy.orm import Session

        with Session(engine) as database:
            auth = AuthSessionService(database)
            session_token = request.cookies.get(SESSION_COOKIE)
            user = auth.resolve(session_token)
            if user is None:
                return JSONResponse(status_code=401, content={"detail": "Authentication required"})
            if request.method not in {"GET", "HEAD", "OPTIONS"} and not auth.verify_csrf(
                session_token, request.headers.get("X-CSRF-Token")
            ):
                return JSONResponse(status_code=403, content={"detail": "CSRF validation failed"})
            stored_voice = database.scalar(
                select(VoiceProfile).where(VoiceProfile.user_id == user.id)
            )
            voice = (
                VoiceProfileConfiguration(
                    account=stored_voice.account_type,
                    language=stored_voice.language,
                    tone=tuple(stored_voice.tone),
                    response_preferences=tuple(stored_voice.response_preferences),
                    guidance=stored_voice.guidance,
                    banned_tendencies=tuple(stored_voice.banned_tendencies),
                )
                if stored_voice is not None
                else None
            )
            legacy_storage_key = database.scalar(
                select(UserAccount.storage_key)
                .where(UserAccount.auth_provider == "local")
                .order_by(UserAccount.created_at, UserAccount.id)
                .limit(1)
            )
            if runtime.auth_mode == "local" and user.storage_key == legacy_storage_key:
                tokens = bind_tenant(engine, runtime, voice)
                try:
                    return await call_next(request)
                finally:
                    reset_tenant(tokens)
            storage_key = user.storage_key
        workspace = tenant_workspaces.get(storage_key)
        tokens = bind_tenant(workspace.engine, workspace.settings, voice)
        try:
            await ensure_tenant_discovery(storage_key, workspace.settings)
            return await call_next(request)
        finally:
            reset_tenant(tokens)

    @app.get("/health/live")
    @app.get("/health")
    def health_live() -> dict[str, str]:
        return live_payload()

    @app.get("/health/ready", response_model=None)
    def health_ready() -> JSONResponse:
        failure = database_readiness(engine)
        if failure is None:
            return JSONResponse(status_code=200, content=ready_payload())
        return JSONResponse(status_code=503, content=not_ready_payload(failure))

    app.include_router(create_auth_router(runtime, engine))
    app.include_router(create_dashboard_router(runtime, engine))

    return app


app = create_app()
