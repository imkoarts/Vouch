"""Typer command-line interface shared by local operators and Docker."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
import uvicorn
from alembic.config import Config
from click import get_current_context
from sqlalchemy import select
from sqlalchemy.orm import Session

from alembic import command
from app.config import Settings, get_settings
from app.container import build_services
from app.database import get_engine, session_scope
from app.models import Draft, UserAccount, VoiceProfile
from app.models.enums import ContentType
from app.repositories.content import ContentRepository
from app.resources import resolve_alembic_config_path
from app.schemas.configuration import VoiceProfileConfiguration
from app.services.auth import AuthSessionService, LocalPasswordlessService, SmtpOtpDelivery
from app.services.benchmark import write_benchmark_manifest
from app.services.configuration import ConfigurationService
from app.services.doctor import DoctorService
from app.services.idea_collector import IdeaCollector
from app.services.judge_demo import run_judge_demo
from app.services.manual_generation import create_researched_topic_idea
from app.services.publishing_service import PublicationPreview
from app.services.tenant_context import active_settings, bind_tenant, reset_tenant
from app.services.tenant_storage import TenantWorkspaceRegistry
from app.services.trend_discovery import AutomaticDiscoveryService, build_x_read_client
from app.services.voice_profiles import (
    DeterministicVoiceAnalyzer,
    OpenAIVoiceAnalyzer,
    VoiceAnalyzer,
    VoiceProfileService,
)
from app.services.x_activity import (
    XActivityProcessor,
    XActivityRunner,
    XActivitySubscriptionManager,
    activity_requires_private_auth,
    build_activity_client,
    desired_activity_subscriptions,
)
from app.telegram.api import TelegramBotApi
from app.telegram.bot import TelegramReviewBot, notify_draft_once
from app.utils.logging import configure_logging

app = typer.Typer(
    name="vouch",
    help="Evidence-grounded social drafting with mandatory human approval.",
    no_args_is_help=True,
)
ideas_app = typer.Typer(help="Collect and inspect ranked content ideas.", no_args_is_help=True)
drafts_app = typer.Typer(
    help="Generate, review and approve drafts; publication remains fail-closed.",
    no_args_is_help=True,
)
app.add_typer(ideas_app, name="ideas")
telegram_app = typer.Typer(
    help="Run and inspect Telegram review integration.", no_args_is_help=True
)
activity_app = typer.Typer(
    help="Inspect and run optional X Activity subscriptions and stream processing.",
    no_args_is_help=True,
)
account_app = typer.Typer(
    help="Inspect the local account and configure its generation voice.", no_args_is_help=True
)
app.add_typer(drafts_app, name="drafts")
app.add_typer(telegram_app, name="telegram")
app.add_typer(activity_app, name="activity")
app.add_typer(account_app, name="account")

_CLI_TENANT_REGISTRY: TenantWorkspaceRegistry | None = None
_CLI_TENANT_TOKENS: tuple[Any, Any, Any] | None = None


@app.callback()
def cli_context(context: typer.Context) -> None:
    """Release any account workspace bound for the selected CLI command."""

    context.call_on_close(_reset_cli_workspace)


def _reset_cli_workspace() -> None:
    global _CLI_TENANT_REGISTRY, _CLI_TENANT_TOKENS

    if _CLI_TENANT_TOKENS is not None:
        reset_tenant(_CLI_TENANT_TOKENS)
        _CLI_TENANT_TOKENS = None
    if _CLI_TENANT_REGISTRY is not None:
        _CLI_TENANT_REGISTRY.dispose()
        _CLI_TENANT_REGISTRY = None


def _bind_cli_workspace(settings: Settings) -> Settings:
    """Route content commands to the account selected by ``account login``."""

    global _CLI_TENANT_REGISTRY, _CLI_TENANT_TOKENS

    if _CLI_TENANT_TOKENS is not None:
        return active_settings(settings)
    context = get_current_context(silent=True)
    command_path = context.command_path.casefold() if context is not None else ""
    parts = set(command_path.split())
    if not (parts & {"ideas", "drafts", "telegram", "activity", "doctor"}):
        return settings
    marker = _active_account_path(settings)
    if not marker.is_file():
        return settings
    storage_key = marker.read_text(encoding="utf-8").strip()
    with Session(get_engine()) as database:
        user = database.scalar(
            select(UserAccount).where(
                UserAccount.auth_provider == "local",
                UserAccount.storage_key == storage_key,
            )
        )
        legacy_storage_key = database.scalar(
            select(UserAccount.storage_key)
            .where(UserAccount.auth_provider == "local")
            .order_by(UserAccount.created_at, UserAccount.id)
            .limit(1)
        )
        if user is None or user.storage_key == legacy_storage_key:
            return settings
        stored_voice = database.scalar(select(VoiceProfile).where(VoiceProfile.user_id == user.id))
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
    _CLI_TENANT_REGISTRY = TenantWorkspaceRegistry(settings)
    workspace = _CLI_TENANT_REGISTRY.get(storage_key)
    _CLI_TENANT_TOKENS = bind_tenant(workspace.engine, workspace.settings, voice)
    return workspace.settings


def _settings() -> Settings:
    settings = _bind_cli_workspace(get_settings())
    configure_logging(settings.log_level, settings.logs_dir, color=settings.color_logs)
    settings.ensure_directories()
    return settings


def _fail(error: Exception) -> NoReturn:
    typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _alembic_config(settings: Settings) -> Config:
    config_path = resolve_alembic_config_path()
    config = Config(str(config_path))
    config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))
    return config


def _active_account_path(settings: Settings) -> Path:
    return (settings.data_dir / ".active_account").resolve()


def _remember_active_account(settings: Settings, storage_key: str) -> None:
    path = _active_account_path(settings)
    if settings.data_dir.resolve() not in path.parents:
        raise ValueError("Active account path escaped the configured data directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(storage_key + "\n", encoding="utf-8")


def _local_account(session: Session, settings: Settings | None = None) -> UserAccount:
    storage_key = ""
    if settings is not None:
        path = _active_account_path(settings)
        if path.is_file():
            storage_key = path.read_text(encoding="utf-8").strip()
    statement = select(UserAccount).where(UserAccount.auth_provider == "local")
    if storage_key:
        statement = statement.where(UserAccount.storage_key == storage_key)
    else:
        statement = statement.order_by(UserAccount.created_at, UserAccount.id)
    user = session.scalar(statement.limit(1))
    if user is None:
        raise ValueError(
            "No local account exists. Run `vouch account login --email you@example.com` first."
        )
    return user


@account_app.command("login")
def account_login(
    email: Annotated[str, typer.Option(help="Email for the local Vouch workspace.")],
    token: Annotated[
        str | None,
        typer.Option(help="Code received by email; omit it to request a new code."),
    ] = None,
) -> None:
    """Create or sign in to the loopback local account with a one-time code."""

    settings = _settings()
    if settings.auth_mode != "local":
        _fail(ValueError("Use the web email flow when AUTH_MODE=supabase"))
    normalized = email.strip().casefold()
    if "@" not in normalized or len(normalized) > 320:
        _fail(ValueError("Enter a valid email address"))
    if token is None:
        if (
            settings.local_otp_delivery != "smtp"
            or settings.smtp_host is None
            or settings.smtp_from_email is None
        ):
            _fail(ValueError("SMTP email delivery is not configured"))
        try:
            with session_scope() as session:
                code = LocalPasswordlessService(
                    session,
                    minimum_interval_seconds=settings.otp_min_interval_seconds,
                    hourly_limit=settings.otp_hourly_limit,
                ).request(normalized)
                asyncio.run(
                    SmtpOtpDelivery(
                        host=settings.smtp_host,
                        port=settings.smtp_port,
                        from_email=settings.smtp_from_email,
                        username=settings.smtp_username,
                        password=(
                            settings.smtp_password.get_secret_value()
                            if settings.smtp_password is not None
                            else None
                        ),
                        starttls=settings.smtp_starttls,
                        timeout_seconds=settings.request_timeout_seconds,
                    ).send(email=normalized, token=code, lifetime_minutes=10)
                )
        except ValueError as exc:
            _fail(exc)
        typer.echo("One-time code sent to your email.")
        token = typer.prompt("Enter the code")
    try:
        with session_scope() as session:
            identity = LocalPasswordlessService(session).verify(normalized, token)
            user, _ = AuthSessionService(session, lifetime_hours=settings.auth_session_hours).issue(
                identity, provider="local"
            )
            account_id = user.id
            storage_key = user.storage_key
    except ValueError as exc:
        _fail(exc)
    _remember_active_account(settings, storage_key)
    typer.secho(f"Local account {account_id} is ready.", fg=typer.colors.GREEN)


@account_app.command("status")
def account_status() -> None:
    """Show account/onboarding state without exposing credentials."""

    settings = _settings()
    try:
        with session_scope() as session:
            user = _local_account(session, settings)
            profile = VoiceProfileService(session).get(user.id)
            payload = {
                "auth_mode": settings.auth_mode,
                "account_id": user.id,
                "email": user.email,
                "voice_configured": profile is not None,
                "voice_analyzed": bool(profile and profile.sample_count),
            }
    except ValueError as exc:
        _fail(exc)
    typer.echo(json.dumps(payload, indent=2))


@account_app.command("voice-show")
def voice_show() -> None:
    """Show the local account's non-sensitive voice profile."""

    settings = _settings()
    try:
        with session_scope() as session:
            user = _local_account(session, settings)
            profile = VoiceProfileService(session).get(user.id)
            if profile is None:
                typer.echo("Voice profile is not configured.")
                return
            payload = {
                "account_type": profile.account_type,
                "language": profile.language,
                "response_preferences": profile.response_preferences,
                "x_username": profile.x_username,
                "tone": profile.tone,
                "guidance": profile.guidance,
                "sample_count": profile.sample_count,
            }
    except ValueError as exc:
        _fail(exc)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@account_app.command("voice-set")
def voice_set(
    account_type: Annotated[
        str,
        typer.Option(help="personal, news, store, company, community or other"),
    ] = "personal",
    language: Annotated[str, typer.Option(help="Generation language code.")] = "en",
    preferences: Annotated[
        str,
        typer.Option(help="Comma-separated response moves, for example direct,question,dry_humor."),
    ] = "direct,question",
    x_username: Annotated[str | None, typer.Option(help="Optional X username to analyze.")] = None,
) -> None:
    """Configure the local voice used by web, desktop and CLI generation."""

    settings = _settings()
    selected = tuple(value.strip() for value in preferences.split(",") if value.strip())
    try:
        with session_scope() as session:
            user = _local_account(session, settings)
            profile = VoiceProfileService(session).save_preferences(
                user=user,
                account_type=account_type,
                language=language,
                preferences=selected,
                x_username=x_username,
            )
            profile_id = profile.id
    except ValueError as exc:
        _fail(exc)
    typer.secho(f"Voice profile {profile_id} saved.", fg=typer.colors.GREEN)


@account_app.command("voice-analyze")
def voice_analyze() -> None:
    """Analyze up to 50 own X posts and apply the resulting voice to generation."""

    settings = _settings()

    async def run_analysis() -> dict[str, object]:
        configuration = ConfigurationService(settings.config_dir).load()
        reader = build_x_read_client(settings, request_pacing=configuration.runtime.request_pacing)
        analyzer: VoiceAnalyzer
        if settings.mock_mode:
            analyzer = DeterministicVoiceAnalyzer()
        elif settings.openai_api_key is not None:
            analyzer = OpenAIVoiceAnalyzer(
                api_key=settings.openai_api_key.get_secret_value(),
                model=configuration.runtime.providers.openai.model,
            )
        else:
            raise ValueError("OPENAI_API_KEY is required for live voice analysis")
        with session_scope() as session:
            user = _local_account(session, settings)
            profile, evidence = await VoiceProfileService(session).analyze_x_profile(
                user=user, reader=reader, analyzer=analyzer
            )
            return {
                "profile_id": profile.id,
                "sample_count": evidence.sample_count,
                "provider": evidence.provider,
                "model": evidence.model,
            }

    try:
        result = asyncio.run(run_analysis())
    except Exception as exc:
        _fail(exc)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("init")
def initialize() -> None:
    """Create directories and apply all local database migrations."""

    settings = _settings()
    try:
        command.upgrade(_alembic_config(settings), "head")
    except Exception as exc:
        _fail(exc)
    typer.secho("Database migrated to head.", fg=typer.colors.GREEN)


@app.command()
def doctor() -> None:
    """Run read-only configuration, filesystem, database and safety checks."""

    settings = _settings()
    checks = DoctorService(settings, get_engine()).run()
    for check in checks:
        color = {
            "PASS": typer.colors.GREEN,
            "WARN": typer.colors.YELLOW,
            "FAIL": typer.colors.RED,
            "SKIP": typer.colors.BLUE,
        }[check.status]
        typer.secho(f"{check.status:4} {check.name}: {check.message}", fg=color)
    if any(check.status == "FAIL" for check in checks):
        raise typer.Exit(code=1)


@app.command()
def demo(
    output_dir: Annotated[
        str, typer.Option(help="Fresh directory for isolated synthetic judge artifacts.")
    ] = "data/judge",
) -> None:
    """Run the safe offline judge flow; export a preview and perform zero remote writes."""

    settings = _settings()
    try:
        report = asyncio.run(run_judge_demo(Path(output_dir), config_dir=settings.config_dir))
    except Exception as exc:
        _fail(exc)
    typer.echo(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


@app.command()
def benchmark(
    output_dir: Annotated[
        str, typer.Option(help="Directory for the synthetic suite and measurement report.")
    ] = "benchmarks",
) -> None:
    """Export six benchmark templates as 60 deterministic entries; live evaluation pending."""

    settings = _settings()
    report = write_benchmark_manifest(
        Path(output_dir), openai_key_available=settings.openai_api_key is not None
    )
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Override the configured bind host.")] = None,
    port: Annotated[int | None, typer.Option(help="Override the configured port.")] = None,
) -> None:
    """Start the local FastAPI interface (loopback by default)."""

    settings = _settings()
    selected_host = host or settings.app_host
    if selected_host != settings.app_host:
        try:
            Settings(**{**settings.model_dump(), "app_host": selected_host})
        except Exception as exc:
            _fail(exc)
    uvicorn.run(
        "app.main:app",
        host=selected_host,
        port=port or settings.app_port,
        reload=False,
        log_config=None,
    )


@ideas_app.command("collect")
def ideas_collect() -> None:
    """Collect synthetic sources and deterministically rank ideas in mock mode."""

    settings = _settings()
    if not settings.mock_mode:
        _fail(RuntimeError("Live source collection is disabled in the stabilization build"))
    try:
        with session_scope() as session:
            source_configuration = ConfigurationService(settings.config_dir).load_sources()
            identifiers = IdeaCollector(
                session,
                source_configuration=source_configuration,
                similarity_threshold=settings.similarity_threshold,
            ).collect_mock()
            typer.echo(json.dumps({"idea_ids": identifiers}, ensure_ascii=False))
    except Exception as exc:
        _fail(exc)


@ideas_app.command("discover")
def ideas_discover() -> None:
    """Read a budget-bounded X sample and generate one review draft."""

    settings = _settings()
    try:
        outcome = asyncio.run(AutomaticDiscoveryService(settings).run_once(actor="cli_discover"))
        payload = {
            "status": outcome.status,
            "draft_id": outcome.draft_id,
            "idea_id": outcome.idea_id,
            "fetched_posts": outcome.fetched_posts,
            "home_posts": outcome.home_posts,
            "search_posts": outcome.search_posts,
            "trends": list(outcome.trends),
            "maximum_estimated_x_cost_usd": str(outcome.estimated_x_cost_usd),
            "message": outcome.message,
        }
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        if outcome.draft_id is not None:
            asyncio.run(_notify_if_enabled(settings, outcome.draft_id))
    except Exception as exc:
        _fail(exc)


@ideas_app.command("list")
def ideas_list() -> None:
    """List ranked ideas from highest to lowest score."""

    _settings()
    try:
        with session_scope() as session:
            rows = [
                {
                    "id": idea.id,
                    "score": round(idea.score, 4),
                    "title": idea.title,
                    "format": idea.recommended_format.value if idea.recommended_format else None,
                }
                for idea in ContentRepository(session).list_ideas()
            ]
            typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
    except Exception as exc:
        _fail(exc)


async def _notify_if_enabled(settings: Settings, draft_id: str) -> None:
    try:
        await notify_draft_once(settings, draft_id)
    except Exception as exc:
        typer.secho(
            f"Telegram notification skipped: {type(exc).__name__}",
            fg=typer.colors.YELLOW,
            err=True,
        )


@drafts_app.command("generate")
def drafts_generate(
    idea_id: Annotated[str, typer.Option("--idea", help="Ranked idea UUID.")],
) -> None:
    """Generate up to three grounded provider variants and persist the selected draft bundle."""

    settings = _settings()
    try:
        with session_scope() as session:
            services = build_services(session, settings)
            draft = asyncio.run(services.drafts.generate_from_idea(idea_id))
            if draft is None:
                payload = {
                    "draft_id": None,
                    "status": "no_post_needed",
                    "reason_code": "NO_GROUNDED_CONTRIBUTION",
                }
            else:
                payload = {
                    "draft_id": draft.id,
                    "status": draft.status.value,
                    "content_hash": draft.current_content_hash,
                    "artifact_path": draft.artifact_path,
                }
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        if payload["draft_id"] is not None:
            asyncio.run(_notify_if_enabled(settings, str(payload["draft_id"])))
    except Exception as exc:
        _fail(exc)


@drafts_app.command("generate-topic")
def drafts_generate_topic(
    topic: Annotated[str, typer.Option("--topic", help="Topic for one new review draft.")],
    content_type: Annotated[
        ContentType, typer.Option("--format", help="short_post, long_post or thread")
    ] = ContentType.SHORT_POST,
) -> None:
    """Research a topic through a bounded X search, then generate one review draft."""

    settings = _settings()
    if content_type not in {
        ContentType.SHORT_POST,
        ContentType.LONG_POST,
        ContentType.THREAD,
    }:
        _fail(ValueError("generate-topic supports short_post, long_post or thread"))
    try:
        idea_id = asyncio.run(
            create_researched_topic_idea(
                settings,
                topic_text=topic,
                content_type=content_type,
                actor="cli_custom_topic",
            )
        )
        with session_scope() as session:
            services = build_services(session, settings)
            draft = asyncio.run(services.drafts.generate_from_idea(idea_id))
            draft_id = draft.id if draft is not None else None
            payload = (
                {
                    "draft_id": draft.id,
                    "status": draft.status.value,
                    "content_hash": draft.current_content_hash,
                    "artifact_path": draft.artifact_path,
                }
                if draft is not None
                else {
                    "draft_id": None,
                    "status": "no_post_needed",
                    "reason_code": "NO_GROUNDED_CONTRIBUTION",
                }
            )
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        if draft_id is not None:
            asyncio.run(_notify_if_enabled(settings, draft_id))
    except Exception as exc:
        _fail(exc)


@drafts_app.command("show")
def drafts_show(draft_id: str) -> None:
    """Show the current draft version and its approval-relevant metadata."""

    settings = _settings()
    try:
        with session_scope() as session:
            services = build_services(session, settings)
            services.drafts.reconcile(draft_id)
            draft = services.drafts.get(draft_id)
            version = services.drafts.current_version(draft)
            typer.echo(
                json.dumps(
                    {
                        "id": draft.id,
                        "status": draft.status.value,
                        "version": version.version_number,
                        "content_hash": draft.current_content_hash,
                        "content": version.content,
                        "fact_check_status": draft.fact_check_status.value,
                        "safety_flags": draft.blocking_safety_flags,
                        "artifact_path": draft.artifact_path,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    except Exception as exc:
        _fail(exc)


@drafts_app.command("approve")
def drafts_approve(
    draft_id: str,
    facts_confirmed: Annotated[
        bool, typer.Option(help="Explicitly mark required facts as verified.")
    ] = False,
) -> None:
    """Approve the exact current version, text hash and media manifest."""

    settings = _settings()
    try:
        with session_scope() as session:
            services = build_services(session, settings)
            approval = services.approvals.approve(draft_id, facts_confirmed=facts_confirmed)
            typer.echo(
                json.dumps(
                    {
                        "approval_id": approval.id,
                        "draft_id": approval.draft_id,
                        "content_hash": approval.content_hash,
                    },
                    indent=2,
                )
            )
    except Exception as exc:
        _fail(exc)


@drafts_app.command("reinspect")
def drafts_reinspect(draft_id: str) -> None:
    """Complete live semantic and factual inspection for the current edited version."""

    settings = _settings()
    try:
        with session_scope() as session:
            services = build_services(session, settings)
            draft = asyncio.run(services.drafts.reinspect_current_version(draft_id))
            version = services.drafts.current_version(draft)
            factual = version.generation_metadata.get("factual_inspection", {})
            editorial = version.generation_metadata.get("editorial_report", {})
            typer.echo(
                json.dumps(
                    {
                        "draft_id": draft.id,
                        "version": version.version_number,
                        "status": draft.status.value,
                        "factual_status": factual.get("status")
                        if isinstance(factual, dict)
                        else None,
                        "editorial_status": editorial.get("status")
                        if isinstance(editorial, dict)
                        else None,
                        "safety_flags": draft.blocking_safety_flags,
                    },
                    indent=2,
                )
            )
    except Exception as exc:
        _fail(exc)


@drafts_app.command("regenerate")
def drafts_regenerate(
    draft_id: str,
    feedback: Annotated[str | None, typer.Option(help="Human regeneration feedback.")] = None,
) -> None:
    """Append a new generated version while retaining all prior versions."""

    settings = _settings()
    try:
        with session_scope() as session:
            services = build_services(session, settings)
            draft = asyncio.run(services.drafts.regenerate(draft_id, feedback=feedback))
            typer.echo(
                json.dumps(
                    {
                        "draft_id": draft.id,
                        "status": draft.status.value,
                        "version": draft.current_version_number,
                        "content_hash": draft.current_content_hash,
                    },
                    indent=2,
                )
            )
    except Exception as exc:
        _fail(exc)


def _print_preview(preview: PublicationPreview) -> None:
    typer.echo("\n--- PUBLICATION PREVIEW ---")
    for position, part in enumerate(preview.parts, start=1):
        typer.echo(f"[{position}] {part}")
    typer.echo(
        json.dumps(
            {
                "weighted_lengths": preview.weighted_lengths,
                "media_manifest": preview.media_manifest,
                "source": preview.source,
                "facts": preview.facts,
                "estimated_cost_usd": (
                    str(preview.estimated_cost_usd)
                    if preview.estimated_cost_usd is not None
                    else "not configured"
                ),
                "account": {
                    "id": preview.account_id,
                    "username": preview.account_username,
                },
                "action": preview.action,
                "required_confirmation": preview.confirmation_phrase,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@drafts_app.command("publish")
def drafts_publish(
    draft_id: str,
    confirm: Annotated[
        str | None,
        typer.Option(
            "--confirm",
            help="Exact phrase: PUBLISH <draft_id> <12-character-hash-prefix>.",
        ),
    ] = None,
) -> None:
    """Run all gates and publish through the configured manual X writer."""

    settings = _settings()
    try:
        with session_scope() as session:
            services = build_services(session, settings)
            preview = asyncio.run(services.publishing.preview(draft_id))
        _print_preview(preview)
        phrase = confirm or typer.prompt("Enter the exact confirmation phrase")
        with session_scope() as session:
            services = build_services(session, settings)
            records = asyncio.run(
                services.publishing.publish(
                    draft_id,
                    confirmation_phrase=phrase,
                )
            )
            typer.echo(
                json.dumps(
                    {
                        "mock": True,
                        "publication_ids": [record.id for record in records],
                        "x_post_ids": [record.x_post_id for record in records],
                    },
                    indent=2,
                )
            )
    except Exception as exc:
        _fail(exc)


@drafts_app.command("list")
def drafts_list() -> None:
    """List local drafts."""

    _settings()
    try:
        with session_scope() as session:
            rows = [
                {"id": draft.id, "status": draft.status.value}
                for draft in session.scalars(select(Draft).order_by(Draft.created_at.desc()))
            ]
            typer.echo(json.dumps(rows, indent=2))
    except Exception as exc:
        _fail(exc)


@drafts_app.command("rebuild-artifacts")
def drafts_rebuild_artifacts(
    draft_id: Annotated[
        str | None,
        typer.Argument(help="Draft UUID whose DB-derived files must be rebuilt."),
    ] = None,
    all_drafts: Annotated[
        bool,
        typer.Option("--all", help="Rebuild projections for every persisted draft."),
    ] = False,
) -> None:
    """Rebuild human-readable artifacts from the authoritative database."""

    settings = _settings()
    if (draft_id is None) == (not all_drafts):
        _fail(ValueError("Provide exactly one draft ID or --all"))
    try:
        with session_scope() as session:
            services = build_services(session, settings)
            draft_ids = (
                list(session.scalars(select(Draft.id).order_by(Draft.id)).all())
                if all_drafts
                else [draft_id]
            )
            rebuilt = [
                services.projections.rebuild(identifier).draft_id
                for identifier in draft_ids
                if identifier is not None
            ]
            typer.echo(json.dumps({"rebuilt_draft_ids": rebuilt}, indent=2))
    except Exception as exc:
        _fail(exc)


@telegram_app.command("run")
def telegram_run() -> None:
    """Run Telegram long polling in the foreground."""

    settings = _settings()
    configuration = ConfigurationService(settings.config_dir).load()
    runtime = configuration.runtime.telegram
    if not runtime.enabled:
        _fail(RuntimeError("Telegram is disabled in config/runtime.yml"))
    telegram_token = settings.telegram_bot_token
    if telegram_token is None:
        _fail(RuntimeError("TELEGRAM_BOT_TOKEN is missing from .env"))

    async def runner() -> None:
        api = TelegramBotApi(
            telegram_token.get_secret_value(),
            timeout_seconds=runtime.request_timeout_seconds,
            proxy_url=settings.outbound_proxy_url,
        )
        try:
            await TelegramReviewBot(settings, api).run_forever()
        finally:
            await api.close()

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        typer.echo("Telegram bot stopped.")


@telegram_app.command("notify")
def telegram_notify(draft_id: str) -> None:
    """Send one existing draft to the configured Telegram review chat."""

    settings = _settings()
    try:
        asyncio.run(notify_draft_once(settings, draft_id, force=True))
    except Exception as exc:
        _fail(exc)


@activity_app.command("doctor")
def activity_doctor() -> None:
    """Validate local X Activity configuration without making a network request."""

    settings = _settings()
    try:
        runtime = ConfigurationService(settings.config_dir).load_runtime().x_activity
        private = activity_requires_private_auth(runtime)
        if runtime.enabled:
            missing = settings.missing_x_activity_credentials(private=private)
            desired = desired_activity_subscriptions(settings, runtime)
        else:
            missing = []
            desired = ()
        payload = {
            "enabled": runtime.enabled,
            "autostart": runtime.autostart,
            "private_events_configured": private,
            "credentials_ready": not missing if runtime.enabled else None,
            "missing": missing,
            "desired_subscriptions": [
                {
                    "event_type": item.event_type,
                    "user_id": item.user_id,
                    "tag": item.tag,
                }
                for item in desired
            ],
            "automatic_subscription_writes": False,
            "status": "disabled" if not runtime.enabled else "configured",
        }
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as exc:
        _fail(exc)


@activity_app.command("plan")
def activity_plan() -> None:
    """Compare desired subscriptions with X without changing remote state."""

    settings = _settings()
    if settings.mock_mode:
        _fail(RuntimeError("Set MOCK_MODE=false before calling the live X Activity API"))
    try:
        runtime = ConfigurationService(settings.config_dir).load_runtime().x_activity
        client = build_activity_client(settings, runtime)
        plan = asyncio.run(
            XActivitySubscriptionManager(client, settings=settings, configuration=runtime).plan()
        )
        typer.echo(
            json.dumps(
                {
                    "desired": [item.request_payload() for item in plan.desired],
                    "existing": [
                        {
                            **item.request_payload(),
                            "subscription_id": item.subscription_id,
                        }
                        for item in plan.existing
                    ],
                    "missing": [item.request_payload() for item in plan.missing],
                    "unmanaged_existing": [
                        {
                            **item.request_payload(),
                            "subscription_id": item.subscription_id,
                        }
                        for item in plan.unmanaged_existing
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as exc:
        _fail(exc)


@activity_app.command("sync")
def activity_sync(
    confirm: Annotated[
        str | None,
        typer.Option(
            "--confirm",
            help="Exact phrase: SYNC X ACTIVITY SUBSCRIPTIONS.",
        ),
    ] = None,
) -> None:
    """Create only missing configured subscriptions after explicit confirmation."""

    settings = _settings()
    if settings.mock_mode:
        _fail(RuntimeError("Set MOCK_MODE=false before changing X Activity subscriptions"))
    phrase = confirm or typer.prompt("Enter the exact confirmation phrase")
    try:
        runtime = ConfigurationService(settings.config_dir).load_runtime().x_activity
        client = build_activity_client(settings, runtime)
        created = asyncio.run(
            XActivitySubscriptionManager(
                client, settings=settings, configuration=runtime
            ).create_missing(confirmation_phrase=phrase)
        )
        typer.echo(
            json.dumps(
                {
                    "created": [
                        {
                            **item.request_payload(),
                            "subscription_id": item.subscription_id,
                        }
                        for item in created
                    ],
                    "deleted": [],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as exc:
        _fail(exc)


@activity_app.command("listen")
def activity_listen() -> None:
    """Run the persistent X Activity stream in the foreground."""

    settings = _settings()
    if settings.mock_mode:
        _fail(RuntimeError("Set MOCK_MODE=false before opening the X Activity stream"))
    configuration = ConfigurationService(settings.config_dir).load()
    runtime = configuration.runtime.x_activity
    if not runtime.enabled:
        _fail(RuntimeError("X Activity is disabled in config/runtime.yml"))

    async def runner() -> None:
        activity_client = build_activity_client(settings, runtime)
        telegram_api: TelegramBotApi | None = None
        telegram_bot: TelegramReviewBot | None = None
        telegram = configuration.runtime.telegram
        if runtime.notify_telegram and telegram.enabled and settings.telegram_bot_token is not None:
            telegram_api = TelegramBotApi(
                settings.telegram_bot_token.get_secret_value(),
                timeout_seconds=telegram.request_timeout_seconds,
                proxy_url=settings.outbound_proxy_url,
            )
            telegram_bot = TelegramReviewBot(settings, telegram_api)

        async def handle(envelope: Mapping[str, Any]) -> None:
            with session_scope() as session:
                services = build_services(session, settings)
                result = await XActivityProcessor(
                    session,
                    settings=settings,
                    runtime=runtime,
                    sources=services.configuration.sources,
                    drafts=services.drafts,
                ).process(envelope)
                draft_id = result.draft_id
                notification = result.notification
            if telegram_bot is not None and draft_id and notification is not None:
                await telegram_bot.notify_draft(draft_id)
            elif (
                telegram_api is not None
                and telegram.effective_review_chat_id is not None
                and notification is not None
            ):
                await telegram_api.send_message(telegram.effective_review_chat_id, notification)
            typer.echo(
                json.dumps(
                    {
                        "event_uuid": result.event_uuid,
                        "event_type": result.event_type,
                        "status": result.status,
                        "draft_id": result.draft_id,
                    },
                    ensure_ascii=False,
                )
            )

        async def report_error(exc: Exception) -> None:
            typer.secho(
                f"X Activity reconnect after {type(exc).__name__}",
                fg=typer.colors.YELLOW,
                err=True,
            )

        try:
            await XActivityRunner(
                activity_client,
                configuration=runtime,
                handler=handle,
                error_handler=report_error,
            ).run_forever()
        finally:
            if telegram_api is not None:
                await telegram_api.close()

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        typer.echo("X Activity stream stopped.")


def run() -> None:
    app()


if __name__ == "__main__":
    run()
