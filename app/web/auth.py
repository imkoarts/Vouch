"""Passwordless account and account-bound voice onboarding API."""

import re
from typing import Annotated, Any

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app import __runtime_revision__, __version__
from app.config import Settings
from app.models import UserAccount, VoiceProfile
from app.services.auth import (
    AuthenticationError,
    AuthSessionService,
    LocalPasswordlessService,
    SmtpOtpDelivery,
    SupabasePasswordlessClient,
)
from app.services.configuration import ConfigurationService
from app.services.tenant_context import ContextSettings
from app.services.trend_discovery import build_x_read_client
from app.services.voice_profiles import (
    DeterministicVoiceAnalyzer,
    OpenAIVoiceAnalyzer,
    VoiceAnalyzer,
    VoiceProfileService,
)
from app.x_api.mock import MockXClient

SESSION_COOKIE = "vouch_session"
CSRF_COOKIE = "vouch_csrf"
_EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class EmailPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(min_length=3, max_length=320)


class VerifyPayload(EmailPayload):
    token: str = Field(min_length=6, max_length=128)


class VoiceProfilePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_type: str = Field(min_length=2, max_length=32)
    language: str = Field(default="en", min_length=2, max_length=16)
    response_preferences: list[str] = Field(default_factory=list, max_length=12)
    x_username: str | None = Field(default=None, max_length=50)


class VoiceOnboardingPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str | None = Field(default=None, max_length=80)
    x_username: str | None = Field(default=None, max_length=50)
    answers: dict[str, list[str]]


def _email(value: str) -> str:
    normalized = value.strip().casefold()
    if not _EMAIL.fullmatch(normalized):
        raise HTTPException(status_code=422, detail="Enter a valid email address")
    return normalized


def _serialize(profile: VoiceProfile | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    return {
        "account_type": profile.account_type,
        "x_username": profile.x_username,
        "language": profile.language,
        "tone": profile.tone,
        "response_preferences": profile.response_preferences,
        "vocabulary": profile.vocabulary,
        "sentence_patterns": profile.sentence_patterns,
        "humor_boundaries": profile.humor_boundaries,
        "banned_tendencies": profile.banned_tendencies,
        "guidance": profile.guidance,
        "sample_count": profile.sample_count,
        "source_digest": profile.source_digest,
        "analysis_provider": profile.analysis_provider,
        "analysis_model": profile.analysis_model,
        "analysis_metadata": profile.analysis_metadata,
        "onboarding_complete": bool(profile.analysis_metadata.get("onboarding_complete")),
        "display_name": str(profile.analysis_metadata.get("display_name") or ""),
    }


def create_auth_router(settings: Settings, engine: Engine) -> APIRouter:
    settings = ContextSettings(settings)  # type: ignore[assignment]
    router = APIRouter(prefix="/api")
    sessions = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    def provider() -> SupabasePasswordlessClient:
        if (
            settings.auth_mode != "supabase"
            or settings.supabase_url is None
            or settings.supabase_publishable_key is None
        ):
            raise HTTPException(status_code=409, detail="Passwordless email auth is not configured")
        return SupabasePasswordlessClient(
            base_url=settings.supabase_url,
            publishable_key=settings.supabase_publishable_key.get_secret_value(),
            timeout_seconds=settings.request_timeout_seconds,
        )

    def current_user(
        session_token: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
    ) -> UserAccount:
        with sessions() as database:
            user = AuthSessionService(database).resolve(session_token)
            if user is None:
                raise HTTPException(status_code=401, detail="Authentication required")
            database.expunge(user)
            return user

    def require_csrf(
        csrf_header: Annotated[str | None, Header(alias="X-CSRF-Token")] = None,
        session_token: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
    ) -> None:
        with sessions() as database:
            if not AuthSessionService(database).verify_csrf(session_token, csrf_header):
                raise HTTPException(status_code=403, detail="CSRF validation failed")

    CurrentUser = Annotated[UserAccount, Depends(current_user)]
    Csrf = Annotated[None, Depends(require_csrf)]

    @router.get("/auth/config")
    def auth_config() -> dict[str, Any]:
        return {
            "mode": settings.auth_mode,
            "passwordless_email": True,
            "delivery": "email",
            "email_configured": (
                settings.auth_mode == "supabase"
                or (
                    settings.local_otp_delivery == "smtp"
                    and settings.smtp_host is not None
                    and settings.smtp_from_email is not None
                )
            ),
            "version": __version__,
            "runtime_revision": __runtime_revision__,
        }

    @router.post("/auth/otp", status_code=202)
    async def request_otp(payload: EmailPayload) -> dict[str, Any]:
        email = _email(payload.email)
        if settings.auth_mode == "local":
            if settings.local_otp_delivery != "smtp":
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Email sign-in is not configured. Configure SMTP in the local environment."
                    ),
                )
            try:
                with sessions.begin() as database:
                    local = LocalPasswordlessService(
                        database,
                        minimum_interval_seconds=settings.otp_min_interval_seconds,
                        hourly_limit=settings.otp_hourly_limit,
                    )
                    code = local.request(email)
                    if settings.smtp_host is None or settings.smtp_from_email is None:
                        raise AuthenticationError("Email delivery is not configured")
                    await SmtpOtpDelivery(
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
                    ).send(email=email, token=code, lifetime_minutes=10)
            except AuthenticationError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            return {"accepted": True, "delivery": "email"}
        try:
            await provider().request_otp(email)
        except AuthenticationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"accepted": True, "delivery": "email"}

    @router.post("/auth/verify")
    async def verify_otp(payload: VerifyPayload, response: Response) -> dict[str, Any]:
        email = _email(payload.email)
        with sessions.begin() as database:
            try:
                if settings.auth_mode == "local":
                    identity = LocalPasswordlessService(database).verify(email, payload.token)
                    auth_provider = "local"
                else:
                    identity = await provider().verify_otp(email, payload.token)
                    auth_provider = "supabase"
            except AuthenticationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            user, issued = AuthSessionService(
                database, lifetime_hours=settings.auth_session_hours
            ).issue(identity, provider=auth_provider)
            user_id = user.id
        secure = settings.app_env == "production"
        response.set_cookie(
            SESSION_COOKIE,
            issued.token,
            max_age=issued.expires_in_seconds,
            httponly=True,
            secure=secure,
            samesite="strict",
            path="/",
        )
        response.set_cookie(
            CSRF_COOKIE,
            issued.csrf_token,
            max_age=issued.expires_in_seconds,
            httponly=False,
            secure=secure,
            samesite="strict",
            path="/",
        )
        return {"authenticated": True, "user_id": user_id}

    @router.get("/auth/session")
    def auth_session(user: CurrentUser) -> dict[str, Any]:
        with sessions() as database:
            profile = VoiceProfileService(database).get(user.id)
            metadata = profile.analysis_metadata if profile is not None else {}
        return {
            "authenticated": True,
            "user_id": user.id,
            "email": user.email,
            "display_name": str(metadata.get("display_name") or user.email.split("@", 1)[0]),
            "onboarding_complete": bool(metadata.get("onboarding_complete")),
        }

    @router.post("/auth/logout")
    def logout(
        response: Response,
        csrf: Csrf,
        session_token: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
    ) -> dict[str, bool]:
        del csrf
        with sessions.begin() as database:
            AuthSessionService(database).revoke(session_token)
        response.delete_cookie(SESSION_COOKIE, path="/")
        response.delete_cookie(CSRF_COOKIE, path="/")
        return {"logged_out": True}

    @router.get("/voice-profile")
    def read_voice_profile(user: CurrentUser) -> dict[str, Any]:
        with sessions() as database:
            return {"profile": _serialize(VoiceProfileService(database).get(user.id))}

    @router.put("/voice-profile")
    def save_voice_profile(
        payload: VoiceProfilePayload, user: CurrentUser, csrf: Csrf
    ) -> dict[str, Any]:
        del csrf
        with sessions.begin() as database:
            try:
                profile = VoiceProfileService(database).save_preferences(
                    user=database.get_one(UserAccount, user.id),
                    account_type=payload.account_type,
                    language=payload.language,
                    preferences=payload.response_preferences,
                    x_username=payload.x_username,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            rendered = _serialize(profile)
        return {"profile": rendered}

    @router.put("/voice-profile/onboarding")
    def save_voice_onboarding(
        payload: VoiceOnboardingPayload, user: CurrentUser, csrf: Csrf
    ) -> dict[str, Any]:
        del csrf
        with sessions.begin() as database:
            try:
                profile = VoiceProfileService(database).save_onboarding(
                    user=database.get_one(UserAccount, user.id),
                    answers=payload.answers,
                    display_name=payload.display_name,
                    x_username=payload.x_username,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            rendered = _serialize(profile)
        return {"profile": rendered}

    @router.post("/voice-profile/analyze")
    async def analyze_voice_profile(user: CurrentUser, csrf: Csrf) -> dict[str, Any]:
        del csrf
        configuration = ConfigurationService(settings.config_dir).load()
        reader = (
            MockXClient()
            if settings.mock_mode
            else build_x_read_client(
                settings,
                request_pacing=configuration.runtime.request_pacing,
            )
        )
        if settings.mock_mode:
            analyzer: VoiceAnalyzer = DeterministicVoiceAnalyzer()
        elif settings.openai_api_key is not None:
            analyzer = OpenAIVoiceAnalyzer(
                api_key=settings.openai_api_key.get_secret_value(), model=settings.openai_model
            )
        else:
            raise HTTPException(status_code=409, detail="OPENAI_API_KEY is required for analysis")
        with sessions.begin() as database:
            try:
                profile, evidence = await VoiceProfileService(database).analyze_x_profile(
                    user=database.get_one(UserAccount, user.id), reader=reader, analyzer=analyzer
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            rendered = _serialize(profile)
        return {"profile": rendered, "evidence": evidence.__dict__}

    return router
