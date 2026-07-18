"""Passwordless auth and voice profiles remain bounded and account-owned."""

from __future__ import annotations

import shutil
import smtplib
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.container import build_services
from app.models import AuthSession, Base, LocalOtpChallenge, UserAccount
from app.services.auth import (
    AuthenticationError,
    AuthSessionService,
    LocalPasswordlessService,
    SmtpOtpDelivery,
    SupabasePasswordlessClient,
    VerifiedIdentity,
)
from app.services.voice_profiles import DeterministicVoiceAnalyzer, VoiceProfileService
from app.web.auth import create_auth_router


@pytest.mark.asyncio
async def test_supabase_passwordless_adapter_uses_otp_and_verify_endpoints() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["apikey"] == "synthetic-publishable-key"
        if request.url.path.endswith("/otp"):
            return httpx.Response(200, json={})
        return httpx.Response(
            200,
            json={"user": {"id": "subject-1", "email": "user@example.test"}},
        )

    client = SupabasePasswordlessClient(
        base_url="https://project.example.test",
        publishable_key="synthetic-publishable-key",
        transport=httpx.MockTransport(handler),
    )

    await client.request_otp("user@example.test")
    identity = await client.verify_otp("user@example.test", "123456")

    assert identity == VerifiedIdentity(subject="subject-1", email="user@example.test")
    assert [request.url.path for request in requests] == ["/auth/v1/otp", "/auth/v1/verify"]


def test_sessions_are_hashed_and_profiles_are_account_bound() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        auth = AuthSessionService(session)
        first, issued = auth.issue(
            VerifiedIdentity(subject="first", email="first@example.test"), provider="supabase"
        )
        second, _ = auth.issue(
            VerifiedIdentity(subject="second", email="second@example.test"), provider="supabase"
        )
        profile_service = VoiceProfileService(session)
        profile_service.save_preferences(
            user=first,
            account_type="personal",
            language="en",
            preferences=["direct", "dry_humor"],
        )
        session.flush()

        assert auth.resolve(issued.token).id == first.id
        assert profile_service.get(first.id) is not None
        assert profile_service.get(second.id) is None
        assert all(row.token_hash != issued.token for row in session.query(AuthSession).all())
    engine.dispose()


def _onboarding_answers() -> dict[str, list[str]]:
    return {
        "response_instinct": ["direct_answer"],
        "disagreement_style": ["state_it_directly", "show_evidence"],
        "reasoning_shape": ["conclusion_first"],
        "certainty_style": ["calibrated"],
        "humor_style": ["dry"],
        "sarcasm_boundary": ["safe_targets_only"],
        "message_rhythm": ["conversational"],
        "voice_qualities": ["confident", "friendly", "sharp"],
        "audience_relationship": ["peers"],
        "feedback_directness": ["direct"],
    }


def test_ten_answer_onboarding_compiles_and_persists_account_voice() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        user = UserAccount(
            auth_provider="local",
            auth_subject="questionnaire-owner",
            email="questionnaire@example.test",
        )
        session.add(user)
        session.flush()

        profile = VoiceProfileService(session).save_onboarding(
            user=user,
            answers=_onboarding_answers(),
            display_name="Synthetic User",
        )
        session.commit()
        persisted = VoiceProfileService(session).get(user.id)

        assert persisted is not None and persisted.id == profile.id
        assert persisted.analysis_metadata["onboarding_complete"] is True
        assert persisted.analysis_metadata["display_name"] == "Synthetic User"
        assert persisted.response_preferences == ["direct", "dry_humor", "sarcasm"]
        assert "calibrated certainty" in persisted.guidance
        assert "synthetic" not in persisted.guidance.casefold()
    engine.dispose()


def test_onboarding_requires_all_questions_and_limits_voice_qualities() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        user = UserAccount(
            auth_provider="local",
            auth_subject="invalid-questionnaire",
            email="invalid@example.test",
        )
        session.add(user)
        session.flush()
        service = VoiceProfileService(session)
        incomplete = _onboarding_answers()
        incomplete.pop("humor_style")
        with pytest.raises(ValueError, match="ten"):
            service.save_onboarding(user=user, answers=incomplete)
        too_many = _onboarding_answers()
        too_many["voice_qualities"] = ["calm", "confident", "friendly", "sharp"]
        with pytest.raises(ValueError, match="three"):
            service.save_onboarding(user=user, answers=too_many)
    engine.dispose()


def test_local_passwordless_code_is_hashed_single_use_and_bounded() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        service = LocalPasswordlessService(session)
        token = service.request("owner@example.test")
        challenge = session.query(LocalOtpChallenge).one()

        assert len(token) == 6 and token.isdigit()
        assert challenge.token_hash != token
        assert "owner@example.test" not in challenge.email_hash
        with pytest.raises(AuthenticationError, match="invalid"):
            service.verify("owner@example.test", "000000" if token != "000000" else "999999")
        identity = service.verify("owner@example.test", token)
        assert identity.email == "owner@example.test"
        with pytest.raises(AuthenticationError, match="missing or expired"):
            service.verify("owner@example.test", token)
    engine.dispose()


def test_local_passwordless_requests_are_rate_limited() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        service = LocalPasswordlessService(session, minimum_interval_seconds=60)
        service.request("owner@example.test")
        with pytest.raises(AuthenticationError, match="wait"):
            service.request("owner@example.test")
    engine.dispose()


@pytest.mark.asyncio
async def test_smtp_delivery_uses_starttls_and_never_logs_or_persists_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_ssl_context(*, cafile: str) -> object:
        assert cafile.endswith("synthetic-ca.pem")
        calls.append("ca_bundle")
        return object()

    class FakeSmtp:
        def __init__(self, host: str, port: int, *, timeout: float) -> None:
            assert (host, port, timeout) == ("smtp.example.test", 587, 12.0)

        def __enter__(self) -> FakeSmtp:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def starttls(self, *, context: object) -> None:
            assert context is not None
            calls.append("starttls")

        def login(self, username: str, password: str) -> None:
            assert (username, password) == ("mailer", "synthetic-secret")
            calls.append("login")

        def send_message(self, message: object) -> None:
            assert "123456" in str(message)
            calls.append("send")

    monkeypatch.setattr("app.services.auth.certifi.where", lambda: "synthetic-ca.pem")
    monkeypatch.setattr("app.services.auth.ssl.create_default_context", fake_ssl_context)
    monkeypatch.setattr("app.services.auth.smtplib.SMTP", FakeSmtp)
    await SmtpOtpDelivery(
        host="smtp.example.test",
        port=587,
        from_email="vouch@example.test",
        username="mailer",
        password="synthetic-secret",
        timeout_seconds=12.0,
    ).send(email="owner@example.test", token="123456", lifetime_minutes=10)
    assert calls == ["ca_bundle", "starttls", "login", "send"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_error", "safe_message"),
    [
        (
            smtplib.SMTPAuthenticationError(535, b"provider detail must stay private"),
            "SMTP authentication failed",
        ),
        (
            smtplib.SMTPDataError(403, b"provider detail must stay private"),
            "configured From address",
        ),
        (
            smtplib.SMTPRecipientsRefused(
                {"owner@example.test": (550, b"provider detail must stay private")}
            ),
            "rejected the recipient",
        ),
    ],
)
async def test_smtp_delivery_returns_safe_actionable_failure_categories(
    monkeypatch: pytest.MonkeyPatch,
    provider_error: smtplib.SMTPException,
    safe_message: str,
) -> None:
    class FailingSmtp:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def __enter__(self) -> FailingSmtp:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def starttls(self, *, context: object) -> None:
            assert context is not None

        def login(self, username: str, password: str) -> None:
            del username, password
            if isinstance(provider_error, smtplib.SMTPAuthenticationError):
                raise provider_error

        def send_message(self, message: object) -> None:
            del message
            raise provider_error

    monkeypatch.setattr("app.services.auth.smtplib.SMTP", FailingSmtp)
    with pytest.raises(AuthenticationError, match=safe_message) as captured:
        await SmtpOtpDelivery(
            host="smtp.example.test",
            port=587,
            from_email="vouch@example.test",
            username="mailer",
            password="synthetic-secret",
        ).send(email="owner@example.test", token="123456", lifetime_minutes=10)

    assert "provider detail" not in str(captured.value)


@pytest.mark.asyncio
async def test_resend_data_error_explains_test_sender_recipient_restriction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RejectingResendSmtp:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def __enter__(self) -> RejectingResendSmtp:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def starttls(self, *, context: object) -> None:
            assert context is not None

        def login(self, username: str, password: str) -> None:
            del username, password

        def send_message(self, message: object) -> None:
            del message
            raise smtplib.SMTPDataError(403, b"private provider detail")

    monkeypatch.setattr("app.services.auth.smtplib.SMTP", RejectingResendSmtp)
    with pytest.raises(AuthenticationError, match="address used to create") as captured:
        await SmtpOtpDelivery(
            host="smtp.resend.com",
            port=587,
            from_email="onboarding@resend.dev",
            username="resend",
            password="synthetic-secret",
        ).send(email="other@example.test", token="123456", lifetime_minutes=10)

    assert "private provider detail" not in str(captured.value)


def test_web_otp_is_delivered_by_email_and_never_returned_in_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    delivered: list[tuple[str, str]] = []

    async def fake_send(
        self: SmtpOtpDelivery, *, email: str, token: str, lifetime_minutes: int
    ) -> None:
        del self
        assert lifetime_minutes == 10
        delivered.append((email, token))

    monkeypatch.setattr(SmtpOtpDelivery, "send", fake_send)
    settings = Settings(
        _env_file=None,
        app_env="test",
        mock_mode=True,
        local_otp_delivery="smtp",
        smtp_host="smtp.example.test",
        smtp_from_email="vouch@example.test",
        database_url=f"sqlite:///{(tmp_path / 'auth.db').as_posix()}",
        config_dir=tmp_path / "config",
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        logs_dir=tmp_path / "logs",
    )
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    app = FastAPI()
    app.include_router(create_auth_router(settings, engine))

    response = TestClient(app).post("/api/auth/otp", json={"email": "mail-owner@example.test"})

    assert response.status_code == 202
    assert response.json() == {"accepted": True, "delivery": "email"}
    assert len(delivered) == 1 and delivered[0][0] == "mail-owner@example.test"
    assert delivered[0][1] not in response.text
    engine.dispose()


def test_web_otp_does_not_expose_a_local_preview_code(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        mock_mode=True,
        local_otp_delivery="preview",
        database_url=f"sqlite:///{(tmp_path / 'preview.db').as_posix()}",
        config_dir=tmp_path / "config",
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        logs_dir=tmp_path / "logs",
    )
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    app = FastAPI()
    app.include_router(create_auth_router(settings, engine))

    response = TestClient(app).post("/api/auth/otp", json={"email": "preview-owner@example.test"})

    assert response.status_code == 503
    assert "preview_code" not in response.text
    engine.dispose()


class FakeProfileReader:
    def __init__(self) -> None:
        self.max_results: int | None = None
        self.exclude_replies: bool | None = None

    async def get_user_by_username(self, username: str) -> dict[str, str]:
        assert username == "synthetic_author"
        return {"id": "x-user-1"}

    async def get_user_posts(
        self,
        user_id: str,
        *,
        max_results: int,
        start_time: datetime,
        exclude_replies: bool = True,
        exclude_retweets: bool = True,
    ) -> tuple[dict[str, Any], ...]:
        del start_time, exclude_retweets
        assert user_id == "x-user-1"
        self.max_results = max_results
        self.exclude_replies = exclude_replies
        return tuple({"text": f"synthetic sample {index}"} for index in range(75))


@pytest.mark.asyncio
async def test_x_voice_analysis_is_capped_at_50_and_stores_no_raw_samples() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    reader = FakeProfileReader()
    with Session(engine) as session:
        user = UserAccount(
            auth_provider="local",
            auth_subject="operator",
            email="operator@example.test",
        )
        session.add(user)
        session.flush()
        service = VoiceProfileService(session)
        service.save_preferences(
            user=user,
            account_type="personal",
            language="en",
            preferences=["direct"],
            x_username="synthetic_author",
        )

        profile, evidence = await service.analyze_x_profile(
            user=user,
            reader=reader,
            analyzer=DeterministicVoiceAnalyzer(),
        )

        assert reader.max_results == 50
        assert reader.exclude_replies is False
        assert profile.sample_count == evidence.sample_count == 50
        assert profile.analysis_metadata == {
            "official_x_api": True,
            "max_posts": 50,
            "raw_stored": False,
        }
        assert "synthetic sample 49" not in profile.guidance
    engine.dispose()


def test_local_account_voice_profile_is_passed_to_draft_generation(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    shutil.copytree(Path("config"), config_dir)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        user = UserAccount(
            auth_provider="local",
            auth_subject="local-voice-owner",
            email="voice-owner@example.test",
        )
        session.add(user)
        session.flush()
        profile = VoiceProfileService(session).save_preferences(
            user=user,
            account_type="personal",
            language="en",
            preferences=["direct", "dry_humor"],
            x_username="synthetic_author",
        )
        profile.tone = ["concise", "dry"]
        profile.guidance = "Prefer one grounded conversational move."
        services = build_services(
            session,
            Settings(
                _env_file=None,
                app_env="test",
                mock_mode=True,
                config_dir=config_dir,
                data_dir=tmp_path / "data",
                drafts_dir=tmp_path / "drafts",
                logs_dir=tmp_path / "logs",
                database_url="sqlite:///:memory:",
            ),
        )

        assert services.drafts.voice_profile.response_preferences == ("direct", "dry_humor")
        assert services.drafts.voice_profile.tone == ("concise", "dry")
        assert services.drafts.voice_profile.guidance.startswith("Prefer one grounded")
    engine.dispose()


def test_default_runtime_does_not_use_paid_provider_for_semantic_rechecks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A normal draft uses the provider for prose, not connection probes or duplicate judges."""

    class RemoteCapableProvider:
        name = "openai"
        model = "synthetic"

        async def generate_structured(self, request: object) -> object:
            raise AssertionError(f"generation was not requested: {request!r}")

        async def inspect_source(self, *_args: object) -> object:
            raise AssertionError("remote source inspection must be opt-in")

        async def inspect_contributions(self, *_args: object) -> object:
            raise AssertionError("remote contribution inspection must be opt-in")

        async def inspect_draft(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("remote draft inspection must be opt-in")

        async def extract_claims(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("remote claim extraction must be opt-in")

    config_dir = tmp_path / "config"
    shutil.copytree(Path("config"), config_dir)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    provider = RemoteCapableProvider()
    monkeypatch.setattr("app.container._build_provider", lambda *_args, **_kwargs: provider)

    with Session(engine) as session:
        services = build_services(
            session,
            Settings(
                _env_file=None,
                app_env="test",
                mock_mode=False,
                openai_api_key="synthetic-openai-key",
                config_dir=config_dir,
                data_dir=tmp_path / "data",
                drafts_dir=tmp_path / "drafts",
                logs_dir=tmp_path / "logs",
                database_url="sqlite:///:memory:",
            ),
        )

        assert services.drafts.provider is provider
        assert (
            services.drafts.semantic_inspector is services.drafts.deterministic_semantic_inspector
        )
        assert services.drafts.claim_extractor is services.drafts.deterministic_claim_extractor
        assert services.drafts.final_text_inspector.live_inspection_configured is False
        assert services.drafts.editorial_quality_retry_count == 1
    engine.dispose()
