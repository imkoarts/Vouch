"""Smoke tests for the two runnable entry points required by the mock MVP."""

import asyncio
from pathlib import Path

import httpx
import pytest
from alembic.config import Config
from typer.testing import CliRunner

from alembic import command

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_cli_help_starts_without_import_error() -> None:
    from app.cli import app as cli_app

    result = CliRunner().invoke(cli_app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "ideas" in result.output
    assert "drafts" in result.output
    assert "doctor" in result.output


def test_activity_doctor_is_safe_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.cli import app as cli_app

    monkeypatch.chdir(PROJECT_ROOT)
    result = CliRunner().invoke(cli_app, ["activity", "doctor"])

    assert result.exit_code == 0, result.output
    assert '"status": "disabled"' in result.output
    assert '"credentials_ready": null' in result.output


def test_cli_serve_rejects_non_loopback_override() -> None:
    from app.cli import app as cli_app

    result = CliRunner().invoke(cli_app, ["serve", "--host", "0.0.0.0"])
    assert result.exit_code == 1
    assert "Non-loopback" in result.output


def test_tenant_discovery_starts_only_with_live_account_x_credentials(tmp_path: Path) -> None:
    from app.config import Settings
    from app.main import _tenant_discovery_ready

    base = {
        "_env_file": None,
        "app_env": "test",
        "mock_mode": False,
        "config_dir": PROJECT_ROOT / "config",
        "data_dir": tmp_path / "data",
        "drafts_dir": tmp_path / "drafts",
        "logs_dir": tmp_path / "logs",
        "database_url": f"sqlite:///{(tmp_path / 'tenant-ready.db').as_posix()}",
    }

    assert _tenant_discovery_ready(Settings(**base)) is False
    assert (
        _tenant_discovery_ready(Settings(**base, x_bearer_token="synthetic-tenant-bearer")) is False
    )
    assert (
        _tenant_discovery_ready(
            Settings(
                **base,
                x_bearer_token="synthetic-tenant-bearer",
                openai_api_key="synthetic-tenant-openai",
            )
        )
        is True
    )
    assert (
        _tenant_discovery_ready(
            Settings(
                **{**base, "mock_mode": True},
                x_bearer_token="synthetic-tenant-bearer",
                openai_api_key="synthetic-tenant-openai",
            )
        )
        is False
    )


def test_cli_account_login_delivers_email_and_remembers_the_selected_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from app import cli
    from app.config import Settings
    from app.database import build_engine
    from app.models import Base, UserAccount
    from app.services.auth import SmtpOtpDelivery

    settings = Settings(
        _env_file=None,
        app_env="test",
        auth_mode="local",
        local_otp_delivery="smtp",
        smtp_host="smtp.example.test",
        smtp_from_email="vouch@example.test",
        database_url=f"sqlite:///{(tmp_path / 'cli-account.db').as_posix()}",
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        logs_dir=tmp_path / "logs",
        config_dir=PROJECT_ROOT / "config",
    )
    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)
    delivered: list[tuple[str, str]] = []

    async def fake_send(
        self: SmtpOtpDelivery, *, email: str, token: str, lifetime_minutes: int
    ) -> None:
        del self, lifetime_minutes
        delivered.append((email, token))

    @contextmanager
    def fake_session_scope():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(cli, "_settings", lambda: settings)
    monkeypatch.setattr(cli, "session_scope", fake_session_scope)
    monkeypatch.setattr(SmtpOtpDelivery, "send", fake_send)
    monkeypatch.setattr(cli.typer, "prompt", lambda *_args, **_kwargs: delivered[-1][1])

    result = CliRunner().invoke(cli.app, ["account", "login", "--email", "cli@example.test"])

    assert result.exit_code == 0, result.output
    assert "One-time code sent to your email" in result.output
    assert delivered and delivered[0][0] == "cli@example.test"
    assert delivered[0][1] not in result.output
    with Session(engine) as session:
        user = session.scalars(select(UserAccount)).one()
        assert user.email == "cli@example.test"
        assert (settings.data_dir / ".active_account").read_text(encoding="utf-8").strip() == (
            user.storage_key
        )
    engine.dispose()


def test_cli_content_commands_bind_the_selected_accounts_isolated_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from sqlalchemy.orm import Session

    from app import cli
    from app.config import Settings
    from app.database import build_engine
    from app.models import Base, UserAccount

    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=f"sqlite:///{(tmp_path / 'master.db').as_posix()}",
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        logs_dir=tmp_path / "logs",
        config_dir=PROJECT_ROOT / "config",
        openai_api_key="legacy-only-key",
    )
    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            (
                UserAccount(
                    id="00000000-0000-0000-0000-000000000001",
                    auth_provider="local",
                    auth_subject="local:first",
                    email="first@example.test",
                    storage_key="11111111-1111-1111-1111-111111111111",
                ),
                UserAccount(
                    id="ffffffff-ffff-ffff-ffff-ffffffffffff",
                    auth_provider="local",
                    auth_subject="local:second",
                    email="second@example.test",
                    storage_key="22222222-2222-2222-2222-222222222222",
                ),
            )
        )
        session.commit()
    settings.data_dir.mkdir(parents=True)
    (settings.data_dir / ".active_account").write_text(
        "22222222-2222-2222-2222-222222222222\n", encoding="utf-8"
    )
    monkeypatch.setattr(cli, "get_engine", lambda: engine)
    monkeypatch.setattr(
        cli,
        "get_current_context",
        lambda **_kwargs: SimpleNamespace(command_path="vouch drafts list"),
    )

    tenant_settings = cli._bind_cli_workspace(settings)
    try:
        assert tenant_settings.data_dir != settings.data_dir
        assert "22222222-2222-2222-2222-222222222222" in str(tenant_settings.data_dir)
        assert tenant_settings.openai_api_key is None
    finally:
        cli._reset_cli_workspace()
        engine.dispose()


@pytest.mark.asyncio
async def test_fastapi_liveness_is_available_and_readiness_fails_without_migrations(
    tmp_path: Path,
) -> None:
    from app.config import Settings
    from app.health import LIVE_PATH, READY_PATH, validate_health_response
    from app.main import create_app

    settings = Settings(
        app_env="test",
        mock_mode=True,
        database_url=f"sqlite:///{(tmp_path / 'health.db').as_posix()}",
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        config_dir=PROJECT_ROOT / "config",
    )
    application = create_app(settings)
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(LIVE_PATH)
        validate_health_response(LIVE_PATH, response.status_code, response.json())
        readiness = await client.get(READY_PATH)
        assert readiness.status_code == 503
        assert readiness.json() == {
            "status": "not_ready",
            "reason": "schema_not_at_head",
        }


@pytest.mark.asyncio
async def test_fastapi_readiness_requires_and_accepts_exact_alembic_head(
    tmp_path: Path,
) -> None:
    from app.config import Settings
    from app.health import READY_PATH, validate_health_response
    from app.main import create_app

    root = PROJECT_ROOT
    database_path = tmp_path / "ready.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    configuration = Config(str(root / "alembic.ini"))
    configuration.set_main_option("script_location", str(root / "alembic"))
    configuration.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(configuration, "head")

    settings = Settings(
        app_env="test",
        mock_mode=True,
        database_url=database_url,
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        config_dir=root / "config",
    )
    application = create_app(settings)
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(READY_PATH)
        validate_health_response(READY_PATH, response.status_code, response.json())


@pytest.mark.asyncio
async def test_local_web_login_onboarding_and_dashboard_are_reachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import Settings
    from app.main import create_app
    from app.services.auth import SmtpOtpDelivery

    delivered: list[tuple[str, str]] = []

    async def fake_send(
        self: SmtpOtpDelivery, *, email: str, token: str, lifetime_minutes: int
    ) -> None:
        del self, lifetime_minutes
        delivered.append((email, token))

    monkeypatch.setattr(SmtpOtpDelivery, "send", fake_send)

    database_path = tmp_path / "local-auth.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    configuration = Config(str(PROJECT_ROOT / "alembic.ini"))
    configuration.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    configuration.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(configuration, "head")
    settings = Settings(
        _env_file=None,
        app_env="test",
        auth_mode="local",
        mock_mode=True,
        local_otp_delivery="smtp",
        smtp_host="smtp.example.test",
        smtp_from_email="vouch@example.test",
        database_url=database_url,
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        logs_dir=tmp_path / "logs",
        config_dir=PROJECT_ROOT / "config",
        openai_api_key="legacy-owner-only-key",
    )
    transport = httpx.ASGITransport(app=create_app(settings))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        auth_config = (await client.get("/api/auth/config")).json()
        assert auth_config["passwordless_email"] is True
        assert auth_config["delivery"] == "email"
        assert (await client.get("/api/auth/session")).status_code == 401

        otp_response = await client.post("/api/auth/otp", json={"email": "owner@example.test"})
        assert otp_response.status_code == 202
        assert otp_response.json() == {"accepted": True, "delivery": "email"}
        assert len(delivered) == 1
        assert delivered[0][0] == "owner@example.test"
        assert delivered[0][1] not in otp_response.text
        verified = await client.post(
            "/api/auth/verify",
            json={"email": "owner@example.test", "token": delivered[0][1]},
        )
        assert verified.status_code == 200
        csrf = client.cookies.get("vouch_csrf")
        assert csrf
        session = (await client.get("/api/auth/session")).json()
        assert session["email"] == "owner@example.test"
        owner_credentials = (await client.get("/api/dashboard/credentials")).json()
        assert owner_credentials["configured"]["openai_api_key"] is True
        assert "legacy-owner-only-key" not in str(owner_credentials)

        profile = await client.put(
            "/api/voice-profile",
            headers={"X-CSRF-Token": csrf},
            json={
                "account_type": "personal",
                "language": "en",
                "response_preferences": ["direct", "dry_humor"],
                "x_username": "synthetic_author",
            },
        )
        assert profile.status_code == 200
        assert profile.json()["profile"]["response_preferences"] == ["direct", "dry_humor"]
        analyzed = await client.post(
            "/api/voice-profile/analyze",
            headers={"X-CSRF-Token": csrf},
        )
        assert analyzed.status_code == 200
        assert analyzed.json()["profile"]["sample_count"] == 1
        assert (await client.get("/api/dashboard/status")).status_code == 200

        logout = await client.post("/api/auth/logout", headers={"X-CSRF-Token": csrf})
        assert logout.status_code == 200
        assert (await client.get("/api/dashboard/status")).status_code == 401

        second_otp = await client.post("/api/auth/otp", json={"email": "second@example.test"})
        assert second_otp.status_code == 202
        second_verified = await client.post(
            "/api/auth/verify",
            json={"email": "second@example.test", "token": delivered[-1][1]},
        )
        assert second_verified.status_code == 200
        second_csrf = client.cookies.get("vouch_csrf")
        assert second_csrf
        second_credentials = (await client.get("/api/dashboard/credentials")).json()
        assert second_credentials["configured"]["openai_api_key"] is False
        assert "legacy-owner-only-key" not in str(second_credentials)

        saved = await client.put(
            "/api/dashboard/credentials",
            headers={"X-CSRF-Token": second_csrf},
            json={
                "values": {"openai_api_key": "second-account-only-key"},
                "clear": [],
            },
        )
        assert saved.status_code == 200
        assert "second-account-only-key" not in saved.text
        account_env_files = list((tmp_path / "data" / "users").glob("*/.env"))
        assert len(account_env_files) == 1
        assert "OPENAI_API_KEY=second-account-only-key" in account_env_files[0].read_text(
            encoding="utf-8"
        )
        assert not (tmp_path / ".env").exists()


@pytest.mark.asyncio
async def test_configured_tenant_starts_exactly_one_account_discovery_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from app.config import Settings
    from app.database import build_engine
    from app.main import create_app
    from app.models import UserAccount
    from app.services.auth import SmtpOtpDelivery
    from app.services.environment_config import update_environment_file

    delivered: list[tuple[str, str]] = []

    async def fake_send(
        self: SmtpOtpDelivery, *, email: str, token: str, lifetime_minutes: int
    ) -> None:
        del self, lifetime_minutes
        delivered.append((email, token))

    monkeypatch.setattr(SmtpOtpDelivery, "send", fake_send)
    database_path = tmp_path / "tenant-runtime.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    configuration = Config(str(PROJECT_ROOT / "alembic.ini"))
    configuration.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    configuration.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(configuration, "head")
    settings = Settings(
        _env_file=None,
        app_env="test",
        auth_mode="local",
        mock_mode=False,
        local_otp_delivery="smtp",
        smtp_host="smtp.example.test",
        smtp_from_email="vouch@example.test",
        database_url=database_url,
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        logs_dir=tmp_path / "logs",
        config_dir=PROJECT_ROOT / "config",
    )
    first_app = create_app(settings)
    first_transport = httpx.ASGITransport(app=first_app)
    async with httpx.AsyncClient(transport=first_transport, base_url="http://test") as client:
        for email in ("first@example.test", "tenant@example.test"):
            requested = await client.post("/api/auth/otp", json={"email": email})
            assert requested.status_code == 202
            verified = await client.post(
                "/api/auth/verify",
                json={"email": email, "token": delivered[-1][1]},
            )
            assert verified.status_code == 200
        session_cookies = dict(client.cookies)

    master_engine = build_engine(database_url)
    with Session(master_engine) as database:
        tenant = database.scalar(
            select(UserAccount).where(UserAccount.email == "tenant@example.test")
        )
        assert tenant is not None
        tenant_root = settings.data_dir / "users" / tenant.storage_key
    update_environment_file(
        tenant_root / ".env",
        values={
            "x_bearer_token": "synthetic-tenant-bearer",
            "openai_api_key": "synthetic-tenant-openai",
        },
    )
    master_engine.dispose()

    started: list[Path] = []

    async def fake_discovery_loop(
        runtime: Settings, *, stop_event: asyncio.Event, on_outcome: object
    ) -> None:
        del on_outcome
        started.append(runtime.data_dir)
        await stop_event.wait()

    monkeypatch.setattr("app.main.automatic_discovery_loop", fake_discovery_loop)
    restarted = create_app(settings)
    async with restarted.router.lifespan_context(restarted):
        transport = httpx.ASGITransport(app=restarted)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies=session_cookies,
        ) as client:
            assert (await client.get("/api/dashboard/credentials")).status_code == 200
            assert (await client.get("/api/dashboard/credentials")).status_code == 200
            await asyncio.sleep(0)
            tenant_starts = [path for path in started if tenant.storage_key in str(path)]
            assert len(tenant_starts) == 1
