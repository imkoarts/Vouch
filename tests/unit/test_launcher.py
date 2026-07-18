from __future__ import annotations

from pathlib import Path

import pytest

import launcher


def test_launcher_files_are_windows_native_one_click_and_release_safe() -> None:
    root = Path(__file__).resolve().parents[2]
    for name in ("START_VOUCH.bat", "CONFIGURE_VOUCH.bat"):
        raw = (root / name).read_bytes()
        text = raw.decode("ascii")

        assert b"\r\n" in raw
        assert b"\n" not in raw.replace(b"\r\n", b"")
        assert "launcher.py" in text
        assert "-3.13" in text and "-3.12" in text
        assert "chcp" not in text.casefold()

    configure = (root / "CONFIGURE_VOUCH.bat").read_text(encoding="ascii")
    assert "--configure" in configure
    assert launcher.ENV_FILE.name == ".env"
    assert launcher.venv_python().name in {"python", "python.exe"}

    setup_script = (root / "scripts" / "setup_local.ps1").read_text(encoding="utf-8")
    assert "launcher.py" in setup_script
    assert '"--configure"' in setup_script
    assert "SMTP_PASSWORD=" not in setup_script


def test_dependency_fingerprint_changes_with_locked_inputs(tmp_path: Path, monkeypatch) -> None:
    requirements = tmp_path / "requirements.lock"
    pyproject = tmp_path / "pyproject.toml"
    requirements.write_text("alpha==1\n", encoding="utf-8")
    pyproject.write_text("[project]\nname='demo'\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "REQUIREMENTS", requirements)
    monkeypatch.setattr(launcher, "ROOT", tmp_path)

    first = launcher.dependency_fingerprint()
    requirements.write_text("alpha==2\n", encoding="utf-8")

    assert launcher.dependency_fingerprint() != first


def test_configuration_round_trip_and_optional_provider_preservation(tmp_path: Path) -> None:
    existing_file = tmp_path / ".env"
    existing_file.write_text(
        'OPENAI_API_KEY="old-openai"\n'
        'TELEGRAM_BOT_TOKEN="old-telegram"\n'
        'XAI_API_KEY="keep-xai"\n'
        'HEYGEN_API_KEY="keep-heygen"\n',
        encoding="utf-8",
    )
    existing = launcher._read_existing_env(existing_file)
    configuration = launcher.ConfigurationValues(
        openai_key="new-openai",
        telegram_token="new-telegram",
        smtp_host="smtp.example.test",
        smtp_port=2525,
        smtp_username="mailer",
        smtp_password="mail-secret",
        smtp_from_email="vouch@example.test",
        x_bearer="bearer",
        x_user_id="123",
        outbound_proxy="127.0.0.1:8080:user:pass",
    )

    content = launcher._build_env_content(configuration, existing)

    assert 'OPENAI_API_KEY="new-openai"' in content
    assert 'TELEGRAM_BOT_TOKEN="new-telegram"' in content
    assert 'LOCAL_OTP_DELIVERY="smtp"' in content
    assert 'SMTP_HOST="smtp.example.test"' in content
    assert 'SMTP_PORT="2525"' in content
    assert 'SMTP_FROM_EMAIL="vouch@example.test"' in content
    assert 'XAI_API_KEY="keep-xai"' in content
    assert 'HEYGEN_API_KEY="keep-heygen"' in content
    assert 'PUBLISH_ENABLED="false"' in content
    assert 'AUTO_PUBLISH="false"' in content
    assert 'OUTBOUND_PROXY_ENABLED="true"' in content
    assert 'OUTBOUND_PROXY="127.0.0.1:8080:user:pass"' in content


def test_configuration_mapping_detects_complete_home_credentials() -> None:
    values = launcher._configuration_from_mapping(
        {
            "OPENAI_API_KEY": "openai",
            "TELEGRAM_BOT_TOKEN": "telegram",
            "LOCAL_OTP_DELIVERY": "smtp",
            "SMTP_HOST": "smtp.example.test",
            "SMTP_FROM_EMAIL": "vouch@example.test",
            "X_USER_ID": "123",
            "X_CONSUMER_KEY": "consumer",
            "X_CONSUMER_SECRET": "consumer-secret",
            "X_ACCESS_TOKEN": "access",
            "X_ACCESS_TOKEN_SECRET": "access-secret",
        }
    )

    assert values.has_home_credentials is True


def test_runtime_layout_creates_missing_sqlite_parent_in_path_with_spaces(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "X bot" / "xbot"
    root.mkdir(parents=True)
    env_file = root / ".env"
    env_file.write_text(
        'DATABASE_URL="sqlite:///./nested data/app.db"\n'
        'DATA_DIR="./data"\n'
        'DRAFTS_DIR="./drafts"\n'
        'LOGS_DIR="./logs"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "ROOT", root)
    monkeypatch.setattr(launcher, "ENV_FILE", env_file)
    monkeypatch.setattr(launcher, "LAUNCHER_LOG", root / "logs" / "launcher.log")

    launcher.ensure_runtime_layout()

    assert (root / "data").is_dir()
    assert (root / "drafts").is_dir()
    assert (root / "logs").is_dir()
    assert (root / "nested data").is_dir()


def test_sqlite_database_path_ignores_memory_and_resolves_relative(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(launcher, "ROOT", tmp_path)

    assert launcher._sqlite_database_path("sqlite:///:memory:") is None
    assert launcher._sqlite_database_path("postgresql://localhost/app") is None
    assert (
        launcher._sqlite_database_path("sqlite:///./data/app.db")
        == (tmp_path / "data" / "app.db").resolve()
    )


def test_launcher_reads_and_updates_post_length_mode(tmp_path: Path, monkeypatch) -> None:
    runtime = tmp_path / "runtime.yml"
    runtime.write_text(
        "generation:\n  provider: openai\n  post_length_mode: short\ntelegram:\n  enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "RUNTIME_CONFIG", runtime)

    assert launcher._read_post_length_mode() == "short"

    launcher._write_post_length_mode("random")

    assert launcher._read_post_length_mode() == "random"
    assert "post_length_mode: random" in runtime.read_text(encoding="utf-8")


def test_launcher_reads_and_updates_request_pacing(tmp_path: Path, monkeypatch) -> None:
    runtime = tmp_path / "runtime.yml"
    runtime.write_text(
        "generation:\n"
        "  post_length_mode: short\n"
        "request_pacing:\n"
        "  x_request_delay_seconds: 2.0\n"
        "  llm_minimum_interval_seconds: 30.0\n"
        "telegram:\n"
        "  enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "RUNTIME_CONFIG", runtime)

    assert launcher._read_runtime_float("request_pacing", "x_request_delay_seconds", 0) == 2
    assert launcher._read_runtime_float("request_pacing", "llm_minimum_interval_seconds", 0) == 30

    launcher._write_request_pacing(4.5, 90)

    rendered = runtime.read_text(encoding="utf-8")
    assert "x_request_delay_seconds: 4.5" in rendered
    assert "llm_minimum_interval_seconds: 90" in rendered


def test_launcher_updates_generation_provider_and_single_telegram_operator_id(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = tmp_path / "runtime.yml"
    runtime.write_text(
        "providers:\n"
        "  openai:\n"
        "    enabled: true\n"
        "generation:\n"
        "  provider: openai\n"
        "telegram:\n"
        "  enabled: true\n"
        "  allowed_user_ids:\n"
        "    - 111\n"
        "  review_chat_id: 111\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "RUNTIME_CONFIG", runtime)

    launcher._write_generation_provider("xai")
    launcher._write_telegram_operator_id("222")

    rendered = runtime.read_text(encoding="utf-8")
    assert "provider: xai" in rendered
    assert "operator_user_id: 222" in rendered
    assert "allowed_user_ids: []" in rendered
    assert "review_chat_id: null" in rendered
    assert "    - 111" not in rendered


def test_configuration_mapping_reads_xai_provider_and_telegram_id(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = tmp_path / "runtime.yml"
    runtime.write_text(
        "generation:\n"
        "  provider: xai\n"
        "  post_length_mode: short\n"
        "telegram:\n"
        "  operator_user_id: 987654321\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "RUNTIME_CONFIG", runtime)

    values = launcher._configuration_from_mapping(
        {
            "OPENAI_API_KEY": "openai",
            "XAI_API_KEY": "xai",
            "TELEGRAM_BOT_TOKEN": "telegram",
        }
    )

    assert values.generation_provider == "xai"
    assert values.xai_key == "xai"
    assert values.telegram_user_id == "987654321"


def test_launcher_reads_and_updates_premium_long_post_profile(tmp_path: Path, monkeypatch) -> None:
    profile = tmp_path / "content_profile.yml"
    profile.write_text(
        "account:\n"
        "  language: en\n"
        "  x_account_tier: standard\n"
        "  premium_long_posts_enabled: false\n"
        "  premium_long_post_max_chars: 25000\n"
        "brand:\n"
        "  name: ''\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "CONTENT_PROFILE_CONFIG", profile)

    assert launcher._read_premium_long_posts_enabled() is False

    launcher._write_premium_long_posts_enabled(True)

    rendered = profile.read_text(encoding="utf-8")
    assert launcher._read_premium_long_posts_enabled() is True
    assert "x_account_tier: premium" in rendered
    assert "premium_long_posts_enabled: true" in rendered
    assert "premium_long_post_max_chars: 25000" in rendered

    launcher._write_premium_long_posts_enabled(False)
    rendered = profile.read_text(encoding="utf-8")
    assert launcher._read_premium_long_posts_enabled() is False
    assert "x_account_tier: standard" in rendered
    assert "premium_long_posts_enabled: false" in rendered


def test_configuration_mapping_reads_premium_long_post_profile(tmp_path: Path, monkeypatch) -> None:
    runtime = tmp_path / "runtime.yml"
    runtime.write_text(
        "generation:\n  provider: openai\n  post_length_mode: medium\n"
        "telegram:\n  operator_user_id: 123\n",
        encoding="utf-8",
    )
    profile = tmp_path / "content_profile.yml"
    profile.write_text(
        "account:\n"
        "  x_account_tier: premium\n"
        "  premium_long_posts_enabled: true\n"
        "  premium_long_post_max_chars: 25000\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "RUNTIME_CONFIG", runtime)
    monkeypatch.setattr(launcher, "CONTENT_PROFILE_CONFIG", profile)

    values = launcher._configuration_from_mapping({})

    assert values.post_length_mode == "medium"
    assert values.premium_long_posts_enabled is True


def test_missing_configuration_fields_detects_runtime_telegram_migration(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = tmp_path / "runtime.yml"
    runtime.write_text(
        "generation:\n"
        "  provider: openai\n"
        "  post_length_mode: short\n"
        "telegram:\n"
        "  enabled: true\n"
        "  operator_user_id: null\n",
        encoding="utf-8",
    )
    profile = tmp_path / "content_profile.yml"
    profile.write_text(
        "account:\n  x_account_tier: standard\n  premium_long_posts_enabled: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "RUNTIME_CONFIG", runtime)
    monkeypatch.setattr(launcher, "CONTENT_PROFILE_CONFIG", profile)

    missing = launcher._missing_configuration_fields(
        {
            "OPENAI_API_KEY": "openai",
            "TELEGRAM_BOT_TOKEN": "telegram",
            "LOCAL_OTP_DELIVERY": "smtp",
            "SMTP_HOST": "smtp.example.test",
            "SMTP_FROM_EMAIL": "vouch@example.test",
        }
    )

    assert missing == ("telegram.operator_user_id",)


def test_missing_configuration_fields_requires_real_email_delivery(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = tmp_path / "runtime.yml"
    runtime.write_text(
        "generation:\n  provider: openai\ntelegram:\n  enabled: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "RUNTIME_CONFIG", runtime)

    assert launcher._missing_configuration_fields({"OPENAI_API_KEY": "openai"}) == (
        "email sign-in delivery",
    )
    assert launcher._missing_configuration_fields(
        {
            "OPENAI_API_KEY": "openai",
            "LOCAL_OTP_DELIVERY": "smtp",
            "SMTP_HOST": "smtp.example.test",
            "SMTP_PORT": "not-a-port",
            "SMTP_FROM_EMAIL": "vouch@example.test",
        }
    ) == ("SMTP email delivery",)


def test_missing_configuration_fields_skips_disabled_telegram(tmp_path: Path, monkeypatch) -> None:
    runtime = tmp_path / "runtime.yml"
    runtime.write_text(
        "generation:\n"
        "  provider: openai\n"
        "  post_length_mode: short\n"
        "telegram:\n"
        "  enabled: false\n"
        "  operator_user_id: null\n",
        encoding="utf-8",
    )
    profile = tmp_path / "content_profile.yml"
    profile.write_text(
        "account:\n  x_account_tier: standard\n  premium_long_posts_enabled: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "RUNTIME_CONFIG", runtime)
    monkeypatch.setattr(launcher, "CONTENT_PROFILE_CONFIG", profile)

    missing = launcher._missing_configuration_fields(
        {
            "OPENAI_API_KEY": "openai",
            "LOCAL_OTP_DELIVERY": "smtp",
            "SMTP_HOST": "smtp.example.test",
            "SMTP_FROM_EMAIL": "vouch@example.test",
        }
    )

    assert missing == ()


def test_configure_env_repairs_existing_install_missing_telegram_operator(
    tmp_path: Path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        'OPENAI_API_KEY="openai"\nTELEGRAM_BOT_TOKEN="telegram"\n',
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime.yml"
    runtime.write_text(
        "generation:\n"
        "  provider: openai\n"
        "  post_length_mode: short\n"
        "request_pacing:\n"
        "  x_request_delay_seconds: 2\n"
        "  llm_minimum_interval_seconds: 30\n"
        "telegram:\n"
        "  enabled: true\n"
        "  operator_user_id: null\n"
        "  allowed_user_ids: []\n"
        "  review_chat_id: null\n",
        encoding="utf-8",
    )
    profile = tmp_path / "content_profile.yml"
    profile.write_text(
        "account:\n"
        "  x_account_tier: standard\n"
        "  premium_long_posts_enabled: false\n"
        "  premium_long_post_max_chars: 25000\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "ENV_FILE", env_file)
    monkeypatch.setattr(launcher, "RUNTIME_CONFIG", runtime)
    monkeypatch.setattr(launcher, "CONTENT_PROFILE_CONFIG", profile)
    monkeypatch.setattr(
        launcher,
        "collect_configuration",
        lambda existing: launcher.ConfigurationValues(
            openai_key=existing.openai_key,
            telegram_token=existing.telegram_token,
            smtp_host="smtp.example.test",
            smtp_from_email="vouch@example.test",
            telegram_user_id="987654321",
            generation_provider="openai",
            post_length_mode="short",
            x_request_delay_seconds=2,
            llm_minimum_interval_seconds=30,
        ),
    )

    launcher.configure_env()

    assert "operator_user_id: 987654321" in runtime.read_text(encoding="utf-8")
    assert list(tmp_path.glob(".env.backup-*"))


def test_configure_env_keeps_complete_existing_install_without_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    original = (
        'OPENAI_API_KEY="openai"\n'
        'TELEGRAM_BOT_TOKEN="telegram"\n'
        'LOCAL_OTP_DELIVERY="smtp"\n'
        'SMTP_HOST="smtp.example.test"\n'
        'SMTP_FROM_EMAIL="vouch@example.test"\n'
    )
    env_file.write_text(original, encoding="utf-8")
    runtime = tmp_path / "runtime.yml"
    runtime.write_text(
        "generation:\n"
        "  provider: openai\n"
        "  post_length_mode: short\n"
        "telegram:\n"
        "  enabled: true\n"
        "  operator_user_id: 987654321\n",
        encoding="utf-8",
    )
    profile = tmp_path / "content_profile.yml"
    profile.write_text(
        "account:\n  x_account_tier: standard\n  premium_long_posts_enabled: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "ENV_FILE", env_file)
    monkeypatch.setattr(launcher, "RUNTIME_CONFIG", runtime)
    monkeypatch.setattr(launcher, "CONTENT_PROFILE_CONFIG", profile)

    def fail_if_prompted(existing: launcher.ConfigurationValues) -> launcher.ConfigurationValues:
        raise AssertionError(f"unexpected setup prompt for {existing}")

    monkeypatch.setattr(launcher, "collect_configuration", fail_if_prompted)

    launcher.configure_env()

    assert env_file.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".env.backup-*"))


def test_launcher_reads_and_updates_automatic_discovery_schedule(
    tmp_path: Path, monkeypatch
) -> None:
    sources = tmp_path / "content_sources.yml"
    sources.write_text(
        "automatic_discovery:\n"
        "  enabled: true\n"
        "  interval_preset: 12h\n"
        "  custom_interval_minutes: 10\n"
        "  max_runs_per_utc_day: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "CONTENT_SOURCES_CONFIG", sources)

    assert launcher._read_discovery_schedule() == ("12h", 10, 2)

    launcher._write_discovery_schedule("custom", 10, 48)

    assert launcher._read_discovery_schedule() == ("custom", 10, 48)
    rendered = sources.read_text(encoding="utf-8")
    assert "interval_preset: custom" in rendered
    assert "custom_interval_minutes: 10" in rendered
    assert "max_runs_per_utc_day: 48" in rendered


def test_launcher_migrates_legacy_discovery_interval_hours(tmp_path: Path, monkeypatch) -> None:
    sources = tmp_path / "content_sources.yml"
    sources.write_text(
        "automatic_discovery:\n  enabled: true\n  interval_hours: 6\n  max_runs_per_utc_day: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "CONTENT_SOURCES_CONFIG", sources)

    assert launcher._read_discovery_schedule() == ("6h", 10, 2)


def test_configuration_can_enable_manual_publish_with_nonzero_budget() -> None:
    configuration = launcher.ConfigurationValues(
        openai_key="openai",
        telegram_token="telegram",
        manual_x_publish_enabled=True,
        enterprise_quote_posts_enabled=True,
        daily_x_write_limit_usd=0.10,
    )

    content = launcher._build_env_content(configuration)

    assert 'PUBLISH_ENABLED="true"' in content
    assert 'AUTO_PUBLISH="false"' in content
    assert 'DAILY_X_WRITE_LIMIT_USD="0.1"' in content


def test_windows_setup_installs_explicit_clipboard_shortcuts() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")

    assert '"<Control-v>"' in source
    assert '"<Control-V>"' in source
    assert '"<Shift-Insert>"' in source
    assert "root.clipboard_get()" in source


def test_manual_publish_low_budget_can_be_accepted_and_raised_to_minimum() -> None:
    confirmations: list[float] = []

    resolved = launcher._resolve_manual_publish_budget(
        enabled=True,
        budget_usd=0.0,
        confirm_increase=lambda minimum: confirmations.append(minimum) or True,
    )

    assert resolved == launcher.MIN_MANUAL_X_WRITE_BUDGET_USD
    assert confirmations == [launcher.MIN_MANUAL_X_WRITE_BUDGET_USD]


def test_manual_publish_low_budget_can_be_declined_without_saving() -> None:
    resolved = launcher._resolve_manual_publish_budget(
        enabled=True,
        budget_usd=0.01,
        confirm_increase=lambda _minimum: False,
    )

    assert resolved is None


def test_disabled_manual_publish_preserves_zero_budget_without_prompt() -> None:
    resolved = launcher._resolve_manual_publish_budget(
        enabled=False,
        budget_usd=0.0,
        confirm_increase=lambda _minimum: (_ for _ in ()).throw(AssertionError("unexpected")),
    )

    assert resolved == 0.0


def test_negative_manual_publish_budget_is_still_invalid() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        launcher._resolve_manual_publish_budget(
            enabled=True,
            budget_usd=-0.01,
            confirm_increase=lambda _minimum: True,
        )


def test_windows_setup_uses_confirmation_instead_of_terminal_low_budget_error() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")

    assert "messagebox.askyesno(" in source
    assert "Set the budget to {minimum:.3f} USD and save these settings?" in source
    assert 'messagebox.showerror(\n                "X write budget is too low"' not in source


def test_launcher_exposes_local_x_read_budget_separately_from_x_console_cap(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = tmp_path / "runtime.yml"
    runtime.write_text(
        "generation:\n  provider: openai\n  post_length_mode: short\n"
        "telegram:\n  operator_user_id: 123\n",
        encoding="utf-8",
    )
    profile = tmp_path / "content_profile.yml"
    profile.write_text(
        "account:\n  premium_long_posts_enabled: false\n",
        encoding="utf-8",
    )
    sources = tmp_path / "content_sources.yml"
    sources.write_text(
        "automatic_discovery:\n  interval_preset: 12h\n"
        "  custom_interval_minutes: 10\n  max_runs_per_utc_day: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "RUNTIME_CONFIG", runtime)
    monkeypatch.setattr(launcher, "CONTENT_PROFILE_CONFIG", profile)
    monkeypatch.setattr(launcher, "CONTENT_SOURCES_CONFIG", sources)

    values = launcher._configuration_from_mapping(
        {
            "OPENAI_API_KEY": "openai",
            "TELEGRAM_BOT_TOKEN": "telegram",
            "DAILY_X_READ_LIMIT_USD": "2.0",
        }
    )
    rendered = launcher._build_env_content(values)

    assert values.daily_x_read_limit_usd == 2.0
    assert 'DAILY_X_READ_LIMIT_USD="2"' in rendered


def test_write_budget_can_follow_read_budget_in_env() -> None:
    configuration = launcher.ConfigurationValues(
        openai_key="openai",
        telegram_token="telegram",
        manual_x_publish_enabled=True,
        daily_x_read_limit_usd=2.0,
        daily_x_write_limit_usd=2.0,
        x_write_budget_matches_read=True,
    )

    content = launcher._build_env_content(configuration)

    assert 'DAILY_X_READ_LIMIT_USD="2"' in content
    assert 'DAILY_X_WRITE_LIMIT_USD="2"' in content
    assert 'X_WRITE_BUDGET_MATCH_READ="true"' in content


def test_existing_write_budget_match_setting_is_loaded(tmp_path: Path, monkeypatch) -> None:
    runtime = tmp_path / "runtime.yml"
    runtime.write_text(
        "generation:\n  provider: openai\n  post_length_mode: short\n"
        "telegram:\n  operator_user_id: 123\n"
        "publication:\n  manual_x_publish_enabled: true\n",
        encoding="utf-8",
    )
    sources = tmp_path / "content_sources.yml"
    sources.write_text(
        "automatic_discovery:\n  interval_preset: 12h\n  custom_interval_minutes: 10\n"
        "  max_runs_per_utc_day: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "RUNTIME_CONFIG", runtime)
    monkeypatch.setattr(launcher, "CONTENT_SOURCES_CONFIG", sources)

    configuration = launcher._configuration_from_mapping(
        {
            "PUBLISH_ENABLED": "true",
            "DAILY_X_READ_LIMIT_USD": "2",
            "DAILY_X_WRITE_LIMIT_USD": "2",
            "X_WRITE_BUDGET_MATCH_READ": "true",
        }
    )

    assert configuration.x_write_budget_matches_read is True
