from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine

from app.config import Settings
from app.services.doctor import DoctorService

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _statuses(settings: Settings) -> dict[str, tuple[str, str]]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    return {
        check.name: (check.status, check.message) for check in DoctorService(settings, engine).run()
    }


def _configured_runtime_dir(tmp_path: Path, *, provider: str = "openai") -> Path:
    import yaml

    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    for name in ("content_profile.yml", "content_sources.yml", "cost_estimates.yml"):
        (config_dir / name).write_text(
            (PROJECT_ROOT / "config" / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    runtime = yaml.safe_load((PROJECT_ROOT / "config" / "runtime.yml").read_text(encoding="utf-8"))
    runtime["generation"]["provider"] = provider
    runtime["telegram"]["enabled"] = True
    runtime["telegram"]["operator_user_id"] = 123456789
    (config_dir / "runtime.yml").write_text(yaml.safe_dump(runtime), encoding="utf-8")
    return config_dir


def test_doctor_reports_missing_telegram_secret_without_network(tmp_path: Path) -> None:
    import yaml

    config_dir = _configured_runtime_dir(tmp_path)
    runtime_path = config_dir / "runtime.yml"
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    runtime["telegram"]["operator_user_id"] = None
    runtime_path.write_text(yaml.safe_dump(runtime), encoding="utf-8")
    settings = Settings(
        _env_file=None,
        config_dir=config_dir,
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        mock_mode=True,
        publish_enabled=False,
    )

    checks = _statuses(settings)

    assert checks["telegram"][0] == "FAIL"
    assert "TELEGRAM_BOT_TOKEN" in checks["telegram"][1]
    assert checks["telegram"][1].count("telegram.operator_user_id") == 1
    assert checks["heygen"] == ("SKIP", "disabled in config/runtime.yml")
    assert checks["publishing"][0] == "PASS"


def test_doctor_accepts_configured_telegram_and_openai(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        config_dir=_configured_runtime_dir(tmp_path),
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        mock_mode=False,
        openai_api_key="synthetic-openai-key",
        telegram_bot_token="synthetic-telegram-token",
        publish_enabled=False,
    )

    checks = _statuses(settings)

    assert checks["generation_provider"][0] == "PASS"
    assert checks["llm_request_policy"] == (
        "PASS",
        "local deterministic validation; doctor/startup send no provider probe",
    )
    assert checks["telegram"][0] == "PASS"
    assert checks["publishing"][0] == "PASS"


def test_doctor_accepts_xai_as_selected_alternative(tmp_path: Path) -> None:
    config_dir = _configured_runtime_dir(tmp_path, provider="xai")
    settings = Settings(
        _env_file=None,
        config_dir=config_dir,
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        mock_mode=False,
        xai_api_key="synthetic-xai-key",
        telegram_bot_token="synthetic-telegram-token",
        publish_enabled=False,
    )

    checks = _statuses(settings)

    assert checks["generation_provider"][0] == "PASS"
    assert "grok-4.5" in checks["generation_provider"][1]


def test_doctor_accepts_medium_mode_when_premium_profile_is_enabled(tmp_path: Path) -> None:
    import yaml

    config_dir = _configured_runtime_dir(tmp_path)
    runtime_path = config_dir / "runtime.yml"
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    runtime["generation"]["post_length_mode"] = "medium"
    runtime_path.write_text(yaml.safe_dump(runtime), encoding="utf-8")

    profile_path = config_dir / "content_profile.yml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["account"]["x_account_tier"] = "premium"
    profile["account"]["premium_long_posts_enabled"] = True
    profile_path.write_text(yaml.safe_dump(profile), encoding="utf-8")

    settings = Settings(
        _env_file=None,
        config_dir=config_dir,
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        mock_mode=False,
        openai_api_key="synthetic-openai-key",
        telegram_bot_token="synthetic-telegram-token",
        publish_enabled=False,
    )

    checks = _statuses(settings)

    assert checks["post_length_mode"][0] == "PASS"
    assert "medium" in checks["post_length_mode"][1]
