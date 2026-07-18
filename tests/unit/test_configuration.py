"""Configuration files and X capability checks fail closed."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.config import ConfigurationError, Settings, XCapability, require_loopback_host
from app.models.enums import ContentSourceKind, ContentType
from app.services.configuration import ConfigurationService

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_safe_defaults_select_only_the_available_mock_providers() -> None:
    settings = Settings(_env_file=None)

    assert settings.mock_mode is True
    assert settings.llm_mode == "mock"
    assert settings.draft_provider == "mock"
    assert settings.critic_provider == "mock"
    assert settings.final_provider == "mock"


def test_default_runtime_directories_follow_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for filename in (
        "content_profile.yml",
        "content_sources.yml",
        "cost_estimates.yml",
        "runtime.yml",
    ):
        (config_dir / filename).write_text(
            (PROJECT_ROOT / "config" / filename).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    monkeypatch.chdir(tmp_path)

    settings = Settings(_env_file=None)

    assert settings.data_dir == (tmp_path / "data").resolve()
    assert settings.drafts_dir == (tmp_path / "drafts").resolve()
    assert settings.config_dir == config_dir.resolve()


def test_checked_in_content_configuration_is_strictly_validated() -> None:
    configuration = ConfigurationService(PROJECT_ROOT / "config").load()

    assert configuration.sources.source_for_kind(
        ContentSourceKind.HOME_TIMELINE
    ).weight == pytest.approx(0.5)
    assert configuration.sources.source_for_kind(
        ContentSourceKind.IMPORT_FILE
    ).weight == pytest.approx(0.1)
    assert ContentType.THREAD in configuration.profile.generation.formats
    assert configuration.profile.generation.variants_per_idea == 3
    assert str(configuration.costs.x.write_usd) == "0.015"
    assert str(configuration.costs.x.read_usd) == "0.005"
    assert str(configuration.costs.x.trends_request_usd) == "0.01"
    assert configuration.sources.sources.home_timeline.max_posts == 5
    assert configuration.sources.sources.recent_search.max_posts_per_query == 10
    assert configuration.sources.automatic_discovery.max_total_posts == 15
    assert configuration.runtime.providers.openai.enabled is True
    assert configuration.runtime.providers.xai.enabled is True
    assert configuration.runtime.providers.heygen.enabled is False
    assert configuration.runtime.generation.provider == "openai"
    assert configuration.runtime.generation.post_length_mode.value == "short"
    assert configuration.runtime.generation.humanizer_enabled is True
    assert configuration.runtime.generation.quality.remote_semantic_validation_enabled is False
    assert configuration.runtime.generation.humanizer_skill_path is None
    assert configuration.runtime.generation.editorial_quality_retry_count == 1
    assert configuration.runtime.generation.quality.max_humanizer_revisions == 1
    assert configuration.runtime.request_pacing.llm_structured_output_max_retries == 0
    assert configuration.runtime.telegram.operator_user_id is None
    assert configuration.runtime.telegram.effective_allowed_user_ids == ()
    assert configuration.runtime.telegram.effective_review_chat_id is None
    assert configuration.runtime.publication.manual_x_publish_enabled is False


def test_unknown_content_source_setting_is_rejected_without_echoing_value(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(
        (PROJECT_ROOT / "config" / "content_sources.yml").read_text(encoding="utf-8")
    )
    raw["sources"]["home_timeline"]["unexpected"] = "do-not-echo-this"
    (tmp_path / "content_sources.yml").write_text(
        yaml.safe_dump(raw),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError) as raised:
        ConfigurationService(tmp_path).load_sources()

    assert "home_timeline.unexpected" in str(raised.value)
    assert "do-not-echo-this" not in str(raised.value)


def test_bearer_token_does_not_authorize_home_timeline() -> None:
    settings = Settings(
        _env_file=None,
        mock_mode=False,
        x_auth_mode="oauth1",
        x_bearer_token="synthetic-bearer",
    )

    assert settings.missing_x_credentials(XCapability.APP_READ) == []
    assert settings.missing_x_home_timeline_credentials()
    assert settings.x_capability_report()["recent_search"] is True
    assert settings.x_capability_report()["home_timeline"] is False


def test_oauth2_user_token_with_required_scopes_authorizes_home_timeline() -> None:
    settings = Settings(
        _env_file=None,
        mock_mode=False,
        x_auth_mode="oauth2",
        x_access_token="synthetic-user-token",
        x_oauth2_scopes="users.read,tweet.read",
    )

    assert settings.missing_x_home_timeline_credentials() == []
    assert settings.missing_x_write_credentials() == ["X_OAUTH2_SCOPES (tweet.write)"]


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.test"])
def test_non_loopback_bind_is_rejected_even_when_admin_password_exists(host: str) -> None:
    with pytest.raises(ValueError, match="Non-loopback"):
        Settings(_env_file=None, app_host=host, admin_password="synthetic")
    with pytest.raises(ConfigurationError, match="Non-loopback"):
        require_loopback_host(host)


def test_public_release_profile_does_not_assume_premium_access() -> None:
    configuration = ConfigurationService(PROJECT_ROOT / "config").load()

    assert configuration.profile.account.x_account_tier == "standard"
    assert configuration.profile.account.premium_long_posts_enabled is False
    assert configuration.profile.account.premium_long_post_max_chars == 25_000
    assert configuration.profile.account.default_post_max_chars == 280


def test_checked_in_request_pacing_is_conservative() -> None:
    runtime = ConfigurationService(PROJECT_ROOT / "config").load_runtime()

    assert runtime.request_pacing.x_request_delay_seconds == 2.0
    assert runtime.request_pacing.x_temporary_error_max_retries == 2
    assert runtime.request_pacing.x_temporary_error_initial_backoff_seconds == 2.0
    assert runtime.request_pacing.x_temporary_error_max_backoff_seconds == 10.0
    assert runtime.request_pacing.llm_pre_request_delay_seconds == 3.0
    assert runtime.request_pacing.llm_minimum_interval_seconds == 30.0
    assert runtime.request_pacing.llm_rate_limit_max_retries == 2
    assert runtime.request_pacing.insufficient_quota_cooldown_minutes == 60
    assert runtime.request_pacing.llm_structured_output_max_retries == 0
    assert runtime.request_pacing.llm_structured_output_retry_delay_seconds == 2.0


def test_public_release_has_no_default_telegram_operator() -> None:
    runtime = ConfigurationService(PROJECT_ROOT / "config").load_runtime().telegram

    assert runtime.operator_user_id is None
    assert runtime.allowed_user_ids == ()
    assert runtime.review_chat_id is None
    assert runtime.effective_allowed_user_ids == ()
    assert runtime.effective_review_chat_id is None


def test_x_temporary_retry_backoff_window_is_validated(tmp_path: Path) -> None:
    raw = yaml.safe_load((PROJECT_ROOT / "config" / "runtime.yml").read_text(encoding="utf-8"))
    raw["request_pacing"]["x_temporary_error_initial_backoff_seconds"] = 12.0
    raw["request_pacing"]["x_temporary_error_max_backoff_seconds"] = 4.0
    (tmp_path / "runtime.yml").write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="request_pacing"):
        ConfigurationService(tmp_path).load_runtime()


def test_automatic_discovery_interval_presets_and_custom_minutes() -> None:
    from app.schemas.configuration import AutomaticDiscoveryConfiguration

    assert AutomaticDiscoveryConfiguration(interval_preset="1h").effective_interval_minutes == 60
    assert AutomaticDiscoveryConfiguration(interval_preset="3h").effective_interval_minutes == 180
    assert AutomaticDiscoveryConfiguration(interval_preset="6h").effective_interval_minutes == 360
    assert AutomaticDiscoveryConfiguration(interval_preset="12h").effective_interval_minutes == 720
    assert (
        AutomaticDiscoveryConfiguration(
            interval_preset="custom", custom_interval_minutes=10
        ).effective_interval_minutes
        == 10
    )


def test_automatic_discovery_legacy_interval_hours_is_migrated() -> None:
    from app.schemas.configuration import AutomaticDiscoveryConfiguration

    configuration = AutomaticDiscoveryConfiguration.model_validate(
        {"interval_hours": 3, "max_runs_per_utc_day": 12}
    )

    assert configuration.interval_preset == "3h"
    assert configuration.effective_interval_minutes == 180
    assert configuration.max_runs_per_utc_day == 12


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("generation", "signal_selection", "enabled"), "mandatory safety stage"),
        (("generation", "angle_selection_enabled"), "mandatory safety stage"),
    ],
)
def test_deprecated_generation_safety_switches_fail_visibly(
    tmp_path: Path, path: tuple[str, ...], message: str
) -> None:
    raw = yaml.safe_load((PROJECT_ROOT / "config" / "runtime.yml").read_text(encoding="utf-8"))
    cursor = raw
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = False
    (tmp_path / "runtime.yml").write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message):
        ConfigurationService(tmp_path).load_runtime()


def test_legacy_retry_setting_must_match_canonical_humanizer_revision_limit(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load((PROJECT_ROOT / "config" / "runtime.yml").read_text(encoding="utf-8"))
    raw["generation"]["editorial_quality_retry_count"] = 0
    raw["generation"]["quality"]["max_humanizer_revisions"] = 1
    (tmp_path / "runtime.yml").write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="deprecated"):
        ConfigurationService(tmp_path).load_runtime()
