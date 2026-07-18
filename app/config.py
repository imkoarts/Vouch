"""Environment and YAML configuration with fail-closed safety defaults."""

from __future__ import annotations

import ipaddress
import re
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.resources import resolve_default_config_dir
from app.utils.proxy import ProxyConfigurationError, normalize_proxy_url


class ConfigurationError(ValueError):
    """Raised when a setting would weaken an architectural safety invariant."""


def is_loopback_host(host: str) -> bool:
    """Return whether a bind target is unambiguously local-only."""

    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.casefold() == "localhost"


def require_loopback_host(host: str) -> str:
    if not is_loopback_host(host):
        raise ConfigurationError(
            "Non-loopback bind is disabled until real web authentication is implemented"
        )
    return host


class XCapability(StrEnum):
    """Official X API capabilities with distinct authentication requirements."""

    APP_READ = "app_read"
    HOME_TIMELINE = "home_timeline"
    POST_WRITE = "post_write"
    MEDIA_WRITE = "media_write"
    ACTIVITY_PUBLIC = "activity_public"
    ACTIVITY_PRIVATE = "activity_private"


class Settings(BaseSettings):
    """Runtime settings.

    Secrets use ``SecretStr`` so accidental repr/logging does not expose their values.
    Empty secret environment variables are normalized to ``None``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["development", "test", "production"] = "development"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    database_url: str = "sqlite:///./data/app.db"
    auth_mode: Literal["local", "supabase"] = "local"
    local_otp_delivery: Literal["preview", "smtp"] = "preview"
    smtp_host: str | None = None
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: str | None = None
    smtp_password: SecretStr | None = None
    smtp_from_email: str | None = None
    smtp_starttls: bool = True
    otp_min_interval_seconds: int = Field(default=60, ge=10, le=3600)
    otp_hourly_limit: int = Field(default=5, ge=1, le=30)
    supabase_url: str | None = None
    supabase_publishable_key: SecretStr | None = None
    auth_session_hours: int = Field(default=168, ge=1, le=720)
    admin_password: SecretStr | None = None
    secret_key: SecretStr | None = None
    mock_mode: bool = True

    data_dir: Path = Path("data")
    drafts_dir: Path = Path("drafts")
    logs_dir: Path = Path("logs")
    config_dir: Path = Field(default_factory=resolve_default_config_dir)

    x_api_base_url: str = "https://api.x.com"
    x_auth_mode: Literal["oauth1", "oauth2"] = "oauth1"
    x_consumer_key: SecretStr | None = None
    x_consumer_secret: SecretStr | None = None
    x_access_token: SecretStr | None = None
    x_access_token_secret: SecretStr | None = None
    x_bearer_token: SecretStr | None = None
    x_client_id: str | None = None
    x_client_secret: SecretStr | None = None
    x_redirect_uri: str | None = None
    x_user_id: str | None = None
    x_oauth2_scopes: str = "tweet.read tweet.write users.read offline.access media.write"

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.6-terra"
    xai_api_key: SecretStr | None = None
    xai_model: str = "grok-4.5"
    xai_base_url: str = "https://api.x.ai/v1"

    llm_mode: Literal["openai", "xai", "dual_review", "mock"] = "mock"
    draft_provider: Literal["openai", "xai", "mock"] = "mock"
    critic_provider: Literal["openai", "xai", "mock"] = "mock"
    final_provider: Literal["openai", "xai", "mock"] = "mock"
    allow_model_fallback: bool = False

    heygen_mode: Literal["disabled", "api", "plugin_manual", "mock"] = "disabled"
    heygen_api_base_url: str = "https://api.heygen.com"
    heygen_api_key: SecretStr | None = None
    heygen_avatar_id: str | None = None
    heygen_voice_id: str | None = None

    telegram_bot_token: SecretStr | None = None

    outbound_proxy_enabled: bool = False
    outbound_proxy: SecretStr | None = None

    publish_enabled: bool = False
    auto_publish: bool = False
    store_llm_payloads: bool = False
    log_level: str = "INFO"
    color_logs: bool = True

    daily_x_read_limit_usd: float | None = Field(default=None, ge=0)
    daily_x_write_limit_usd: float | None = Field(default=None, ge=0)
    daily_openai_limit_usd: float | None = Field(default=None, ge=0)
    daily_xai_limit_usd: float | None = Field(default=None, ge=0)
    daily_heygen_limit_usd: float | None = Field(default=None, ge=0)

    similarity_threshold: float = Field(default=0.82, ge=0, le=1)
    x_max_attempts: int = Field(default=4, ge=1, le=10)
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=3600)

    @field_validator("data_dir", "drafts_dir", "logs_dir", "config_dir", mode="after")
    @classmethod
    def resolve_runtime_directory(cls, value: Path) -> Path:
        """Resolve operator paths from the process working directory, not site-packages."""

        return value.expanduser().resolve()

    @field_validator(
        "admin_password",
        "secret_key",
        "supabase_publishable_key",
        "smtp_password",
        "x_consumer_key",
        "x_consumer_secret",
        "x_access_token",
        "x_access_token_secret",
        "x_bearer_token",
        "x_client_secret",
        "openai_api_key",
        "xai_api_key",
        "heygen_api_key",
        "telegram_bot_token",
        "outbound_proxy",
        mode="before",
    )
    @classmethod
    def empty_secret_is_none(cls, value: Any) -> Any:
        return None if value == "" else value

    @field_validator(
        "x_client_id",
        "supabase_url",
        "smtp_host",
        "smtp_username",
        "smtp_from_email",
        "x_redirect_uri",
        "x_user_id",
        "heygen_avatar_id",
        "heygen_voice_id",
        mode="before",
    )
    @classmethod
    def empty_text_is_none(cls, value: Any) -> Any:
        return None if value == "" else value

    @field_validator(
        "daily_x_read_limit_usd",
        "daily_x_write_limit_usd",
        "daily_openai_limit_usd",
        "daily_xai_limit_usd",
        "daily_heygen_limit_usd",
        mode="before",
    )
    @classmethod
    def empty_optional_number_is_none(cls, value: Any) -> Any:
        return None if value == "" else value

    @model_validator(mode="after")
    def enforce_safety_defaults(self) -> Settings:
        if self.auto_publish:
            raise ConfigurationError(
                "AUTO_PUBLISH=true is rejected: automatic publication is intentionally "
                "unsupported in the premoderation MVP"
            )
        if not is_loopback_host(self.app_host) and (
            self.auth_mode != "supabase"
            or self.supabase_url is None
            or self.supabase_publishable_key is None
        ):
            raise ConfigurationError(
                "Non-loopback bind requires AUTH_MODE=supabase, SUPABASE_URL, and "
                "SUPABASE_PUBLISHABLE_KEY"
            )
        if self.auth_mode == "supabase" and (
            self.supabase_url is None or self.supabase_publishable_key is None
        ):
            raise ConfigurationError(
                "AUTH_MODE=supabase requires SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY"
            )
        if (
            self.auth_mode == "local"
            and self.local_otp_delivery == "smtp"
            and (self.smtp_host is None or self.smtp_from_email is None)
        ):
            raise ConfigurationError(
                "LOCAL_OTP_DELIVERY=smtp requires SMTP_HOST and SMTP_FROM_EMAIL"
            )
        if (
            self.app_env == "production"
            and self.auth_mode == "local"
            and self.local_otp_delivery != "smtp"
        ):
            raise ConfigurationError(
                "Production local authentication requires LOCAL_OTP_DELIVERY=smtp"
            )
        if self.outbound_proxy_enabled and self.outbound_proxy is None:
            raise ConfigurationError("OUTBOUND_PROXY_ENABLED=true requires OUTBOUND_PROXY")
        if self.outbound_proxy is not None:
            try:
                normalize_proxy_url(self.outbound_proxy.get_secret_value())
            except ProxyConfigurationError as exc:
                raise ConfigurationError(f"Invalid OUTBOUND_PROXY: {exc}") from exc
        return self

    @property
    def outbound_proxy_url(self) -> str | None:
        """Return the normalized proxy URL only when explicitly enabled."""

        if not self.outbound_proxy_enabled or self.outbound_proxy is None:
            return None
        return normalize_proxy_url(self.outbound_proxy.get_secret_value())

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.drafts_dir, self.logs_dir, self.config_dir):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def x_oauth2_scope_set(self) -> frozenset[str]:
        """Return normalized scopes advertised for the OAuth2 user token."""

        return frozenset(
            scope for scope in re.split(r"[\s,]+", self.x_oauth2_scopes.strip()) if scope
        )

    def _missing_oauth1_user_credentials(self) -> list[str]:
        fields = {
            "X_CONSUMER_KEY": self.x_consumer_key,
            "X_CONSUMER_SECRET": self.x_consumer_secret,
            "X_ACCESS_TOKEN": self.x_access_token,
            "X_ACCESS_TOKEN_SECRET": self.x_access_token_secret,
        }
        return [name for name, value in fields.items() if value is None]

    def _missing_oauth2_user_credentials(self, required_scopes: frozenset[str]) -> list[str]:
        missing: list[str] = []
        if self.x_access_token is None:
            missing.append("X_ACCESS_TOKEN (OAuth2 user token)")
        missing_scopes = sorted(required_scopes - self.x_oauth2_scope_set)
        missing.extend(f"X_OAUTH2_SCOPES ({scope})" for scope in missing_scopes)
        return missing

    def missing_x_credentials(self, capability: XCapability) -> list[str]:
        """Validate credentials for one capability without making a network call."""

        if self.mock_mode:
            return []

        if capability is XCapability.APP_READ:
            if self.x_bearer_token is not None:
                return []
            required_scopes = frozenset({"tweet.read", "users.read"})
        elif capability is XCapability.HOME_TIMELINE:
            # Home timeline is a user-context endpoint. An app-only bearer token
            # is intentionally ignored even when it is otherwise valid for reads.
            required_scopes = frozenset({"tweet.read", "users.read"})
        elif capability is XCapability.POST_WRITE:
            required_scopes = frozenset({"tweet.read", "tweet.write", "users.read"})
        elif capability is XCapability.MEDIA_WRITE:
            required_scopes = frozenset({"media.write", "tweet.read", "tweet.write", "users.read"})
        elif capability is XCapability.ACTIVITY_PUBLIC:
            if self.x_bearer_token is not None:
                return []
            if self.x_auth_mode != "oauth2":
                return ["X_BEARER_TOKEN or X_AUTH_MODE=oauth2"]
            required_scopes = frozenset({"tweet.read", "users.read"})
        else:
            if self.x_auth_mode != "oauth2":
                return ["X_AUTH_MODE=oauth2 (required for private X Activity events)"]
            required_scopes = frozenset({"tweet.read", "users.read"})

        if self.x_auth_mode == "oauth1":
            return self._missing_oauth1_user_credentials()
        return self._missing_oauth2_user_credentials(required_scopes)

    def missing_x_read_credentials(self) -> list[str]:
        """Validate generic reads, for which an app-only bearer token is valid."""

        return self.missing_x_credentials(XCapability.APP_READ)

    def missing_x_home_timeline_credentials(self) -> list[str]:
        """Validate the user-context credentials required by home timeline."""

        return self.missing_x_credentials(XCapability.HOME_TIMELINE)

    def missing_x_write_credentials(self) -> list[str]:
        """Validate user-context credentials and scopes for post creation."""

        return self.missing_x_credentials(XCapability.POST_WRITE)

    def missing_x_media_write_credentials(self) -> list[str]:
        """Validate the additional OAuth2 scope required for media writes."""

        return self.missing_x_credentials(XCapability.MEDIA_WRITE)

    def missing_x_activity_credentials(self, *, private: bool) -> list[str]:
        """Validate credentials for public or private X Activity subscriptions."""

        capability = XCapability.ACTIVITY_PRIVATE if private else XCapability.ACTIVITY_PUBLIC
        return self.missing_x_credentials(capability)

    def x_activity_token(self, *, private: bool) -> str | None:
        """Return the bearer credential appropriate for the configured event privacy."""

        if private:
            if self.x_auth_mode != "oauth2" or self.x_access_token is None:
                return None
            return self.x_access_token.get_secret_value()
        if self.x_bearer_token is not None:
            return self.x_bearer_token.get_secret_value()
        if self.x_auth_mode == "oauth2" and self.x_access_token is not None:
            return self.x_access_token.get_secret_value()
        return None

    def x_capability_report(self) -> dict[str, bool]:
        """Summarize credential-only readiness without probing any endpoint."""

        app_read = not self.missing_x_credentials(XCapability.APP_READ)
        return {
            "app_read": app_read,
            # Compatibility key for older integrations. Recent Search is not used by the app.
            "recent_search": app_read,
            "home_timeline": not self.missing_x_credentials(XCapability.HOME_TIMELINE),
            "mentions": not self.missing_x_credentials(XCapability.HOME_TIMELINE),
            "create_post": not self.missing_x_credentials(XCapability.POST_WRITE),
            "media_upload": not self.missing_x_credentials(XCapability.MEDIA_WRITE),
            "activity_public": not self.missing_x_credentials(XCapability.ACTIVITY_PUBLIC),
            "activity_private": not self.missing_x_credentials(XCapability.ACTIVITY_PRIVATE),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a mapping-only YAML file without evaluating custom tags."""

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"Expected a YAML mapping in {path}")
    return value
