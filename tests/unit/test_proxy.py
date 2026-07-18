from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.domain.security import redact_secrets
from app.utils.proxy import ProxyConfigurationError, normalize_proxy_url


def test_proxy_accepts_host_first_and_credentials_first_formats() -> None:
    expected = "http://user:pass@127.0.0.1:8080"
    assert normalize_proxy_url("127.0.0.1:8080:user:pass") == expected
    assert normalize_proxy_url("user:pass:127.0.0.1:8080") == expected


def test_proxy_percent_encodes_credentials_in_full_compact_form() -> None:
    assert (
        normalize_proxy_url("proxy.example.com:3128:user@example.com:p@ss word")
        == "http://user%40example.com:p%40ss%20word@proxy.example.com:3128"
    )


def test_proxy_rejects_unsupported_or_ambiguous_values() -> None:
    with pytest.raises(ProxyConfigurationError):
        normalize_proxy_url("socks5://user:pass@127.0.0.1:1080")
    with pytest.raises(ProxyConfigurationError):
        normalize_proxy_url("not-a-proxy")


def test_settings_proxy_is_fail_closed_and_disabled_by_default() -> None:
    disabled = Settings(_env_file=None, outbound_proxy="127.0.0.1:8080:user:pass")
    assert disabled.outbound_proxy_url is None

    enabled = Settings(
        _env_file=None,
        outbound_proxy_enabled=True,
        outbound_proxy="127.0.0.1:8080:user:pass",
    )
    assert enabled.outbound_proxy_url == "http://user:pass@127.0.0.1:8080"

    with pytest.raises(ValidationError, match="requires OUTBOUND_PROXY"):
        Settings(_env_file=None, outbound_proxy_enabled=True)


def test_proxy_credentials_are_redacted_from_fields_and_urls() -> None:
    raw = "http://user:password@127.0.0.1:8080"
    assert redact_secrets({"outbound_proxy": raw})["outbound_proxy"] == "***REDACTED***"
    rendered = redact_secrets(f"proxy connection failed: {raw}")
    assert isinstance(rendered, str)
    assert "user:password" not in rendered
    assert "***REDACTED***@127.0.0.1:8080" in rendered
