"""Validated outbound HTTP proxy helpers.

The compact operator format accepts either ``host:port:user:password`` or
``user:password:host:port``.  Values are converted to an HTTP proxy URL and
must never be logged.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import quote, urlsplit, urlunsplit


class ProxyConfigurationError(ValueError):
    """Raised when an outbound proxy value is malformed or unsupported."""


def _looks_like_host(value: str) -> bool:
    if not value or any(character.isspace() for character in value):
        return False
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return "." in value or value.casefold() == "localhost"
    return True


def _validate_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise ProxyConfigurationError("Proxy port must be a number") from exc
    if not 1 <= port <= 65535:
        raise ProxyConfigurationError("Proxy port must be between 1 and 65535")
    return port


def normalize_proxy_url(raw_value: str) -> str:
    """Return a safe HTTP(S) proxy URL from URL or compact four-part input.

    Supported compact forms:
    - ``host:port:username:password`` (matches common proxy seller exports)
    - ``username:password:host:port``

    For credentials containing colons, use a full URL instead.
    """

    value = raw_value.strip()
    if not value:
        raise ProxyConfigurationError("Proxy value is empty")

    if "://" in value:
        parsed = urlsplit(value)
        if parsed.scheme.casefold() not in {"http", "https"}:
            raise ProxyConfigurationError("Only HTTP and HTTPS proxies are supported")
        if not parsed.hostname or parsed.port is None:
            raise ProxyConfigurationError("Proxy URL must include host and port")
        _validate_port(str(parsed.port))
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ProxyConfigurationError("Proxy URL must not include path, query, or fragment")
        return urlunsplit((parsed.scheme.casefold(), parsed.netloc, "", "", ""))

    parts = value.split(":")
    if len(parts) != 4:
        raise ProxyConfigurationError(
            "Use host:port:username:password, username:password:host:port, or a full URL"
        )

    if parts[1].isdigit() and _looks_like_host(parts[0]):
        host, port_text, username, password = parts
    elif parts[3].isdigit() and _looks_like_host(parts[2]):
        username, password, host, port_text = parts
    else:
        raise ProxyConfigurationError(
            "Could not determine proxy format; use host:port:username:password"
        )

    port = _validate_port(port_text)
    if not username or not password:
        raise ProxyConfigurationError("Proxy username and password are required")
    if ":" in host:
        raise ProxyConfigurationError("Use a full URL for IPv6 proxy hosts")

    encoded_user = quote(username, safe="")
    encoded_password = quote(password, safe="")
    return f"http://{encoded_user}:{encoded_password}@{host}:{port}"
