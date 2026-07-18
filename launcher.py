"""One-click local installer and launcher for Vouch.

This file uses only the Python standard library so it can create the project's
virtual environment before any third-party dependency is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from urllib.parse import unquote

from app.utils.proxy import ProxyConfigurationError, normalize_proxy_url

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
ENV_FILE = ROOT / ".env"
REQUIREMENTS = ROOT / "requirements.lock"
DESKTOP_REQUIREMENTS = ROOT / "requirements-desktop.lock"
RUNTIME_CONFIG = ROOT / "config" / "runtime.yml"
CONTENT_PROFILE_CONFIG = ROOT / "config" / "content_profile.yml"
CONTENT_SOURCES_CONFIG = ROOT / "config" / "content_sources.yml"
MARKER = VENV / ".xbot-requirements.sha256"
DESKTOP_MARKER = VENV / ".xbot-desktop-requirements.sha256"
LAUNCHER_LOG = ROOT / "logs" / "launcher.log"
SUPPORTED_MIN = (3, 12)
SUPPORTED_MAX = (3, 14)
MIN_MANUAL_X_WRITE_BUDGET_USD = 0.015


class LauncherError(RuntimeError):
    """A user-actionable launcher failure."""


@dataclass(frozen=True)
class ConfigurationValues:
    """Secret and account values entered by the local operator."""

    openai_key: str
    telegram_token: str
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from_email: str = ""
    smtp_starttls: bool = True
    xai_key: str = ""
    telegram_user_id: str = ""
    generation_provider: str = "openai"
    humanizer_skill_path: str = ""
    x_bearer: str = ""
    x_user_id: str = ""
    x_consumer_key: str = ""
    x_consumer_secret: str = ""
    x_access_token: str = ""
    x_access_token_secret: str = ""
    outbound_proxy: str = ""
    post_length_mode: str = "short"
    premium_long_posts_enabled: bool = False
    manual_x_publish_enabled: bool = False
    enterprise_quote_posts_enabled: bool = False
    daily_x_read_limit_usd: float = 0.25
    daily_x_write_limit_usd: float = 0.10
    x_write_budget_matches_read: bool = False
    x_request_delay_seconds: float = 2.0
    llm_minimum_interval_seconds: float = 30.0
    discovery_interval_preset: str = "12h"
    discovery_custom_interval_minutes: int = 10
    discovery_max_runs_per_utc_day: int = 5

    @property
    def has_oauth1_credentials(self) -> bool:
        return bool(
            self.x_consumer_key
            and self.x_consumer_secret
            and self.x_access_token
            and self.x_access_token_secret
        )

    @property
    def has_home_credentials(self) -> bool:
        return bool(self.x_user_id and self.has_oauth1_credentials)

    @property
    def has_email_delivery(self) -> bool:
        return bool(
            self.smtp_host
            and self.smtp_from_email
            and 1 <= self.smtp_port <= 65_535
            and bool(self.smtp_username) == bool(self.smtp_password)
        )


def _log(message: str) -> None:
    LAUNCHER_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with LAUNCHER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")


def _is_supported(version: tuple[int, int]) -> bool:
    return SUPPORTED_MIN <= version < SUPPORTED_MAX


def _resolve_manual_publish_budget(
    *,
    enabled: bool,
    budget_usd: float,
    confirm_increase: Callable[[float], bool],
) -> float | None:
    """Return a valid budget, or ``None`` when the operator declines the increase.

    The configurator treats the configured one-write estimate as a minimum daily
    allowance.  A low value is recoverable UX, not a terminal validation error:
    the operator may accept the minimum and continue saving.
    """

    if budget_usd < 0:
        raise ValueError("Daily X write budget must be non-negative.")
    if not enabled or budget_usd >= MIN_MANUAL_X_WRITE_BUDGET_USD:
        return budget_usd
    if confirm_increase(MIN_MANUAL_X_WRITE_BUDGET_USD):
        return MIN_MANUAL_X_WRITE_BUDGET_USD
    return None


def _probe(command: Sequence[str]) -> bool:
    try:
        result = subprocess.run(
            [
                *command,
                "-c",
                (
                    "import sys; raise SystemExit("
                    "0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)"
                ),
            ],
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def find_base_python() -> list[str]:
    """Find CPython 3.12/3.13 without relying on shell activation."""

    candidates: list[list[str]] = []
    if _is_supported(sys.version_info[:2]):
        candidates.append([sys.executable])
    if os.name == "nt" and shutil.which("py"):
        candidates.extend((["py", "-3.13"], ["py", "-3.12"]))
    for executable in ("python3.13", "python3.12", "python"):
        if shutil.which(executable):
            candidates.append([executable])
    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _probe(candidate):
            return candidate
    raise LauncherError(
        "CPython 3.12 or 3.13 was not found. Install Python from python.org, "
        "enable Add Python to PATH, and run START_VOUCH.bat again."
    )


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _run_streaming(command: Sequence[str], *, label: str) -> None:
    """Run a non-secret command while mirroring output to console and launcher log."""

    _log(f"START {label}: {' '.join(command)}")
    process = subprocess.Popen(
        list(command),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
    )
    if process.stdout is None:  # pragma: no cover - PIPE is guaranteed above
        raise LauncherError(f"No output stream for {label}")
    for line in process.stdout:
        print(line, end="")
        _log(f"{label}: {line.rstrip()}")
    code = process.wait()
    _log(f"END {label}: exit={code}")
    if code != 0:
        raise LauncherError(f"Stage '{label}' exited with code {code}. Details: {LAUNCHER_LOG}")


def ensure_venv(base_python: Sequence[str]) -> Path:
    target = venv_python()
    if not target.is_file():
        print("Creating local Python environment .venv...")
        _run_streaming([*base_python, "-m", "venv", str(VENV)], label="create-venv")
    if not target.is_file():
        raise LauncherError("Could not create .venv: environment Python executable is missing")
    return target


def dependency_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (REQUIREMENTS, ROOT / "pyproject.toml"):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def ensure_dependencies(python: Path) -> None:
    expected = dependency_fingerprint()
    installed = MARKER.read_text(encoding="utf-8").strip() if MARKER.is_file() else ""
    if installed == expected:
        print("Dependencies are already installed.")
        return
    print("Installing locked dependencies...")
    _run_streaming(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(REQUIREMENTS),
        ],
        label="install-dependencies",
    )
    MARKER.write_text(expected + "\n", encoding="utf-8")


def ensure_desktop_dependencies(python: Path) -> None:
    """Install the small desktop-only dependency set when requested."""

    digest = hashlib.sha256(DESKTOP_REQUIREMENTS.read_bytes()).hexdigest()
    installed = (
        DESKTOP_MARKER.read_text(encoding="utf-8").strip() if DESKTOP_MARKER.is_file() else ""
    )
    if installed == digest:
        return
    print("Installing desktop interface dependencies...")
    _run_streaming(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(DESKTOP_REQUIREMENTS),
        ],
        label="install-desktop-dependencies",
    )
    DESKTOP_MARKER.write_text(digest + "\n", encoding="utf-8")


def _plain(prompt: str, *, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or default


def _dotenv_value(value: str) -> str:
    if not value:
        return ""
    return json.dumps(value, ensure_ascii=False)


def _read_existing_env(path: Path | None = None) -> dict[str, str]:
    """Read the small local dotenv file written by this launcher."""

    path = ENV_FILE if path is None else path
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        value = raw_value.strip()
        if value.startswith('"'):
            with suppress(json.JSONDecodeError):
                value = str(json.loads(value))
        result[key.strip()] = value
    return result


POST_LENGTH_MODES = ("short", "medium", "longread", "random")
GENERATION_PROVIDERS = ("openai", "xai")
DISCOVERY_INTERVAL_PRESETS = ("1h", "3h", "6h", "12h", "custom")


def _read_yaml_scalar(path: Path, section: str, field: str, default: str) -> str:
    """Read one top-level YAML scalar without importing third-party packages."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return default
    in_section = False
    for line in lines:
        if line and not line.startswith((" ", "\t", "#")):
            in_section = line.strip() == f"{section}:"
            continue
        if in_section and line.strip().startswith(f"{field}:"):
            return line.split(":", 1)[1].strip().strip("'\"") or default
    return default


def _read_runtime_scalar(section: str, field: str, default: str) -> str:
    return _read_yaml_scalar(RUNTIME_CONFIG, section, field, default)


def _read_runtime_float(section: str, field: str, default: float) -> float:
    try:
        return float(_read_runtime_scalar(section, field, str(default)))
    except ValueError:
        return default


def _read_runtime_bool(section: str, field: str, default: bool) -> bool:
    value = _read_runtime_scalar(section, field, "true" if default else "false").casefold()
    if value in {"true", "yes", "1", "on"}:
        return True
    if value in {"false", "no", "0", "off"}:
        return False
    return default


def _write_yaml_scalars(path: Path, section: str, values: Mapping[str, str]) -> None:
    """Update scalar fields and safely replace legacy nested YAML values."""

    try:
        original = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LauncherError(f"Could not read {path}") from exc
    newline = "\r\n" if "\r\n" in original else "\n"
    lines = original.splitlines()

    def section_bounds() -> tuple[int, int]:
        section_index: int | None = None
        section_end = len(lines)
        for index, line in enumerate(lines):
            if line == f"{section}:":
                section_index = index
                continue
            if section_index is not None and line and not line.startswith((" ", "\t", "#")):
                section_end = index
                break
        if section_index is None:
            if lines and lines[-1].strip():
                lines.append("")
            section_index = len(lines)
            lines.append(f"{section}:")
            section_end = len(lines)
        return section_index, section_end

    for field, value in values.items():
        section_index, section_end = section_bounds()
        position: int | None = None
        for index in range(section_index + 1, section_end):
            if lines[index].strip().startswith(f"{field}:"):
                position = index
                break
        rendered = f"  {field}: {value}"
        if position is None:
            lines.insert(section_end, rendered)
            continue
        base_indent = len(lines[position]) - len(lines[position].lstrip())
        lines[position] = rendered
        child_end = position + 1
        while child_end < len(lines):
            candidate = lines[child_end]
            if not candidate.strip():
                child_end += 1
                continue
            indent = len(candidate) - len(candidate.lstrip())
            if indent <= base_indent:
                break
            child_end += 1
        del lines[position + 1 : child_end]

    path.write_text(newline.join(lines) + newline, encoding="utf-8")


def _write_runtime_scalars(section: str, values: Mapping[str, str]) -> None:
    _write_yaml_scalars(RUNTIME_CONFIG, section, values)


def _read_discovery_schedule() -> tuple[str, int, int]:
    preset = _read_yaml_scalar(CONTENT_SOURCES_CONFIG, "automatic_discovery", "interval_preset", "")
    if preset not in DISCOVERY_INTERVAL_PRESETS:
        legacy_hours = _read_yaml_scalar(
            CONTENT_SOURCES_CONFIG, "automatic_discovery", "interval_hours", "12"
        )
        preset = (
            f"{legacy_hours}h" if f"{legacy_hours}h" in DISCOVERY_INTERVAL_PRESETS else "custom"
        )
    raw_minutes = _read_yaml_scalar(
        CONTENT_SOURCES_CONFIG, "automatic_discovery", "custom_interval_minutes", "10"
    )
    try:
        minutes = int(raw_minutes)
    except ValueError:
        minutes = 10
    raw_max_runs = _read_yaml_scalar(
        CONTENT_SOURCES_CONFIG, "automatic_discovery", "max_runs_per_utc_day", "5"
    )
    try:
        max_runs = int(raw_max_runs)
    except ValueError:
        max_runs = 5
    return preset, min(max(minutes, 10), 10_080), min(max(max_runs, 1), 144)


def _write_discovery_schedule(preset: str, custom_minutes: int, max_runs: int) -> None:
    if preset not in DISCOVERY_INTERVAL_PRESETS:
        raise LauncherError(f"Unknown automatic-discovery interval: {preset}")
    if not 10 <= custom_minutes <= 10_080:
        raise LauncherError("Custom discovery interval must be between 10 and 10080 minutes")
    if not 1 <= max_runs <= 144:
        raise LauncherError("Automatic discovery daily limit must be between 1 and 144")
    _write_yaml_scalars(
        CONTENT_SOURCES_CONFIG,
        "automatic_discovery",
        {
            "interval_preset": preset,
            "custom_interval_minutes": str(custom_minutes),
            "max_runs_per_utc_day": str(max_runs),
        },
    )


def _read_post_length_mode() -> str:
    """Read the non-secret mode from runtime.yml without requiring PyYAML."""

    mode = _read_runtime_scalar("generation", "post_length_mode", "short")
    return mode if mode in POST_LENGTH_MODES else "short"


def _write_post_length_mode(mode: str) -> None:
    """Update only generation.post_length_mode while preserving the YAML file."""

    if mode not in POST_LENGTH_MODES:
        raise LauncherError(f"Unknown post length mode: {mode}")
    _write_runtime_scalars("generation", {"post_length_mode": mode})


def _read_premium_long_posts_enabled() -> bool:
    tier = _read_yaml_scalar(CONTENT_PROFILE_CONFIG, "account", "x_account_tier", "standard")
    enabled = _read_yaml_scalar(
        CONTENT_PROFILE_CONFIG, "account", "premium_long_posts_enabled", "false"
    )
    return tier == "premium" and enabled.casefold() == "true"


def _write_premium_long_posts_enabled(enabled: bool) -> None:
    _write_yaml_scalars(
        CONTENT_PROFILE_CONFIG,
        "account",
        {
            "x_account_tier": "premium" if enabled else "standard",
            "premium_long_posts_enabled": "true" if enabled else "false",
            "premium_long_post_max_chars": "25000",
        },
    )


def _read_generation_provider() -> str:
    provider = _read_runtime_scalar("generation", "provider", "openai")
    return provider if provider in GENERATION_PROVIDERS else "openai"


def _write_generation_provider(provider: str) -> None:
    if provider not in GENERATION_PROVIDERS:
        raise LauncherError(f"Unknown AI provider: {provider}")
    _write_runtime_scalars("generation", {"provider": provider})


def _read_telegram_operator_id() -> str:
    value = _read_runtime_scalar("telegram", "operator_user_id", "")
    return value if value.isdigit() and int(value) > 0 else ""


def _write_telegram_operator_id(user_id: str) -> None:
    if not user_id.isdigit() or int(user_id) <= 0:
        raise LauncherError("Telegram ID must be a positive integer")
    _write_runtime_scalars(
        "telegram",
        {
            "operator_user_id": user_id,
            "allowed_user_ids": "[]",
            "review_chat_id": "null",
        },
    )


def _write_request_pacing(x_delay: float, llm_interval: float) -> None:
    _write_runtime_scalars(
        "request_pacing",
        {
            "x_request_delay_seconds": f"{x_delay:g}",
            "llm_minimum_interval_seconds": f"{llm_interval:g}",
        },
    )


def _configuration_from_mapping(values: Mapping[str, str]) -> ConfigurationValues:
    discovery_preset, discovery_custom_minutes, discovery_max_runs = _read_discovery_schedule()
    try:
        smtp_port = int(values.get("SMTP_PORT", "587") or "587")
    except ValueError:
        smtp_port = 0
    return ConfigurationValues(
        openai_key=values.get("OPENAI_API_KEY", ""),
        telegram_token=values.get("TELEGRAM_BOT_TOKEN", ""),
        smtp_host=values.get("SMTP_HOST", ""),
        smtp_port=smtp_port,
        smtp_username=values.get("SMTP_USERNAME", ""),
        smtp_password=values.get("SMTP_PASSWORD", ""),
        smtp_from_email=values.get("SMTP_FROM_EMAIL", ""),
        smtp_starttls=(
            values.get("SMTP_STARTTLS", "true").casefold() in {"true", "1", "yes", "on"}
        ),
        xai_key=values.get("XAI_API_KEY", ""),
        telegram_user_id=_read_telegram_operator_id(),
        generation_provider=_read_generation_provider(),
        humanizer_skill_path=(
            ""
            if (path := _read_runtime_scalar("generation", "humanizer_skill_path", ""))
            in {"null", "none", "~"}
            else path
        ),
        x_bearer=values.get("X_BEARER_TOKEN", ""),
        x_user_id=values.get("X_USER_ID", ""),
        x_consumer_key=values.get("X_CONSUMER_KEY", ""),
        x_consumer_secret=values.get("X_CONSUMER_SECRET", ""),
        x_access_token=values.get("X_ACCESS_TOKEN", ""),
        x_access_token_secret=values.get("X_ACCESS_TOKEN_SECRET", ""),
        outbound_proxy=values.get("OUTBOUND_PROXY", ""),
        post_length_mode=_read_post_length_mode(),
        premium_long_posts_enabled=_read_premium_long_posts_enabled(),
        manual_x_publish_enabled=(
            values.get("PUBLISH_ENABLED", "false").casefold() in {"true", "1", "yes", "on"}
            and _read_runtime_bool("publication", "manual_x_publish_enabled", False)
        ),
        enterprise_quote_posts_enabled=_read_runtime_bool(
            "publication", "enterprise_quote_posts_enabled", False
        ),
        daily_x_read_limit_usd=float(values.get("DAILY_X_READ_LIMIT_USD", "0.25") or "0.25"),
        daily_x_write_limit_usd=float(values.get("DAILY_X_WRITE_LIMIT_USD", "0.10") or "0.10"),
        x_write_budget_matches_read=(
            values.get("X_WRITE_BUDGET_MATCH_READ", "false").casefold()
            in {"true", "1", "yes", "on"}
        ),
        x_request_delay_seconds=_read_runtime_float(
            "request_pacing", "x_request_delay_seconds", 2.0
        ),
        llm_minimum_interval_seconds=_read_runtime_float(
            "request_pacing", "llm_minimum_interval_seconds", 30.0
        ),
        discovery_interval_preset=discovery_preset,
        discovery_custom_interval_minutes=discovery_custom_minutes,
        discovery_max_runs_per_utc_day=discovery_max_runs,
    )


def _missing_configuration_fields(values: Mapping[str, str]) -> tuple[str, ...]:
    """Return required settings missing from an existing local installation."""

    configuration = _configuration_from_mapping(values)
    missing: list[str] = []
    if configuration.generation_provider == "openai" and not configuration.openai_key:
        missing.append("OPENAI_API_KEY")
    if configuration.generation_provider == "xai" and not configuration.xai_key:
        missing.append("XAI_API_KEY")
    if values.get("LOCAL_OTP_DELIVERY", "preview").casefold() != "smtp":
        missing.append("email sign-in delivery")
    elif not configuration.has_email_delivery:
        missing.append("SMTP email delivery")
    if _read_runtime_bool("telegram", "enabled", False):
        if not configuration.telegram_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not configuration.telegram_user_id:
            missing.append("telegram.operator_user_id")
    return tuple(missing)


def _collect_configuration_console(existing: ConfigurationValues) -> ConfigurationValues:
    """Paste-friendly fallback when the Windows GUI cannot be opened."""

    print("\nThe graphical setup window is unavailable.")
    print("The console fallback shows characters on screen but supports paste.")
    print("Close the window after setup and do not capture its contents in screenshots.\n")

    def ask(prompt: str, current: str = "", *, required: bool = False) -> str:
        while True:
            suffix = " (press Enter to keep the current value)" if current else ""
            value = input(f"{prompt}{suffix}: ").strip()
            if value:
                return value
            if current:
                return current
            if not required:
                return ""
            print("This field is required.")

    configure_home_default = "y" if existing.has_home_credentials else "N"
    configure_home = _plain(
        "Configure OAuth 1.0a for home timeline? y/N",
        default=configure_home_default,
    ).casefold() in {"y", "yes"}
    provider_prompt = "Text AI provider: openai or xai"
    generation_provider = ask(
        provider_prompt,
        existing.generation_provider or "openai",
        required=True,
    ).casefold()
    while generation_provider not in GENERATION_PROVIDERS:
        print("Allowed values: openai, xai.")
        generation_provider = ask(provider_prompt, "openai", required=True).casefold()
    mode_prompt = "Post length mode: short, medium, longread, or random"
    post_length_mode = ask(mode_prompt, existing.post_length_mode or "short", required=True)
    while post_length_mode not in POST_LENGTH_MODES:
        print("Allowed values: short, medium, longread, random.")
        post_length_mode = ask(mode_prompt, "short", required=True)
    premium_default = "y" if existing.premium_long_posts_enabled else "N"
    premium_long_posts_enabled = _plain(
        "Does this X account have Premium long-post access? y/N",
        default=premium_default,
    ).casefold() in {"y", "yes"}
    if post_length_mode in {"medium", "longread", "random"} and not premium_long_posts_enabled:
        raise LauncherError(
            "The selected post length may exceed 280 characters. Enable X Premium long-post "
            "access in CONFIGURE_VOUCH.bat or choose short."
        )

    def ask_delay(prompt: str, current: float, maximum: float) -> float:
        while True:
            raw = ask(prompt, f"{current:g}", required=True)
            try:
                value = float(raw.replace(",", "."))
            except ValueError:
                print("Enter a number of seconds.")
                continue
            if 0 <= value <= maximum:
                return value
            print(f"Allowed range: 0-{maximum:g} seconds.")

    x_delay = ask_delay("Delay between X requests, seconds", existing.x_request_delay_seconds, 60.0)
    llm_interval = ask_delay(
        "Minimum interval between AI requests, seconds",
        existing.llm_minimum_interval_seconds,
        3600.0,
    )
    discovery_prompt = "Automatic discovery interval: 1h, 3h, 6h, 12h, or custom"
    discovery_preset = ask(
        discovery_prompt,
        existing.discovery_interval_preset or "12h",
        required=True,
    ).casefold()
    while discovery_preset not in DISCOVERY_INTERVAL_PRESETS:
        print("Allowed values: 1h, 3h, 6h, 12h, custom.")
        discovery_preset = ask(discovery_prompt, "12h", required=True).casefold()
    discovery_custom_minutes = existing.discovery_custom_interval_minutes
    if discovery_preset == "custom":
        while True:
            raw = ask(
                "Custom automatic discovery interval, minutes (10-10080)",
                str(existing.discovery_custom_interval_minutes),
                required=True,
            )
            try:
                discovery_custom_minutes = int(raw)
            except ValueError:
                print("Enter a whole number of minutes.")
                continue
            if 10 <= discovery_custom_minutes <= 10_080:
                break
            print("Allowed range: 10-10080 minutes.")
    while True:
        raw_max_runs = ask(
            "Maximum successful automatic discovery runs per UTC day (1-144)",
            str(existing.discovery_max_runs_per_utc_day),
            required=True,
        )
        try:
            discovery_max_runs = int(raw_max_runs)
        except ValueError:
            print("Enter a whole number.")
            continue
        if 1 <= discovery_max_runs <= 144:
            break
        print("Allowed range: 1-144 runs per UTC day.")
    openai_key = ask(
        "OPENAI_API_KEY" + (" (optional)" if generation_provider == "xai" else ""),
        existing.openai_key,
        required=generation_provider == "openai",
    )
    xai_key = ask(
        "XAI_API_KEY" + (" (optional)" if generation_provider == "openai" else ""),
        existing.xai_key,
        required=generation_provider == "xai",
    )
    humanizer_skill_path = ask(
        "Personal Humanizer SKILL.md path (optional; blank uses bundled runtime)",
        existing.humanizer_skill_path,
    )
    print("\nEmail sign-in delivery (stored only in the local .env file).")
    smtp_host = ask("SMTP host", existing.smtp_host, required=True)
    while True:
        raw_smtp_port = ask("SMTP port", str(existing.smtp_port), required=True)
        try:
            smtp_port = int(raw_smtp_port)
        except ValueError:
            print("Enter a whole-number SMTP port.")
            continue
        if 1 <= smtp_port <= 65_535:
            break
        print("SMTP port must be between 1 and 65535.")
    smtp_username = ask(
        "SMTP username (optional for a trusted local relay)", existing.smtp_username
    )
    smtp_password = ask(
        "SMTP password/app password" + (" (optional)" if not smtp_username else ""),
        existing.smtp_password,
        required=bool(smtp_username),
    )
    if bool(smtp_username) != bool(smtp_password):
        raise LauncherError("SMTP username and password must either both be set or both be blank.")
    smtp_from_email = ask("From email", existing.smtp_from_email, required=True)
    smtp_starttls = _plain(
        "Use STARTTLS for SMTP? y/N",
        default="y" if existing.smtp_starttls else "N",
    ).casefold() in {"y", "yes"}
    manual_publish_default = "y" if existing.manual_x_publish_enabled else "N"
    manual_x_publish_enabled = _plain(
        "Enable manual X publishing after Telegram confirmation? y/N",
        default=manual_publish_default,
    ).casefold() in {"y", "yes"}
    enterprise_quote_default = "y" if existing.enterprise_quote_posts_enabled else "N"
    enterprise_quote_posts_enabled = False
    if manual_x_publish_enabled:
        enterprise_quote_posts_enabled = _plain(
            "Does this account have X Enterprise quote-post API access? y/N",
            default=enterprise_quote_default,
        ).casefold() in {"y", "yes"}
    while True:
        raw_read_budget = ask(
            "Local daily X read safety budget in USD (separate from X Console spend cap)",
            f"{existing.daily_x_read_limit_usd:g}",
            required=True,
        )
        try:
            daily_x_read_limit_usd = float(raw_read_budget.replace(",", "."))
        except ValueError:
            print("Enter a positive number.")
            continue
        if daily_x_read_limit_usd > 0:
            break
        print("The local X read safety budget must be greater than zero.")

    match_write_default = "y" if existing.x_write_budget_matches_read else "N"
    x_write_budget_matches_read = False
    if manual_x_publish_enabled:
        x_write_budget_matches_read = _plain(
            "Use the same local budget for X writes as X reads? y/N",
            default=match_write_default,
        ).casefold() in {"y", "yes"}
    if x_write_budget_matches_read:
        daily_x_write_limit_usd = daily_x_read_limit_usd
    else:
        while True:
            raw_budget = ask(
                "Daily X write budget in USD",
                f"{existing.daily_x_write_limit_usd:g}",
                required=True,
            )
            try:
                daily_x_write_limit_usd = float(raw_budget.replace(",", "."))
            except ValueError:
                print("Enter a non-negative number.")
                continue
            minimum_budget = MIN_MANUAL_X_WRITE_BUDGET_USD if manual_x_publish_enabled else 0
            if daily_x_write_limit_usd >= minimum_budget:
                break
            print(
                "Manual publishing requires at least the configured one-post estimate "
                f"({MIN_MANUAL_X_WRITE_BUDGET_USD:.3f} USD)."
            )

    telegram_user_id = ask(
        "Telegram user ID",
        existing.telegram_user_id,
        required=True,
    )
    while not telegram_user_id.isdigit() or int(telegram_user_id) <= 0:
        print("Telegram ID must be a positive integer.")
        telegram_user_id = ask("Telegram user ID", existing.telegram_user_id, required=True)
    values = ConfigurationValues(
        openai_key=openai_key,
        telegram_token=ask(
            "TELEGRAM_BOT_TOKEN",
            existing.telegram_token,
            required=True,
        ),
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_username=smtp_username,
        smtp_password=smtp_password,
        smtp_from_email=smtp_from_email,
        smtp_starttls=smtp_starttls,
        xai_key=xai_key,
        telegram_user_id=telegram_user_id,
        generation_provider=generation_provider,
        humanizer_skill_path=humanizer_skill_path,
        x_bearer=ask("X_BEARER_TOKEN (optional)", existing.x_bearer),
        x_user_id=ask("X_USER_ID (optional)", existing.x_user_id),
        x_consumer_key=existing.x_consumer_key,
        x_consumer_secret=existing.x_consumer_secret,
        x_access_token=existing.x_access_token,
        x_access_token_secret=existing.x_access_token_secret,
        outbound_proxy=ask(
            "Proxy (optional; host:port:login:password)",
            existing.outbound_proxy,
        ),
        post_length_mode=post_length_mode,
        premium_long_posts_enabled=premium_long_posts_enabled,
        manual_x_publish_enabled=manual_x_publish_enabled,
        enterprise_quote_posts_enabled=enterprise_quote_posts_enabled,
        daily_x_read_limit_usd=daily_x_read_limit_usd,
        daily_x_write_limit_usd=daily_x_write_limit_usd,
        x_write_budget_matches_read=x_write_budget_matches_read,
        x_request_delay_seconds=x_delay,
        llm_minimum_interval_seconds=llm_interval,
        discovery_interval_preset=discovery_preset,
        discovery_custom_interval_minutes=discovery_custom_minutes,
        discovery_max_runs_per_utc_day=discovery_max_runs,
    )
    if not configure_home and not manual_x_publish_enabled:
        return values
    return replace(
        values,
        x_consumer_key=ask("X_CONSUMER_KEY", existing.x_consumer_key, required=True),
        x_consumer_secret=ask("X_CONSUMER_SECRET", existing.x_consumer_secret, required=True),
        x_access_token=ask("X_ACCESS_TOKEN", existing.x_access_token, required=True),
        x_access_token_secret=ask(
            "X_ACCESS_TOKEN_SECRET", existing.x_access_token_secret, required=True
        ),
    )


def _collect_configuration_gui(existing: ConfigurationValues) -> ConfigurationValues | None:
    """Open the grouped Windows setup window; return None when cancelled."""

    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as exc:  # pragma: no cover - depends on local Python build
        raise LauncherError("tkinter is unavailable in the installed Python") from exc

    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - depends on desktop availability
        raise LauncherError(f"Could not open the setup window: {exc}") from exc

    root.title("Vouch - Setup")
    root.resizable(True, True)
    root.minsize(980, 680)

    bg = "#f7f6f3"
    sidebar_bg = "#fbfaf7"
    card_bg = "#ffffff"
    text_color = "#2f2b27"
    muted_color = "#827b73"
    border_color = "#e9e4de"
    accent = "#ef8d3c"
    accent_soft = "#fff0e3"
    accent_dark = "#cc6f27"

    root.configure(background=bg)
    style = ttk.Style(root)
    with suppress(tk.TclError):
        style.theme_use("clam")
    style.configure("App.TFrame", background=bg)
    style.configure("Page.TFrame", background=bg)
    style.configure("Card.TLabelframe", background=card_bg, bordercolor=border_color)
    style.configure(
        "Card.TLabelframe.Label",
        background=card_bg,
        foreground=text_color,
        font=("Segoe UI Semibold", 10),
    )
    style.configure("Card.TLabel", background=card_bg, foreground=text_color)
    style.configure("Muted.Card.TLabel", background=card_bg, foreground=muted_color)
    style.configure(
        "PageTitle.TLabel", background=bg, foreground=text_color, font=("Segoe UI Semibold", 13)
    )
    style.configure("PageSubtitle.TLabel", background=bg, foreground=muted_color)
    style.configure("Footer.TFrame", background=bg)
    style.configure("Accent.TButton", background=accent, foreground="#ffffff", padding=(14, 7))
    style.map(
        "Accent.TButton",
        background=[("active", accent_dark), ("pressed", accent_dark)],
        foreground=[("disabled", "#f5d8c1")],
    )
    style.configure("Secondary.TButton", padding=(14, 7))
    style.configure("TEntry", padding=4)
    style.configure("TCombobox", padding=3)

    result: list[ConfigurationValues] = []
    variables = {
        "openai_key": tk.StringVar(value=existing.openai_key),
        "xai_key": tk.StringVar(value=existing.xai_key),
        "telegram_token": tk.StringVar(value=existing.telegram_token),
        "telegram_user_id": tk.StringVar(value=existing.telegram_user_id),
        "smtp_host": tk.StringVar(value=existing.smtp_host),
        "smtp_port": tk.StringVar(value=str(existing.smtp_port)),
        "smtp_username": tk.StringVar(value=existing.smtp_username),
        "smtp_password": tk.StringVar(value=existing.smtp_password),
        "smtp_from_email": tk.StringVar(value=existing.smtp_from_email),
        "smtp_starttls": tk.BooleanVar(value=existing.smtp_starttls),
        "generation_provider": tk.StringVar(value=existing.generation_provider or "openai"),
        "humanizer_skill_path": tk.StringVar(value=existing.humanizer_skill_path),
        "x_bearer": tk.StringVar(value=existing.x_bearer),
        "x_user_id": tk.StringVar(value=existing.x_user_id),
        "x_consumer_key": tk.StringVar(value=existing.x_consumer_key),
        "x_consumer_secret": tk.StringVar(value=existing.x_consumer_secret),
        "x_access_token": tk.StringVar(value=existing.x_access_token),
        "x_access_token_secret": tk.StringVar(value=existing.x_access_token_secret),
        "outbound_proxy": tk.StringVar(value=existing.outbound_proxy),
        "post_length_mode": tk.StringVar(value=existing.post_length_mode or "short"),
        "premium_long_posts_enabled": tk.BooleanVar(value=existing.premium_long_posts_enabled),
        "manual_x_publish_enabled": tk.BooleanVar(value=existing.manual_x_publish_enabled),
        "enterprise_quote_posts_enabled": tk.BooleanVar(
            value=existing.enterprise_quote_posts_enabled
        ),
        "daily_x_read_limit_usd": tk.StringVar(value=f"{existing.daily_x_read_limit_usd:g}"),
        "daily_x_write_limit_usd": tk.StringVar(value=f"{existing.daily_x_write_limit_usd:g}"),
        "x_write_budget_matches_read": tk.BooleanVar(value=existing.x_write_budget_matches_read),
        "x_request_delay_seconds": tk.StringVar(value=f"{existing.x_request_delay_seconds:g}"),
        "llm_minimum_interval_seconds": tk.StringVar(
            value=f"{existing.llm_minimum_interval_seconds:g}"
        ),
        "discovery_interval_preset": tk.StringVar(
            value=existing.discovery_interval_preset or "12h"
        ),
        "discovery_custom_interval_minutes": tk.StringVar(
            value=str(existing.discovery_custom_interval_minutes)
        ),
        "discovery_max_runs_per_utc_day": tk.StringVar(
            value=str(existing.discovery_max_runs_per_utc_day)
        ),
    }
    show_secrets = tk.BooleanVar(value=False)
    enable_home = tk.BooleanVar(value=existing.has_home_credentials)
    secret_entries: list[ttk.Entry] = []
    oauth_entries: list[ttk.Entry] = []
    custom_interval_entries: list[ttk.Entry] = []
    write_budget_entries: list[ttk.Entry] = []
    enterprise_quote_checkboxes: list[ttk.Checkbutton] = []
    focus_entries: dict[str, ttk.Entry] = {}

    shell = tk.Frame(root, bg=bg)
    shell.pack(fill="both", expand=True)

    topbar = tk.Frame(
        shell, bg=card_bg, height=50, highlightbackground=border_color, highlightthickness=1
    )
    topbar.pack(fill="x")
    topbar.pack_propagate(False)
    tk.Label(
        topbar,
        text="‹",
        bg=card_bg,
        fg=muted_color,
        font=("Segoe UI", 15),
    ).pack(side="left", padx=(16, 6))
    tk.Label(
        topbar,
        text="Vouch — Setup",
        bg=card_bg,
        fg=text_color,
        font=("Segoe UI Semibold", 10),
    ).pack(side="left")

    body = tk.Frame(shell, bg=bg)
    body.pack(fill="both", expand=True)

    sidebar = tk.Frame(
        body, bg=sidebar_bg, width=190, highlightbackground=border_color, highlightthickness=1
    )
    sidebar.pack(side="left", fill="y")
    sidebar.pack_propagate(False)

    main = tk.Frame(body, bg=bg)
    main.pack(side="left", fill="both", expand=True)

    page_header = tk.Frame(main, bg=bg)
    page_header.pack(fill="x", padx=18, pady=(16, 8))
    page_title = ttk.Label(page_header, text="General", style="PageTitle.TLabel")
    page_title.pack(anchor="w")
    page_subtitle = ttk.Label(
        page_header,
        text="All credentials are stored only in the local .env file.",
        style="PageSubtitle.TLabel",
    )
    page_subtitle.pack(anchor="w", pady=(2, 0))

    page_host = tk.Frame(main, bg=bg)
    page_host.pack(fill="both", expand=True, padx=18)
    page_host.grid_rowconfigure(0, weight=1)
    page_host.grid_columnconfigure(0, weight=1)

    pages: dict[str, tk.Frame] = {}
    nav_buttons: dict[str, tk.Button] = {}
    page_metadata = {
        "General": "Overview of the settings used by the local bot.",
        "AI & Telegram": "Language model and Telegram operator credentials.",
        "Email Sign-in": "SMTP delivery for one-time account sign-in codes.",
        "X (Twitter) Access": "Read access, OAuth credentials, and proxy settings.",
        "Discovery & Budget": "Generation length, automatic discovery, pacing, and limits.",
        "Publishing": "Manual X publishing and write-budget controls.",
        "Security & Tokens": "Review and update masked local secrets.",
        "Advanced": "Less frequently changed network and pacing settings.",
    }

    def create_page(name: str) -> tk.Frame:
        page = tk.Frame(page_host, bg=bg)
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_columnconfigure(0, weight=1)
        page.grid_columnconfigure(1, weight=1)
        pages[name] = page
        return page

    def show_page(name: str) -> None:
        pages[name].tkraise()
        page_title.configure(text=name)
        page_subtitle.configure(text=page_metadata[name])
        for label, button in nav_buttons.items():
            active = label == name
            button.configure(
                bg=accent_soft if active else sidebar_bg,
                fg=accent_dark if active else text_color,
                activebackground=accent_soft if active else "#f1eee9",
            )

    nav_items = (
        ("General", "⚙"),
        ("AI & Telegram", "✦"),
        ("Email Sign-in", "@"),
        ("X (Twitter) Access", "𝕏"),
        ("Discovery & Budget", "◷"),
        ("Publishing", "↗"),
        ("Security & Tokens", "♙"),
        ("Advanced", "☼"),
    )
    nav_wrap = tk.Frame(sidebar, bg=sidebar_bg)
    nav_wrap.pack(fill="x", padx=12, pady=(16, 8))
    for name, icon in nav_items:
        button = tk.Button(
            nav_wrap,
            text=f"  {icon}   {name}",
            command=partial(show_page, name),
            anchor="w",
            relief="flat",
            borderwidth=0,
            padx=8,
            pady=9,
            bg=sidebar_bg,
            fg=text_color,
            activeforeground=accent_dark,
            font=("Segoe UI", 9),
            cursor="hand2",
        )
        button.pack(fill="x", pady=2)
        nav_buttons[name] = button

    premium_card = tk.Frame(
        sidebar,
        bg=card_bg,
        highlightbackground=border_color,
        highlightthickness=1,
        padx=10,
        pady=9,
    )
    premium_card.pack(side="bottom", fill="x", padx=12, pady=12)
    premium_row = tk.Frame(premium_card, bg=card_bg)
    premium_row.pack(fill="x")
    tk.Label(
        premium_row,
        text="Premium access",
        bg=card_bg,
        fg=text_color,
        font=("Segoe UI Semibold", 8),
    ).pack(side="left")
    premium_badge = tk.Label(
        premium_row,
        text="Active" if variables["premium_long_posts_enabled"].get() else "Off",
        bg=accent_soft,
        fg=accent_dark,
        padx=6,
        pady=2,
        font=("Segoe UI Semibold", 7),
    )
    premium_badge.pack(side="right")
    tk.Label(
        premium_card,
        text="Long post, quote-post, and publishing controls are configured locally.",
        bg=card_bg,
        fg=muted_color,
        justify="left",
        wraplength=145,
        font=("Segoe UI", 7),
    ).pack(anchor="w", pady=(7, 0))

    def make_card(
        parent: tk.Frame, title: str, row: int, column: int, *, columnspan: int = 1
    ) -> ttk.LabelFrame:
        card = ttk.LabelFrame(parent, text=title, style="Card.TLabelframe", padding=(10, 8))
        card.grid(
            row=row,
            column=column,
            columnspan=columnspan,
            sticky="nsew",
            padx=(0 if column == 0 else 6, 6 if column == 0 else 0),
            pady=(0, 10),
        )
        card.columnconfigure(1, weight=1)
        return card

    def add_entry(
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        name: str,
        *,
        secret: bool,
        oauth: bool = False,
        placeholder: str | None = None,
    ) -> ttk.Entry:
        ttk.Label(parent, text=label, style="Card.TLabel").grid(
            row=row, column=0, sticky="w", padx=(0, 10), pady=3
        )
        entry = ttk.Entry(
            parent,
            textvariable=variables[name],
            width=31,
            show="*" if secret else "",
        )
        entry.grid(row=row, column=1, sticky="ew", pady=3)
        focus_entries.setdefault(name, entry)
        menu = tk.Menu(entry, tearoff=False)
        menu.add_command(label="Cut", command=lambda: entry.event_generate("<<Cut>>"))
        menu.add_command(label="Copy", command=lambda: entry.event_generate("<<Copy>>"))
        menu.add_command(label="Paste", command=lambda: entry.event_generate("<<Paste>>"))
        menu.add_separator()
        menu.add_command(label="Select all", command=lambda: entry.selection_range(0, tk.END))

        def open_menu(event: tk.Event[tk.Misc]) -> None:
            menu.tk_popup(event.x_root, event.y_root)

        entry.bind("<Button-3>", open_menu)

        def paste_from_clipboard(_event: object | None = None) -> str:
            try:
                value = root.clipboard_get()
            except tk.TclError:
                return "break"
            with suppress(tk.TclError):
                entry.delete("sel.first", "sel.last")
            entry.insert(tk.INSERT, value)
            return "break"

        for sequence in ("<Control-v>", "<Control-V>", "<Control-Shift-V>", "<Shift-Insert>"):
            entry.bind(sequence, paste_from_clipboard)

        def select_all(_event: object | None = None) -> str:
            entry.selection_range(0, tk.END)
            entry.icursor(tk.END)
            return "break"

        entry.bind("<Control-a>", select_all)
        if secret:
            secret_entries.append(entry)
        if oauth:
            oauth_entries.append(entry)
        if placeholder and not variables[name].get():
            entry.insert(0, placeholder)
            entry.configure(foreground="#aaa39b")

            def clear_placeholder(_event: object) -> None:
                if entry.get() == placeholder:
                    entry.delete(0, tk.END)
                    entry.configure(foreground=text_color)

            entry.bind("<FocusIn>", clear_placeholder, add="+")
        return entry

    def build_ai_card(parent: tk.Frame, row: int, column: int, *, columnspan: int = 1) -> None:
        card = make_card(parent, "AI & Telegram", row, column, columnspan=columnspan)
        ttk.Label(card, text="AI provider", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=3
        )
        ttk.Combobox(
            card,
            textvariable=variables["generation_provider"],
            values=GENERATION_PROVIDERS,
            state="readonly",
            width=28,
        ).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(
            card,
            text="Default: use GPT-5.6 through the configured API.",
            style="Muted.Card.TLabel",
            wraplength=330,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))
        add_entry(card, 2, "OpenAI API key", "openai_key", secret=True)
        add_entry(card, 3, "xAI API key", "xai_key", secret=True)
        add_entry(card, 4, "Telegram bot token", "telegram_token", secret=True)
        add_entry(card, 5, "Telegram user ID", "telegram_user_id", secret=False)
        add_entry(
            card,
            6,
            "Humanizer SKILL.md (optional)",
            "humanizer_skill_path",
            secret=False,
        )
        ttk.Label(
            card,
            text="Blank uses the bundled version. A file or skill directory is accepted.",
            style="Muted.Card.TLabel",
            wraplength=330,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(2, 0))

    def build_generation_card(
        parent: tk.Frame, row: int, column: int, *, columnspan: int = 1
    ) -> None:
        card = make_card(parent, "Generation & Timing", row, column, columnspan=columnspan)
        ttk.Label(card, text="Post length", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=3
        )
        ttk.Combobox(
            card,
            textvariable=variables["post_length_mode"],
            values=POST_LENGTH_MODES,
            state="readonly",
            width=28,
        ).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(
            card,
            text="Short <=280, medium 300-500, long 500-1000, random chooses before API.",
            style="Muted.Card.TLabel",
            wraplength=330,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Checkbutton(
            card,
            text="Premium long-post access",
            variable=variables["premium_long_posts_enabled"],
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=3)
        add_entry(card, 3, "Delay between X requests, sec", "x_request_delay_seconds", secret=False)
        add_entry(
            card,
            4,
            "Minimum interval between AI requests, sec",
            "llm_minimum_interval_seconds",
            secret=False,
        )
        ttk.Label(
            card,
            text="Delays reduce temporary rate limits but cannot fix insufficient_quota.",
            style="Muted.Card.TLabel",
            wraplength=330,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(2, 0))

    def build_email_card(parent: tk.Frame, row: int, column: int, *, columnspan: int = 1) -> None:
        card = make_card(parent, "Email Sign-in Delivery", row, column, columnspan=columnspan)
        add_entry(card, 0, "SMTP host", "smtp_host", secret=False)
        add_entry(card, 1, "SMTP port", "smtp_port", secret=False)
        add_entry(card, 2, "SMTP username", "smtp_username", secret=False)
        add_entry(card, 3, "SMTP password / app password", "smtp_password", secret=True)
        add_entry(card, 4, "From email", "smtp_from_email", secret=False)
        ttk.Checkbutton(
            card,
            text="Use STARTTLS",
            variable=variables["smtp_starttls"],
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Label(
            card,
            text=(
                "Used only by the local passwordless login flow. Values stay in .env and "
                "are never exposed by the web API or App Settings."
            ),
            style="Muted.Card.TLabel",
            wraplength=620,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(2, 0))

    def build_x_read_card(parent: tk.Frame, row: int, column: int, *, columnspan: int = 1) -> None:
        card = make_card(parent, "X Read Access", row, column, columnspan=columnspan)
        add_entry(card, 0, "X bearer token", "x_bearer", secret=True)
        add_entry(card, 1, "X user ID", "x_user_id", secret=False)
        add_entry(card, 2, "Proxy (optional)", "outbound_proxy", secret=True)
        ttk.Label(
            card,
            text="Proxy format: IP:PORT:LOGIN:PASSWORD or LOGIN:PASSWORD:IP:PORT",
            style="Muted.Card.TLabel",
            wraplength=330,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 0))

    def build_discovery_card(
        parent: tk.Frame, row: int, column: int, *, columnspan: int = 1
    ) -> None:
        card = make_card(parent, "Discovery & Budgets", row, column, columnspan=columnspan)
        ttk.Label(card, text="Discovery interval", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=3
        )
        ttk.Combobox(
            card,
            textvariable=variables["discovery_interval_preset"],
            values=DISCOVERY_INTERVAL_PRESETS,
            state="readonly",
            width=16,
        ).grid(row=0, column=1, sticky="ew", pady=3)
        custom = add_entry(
            card,
            1,
            "Custom interval, minutes",
            "discovery_custom_interval_minutes",
            secret=False,
        )
        custom_interval_entries.append(custom)
        add_entry(
            card,
            2,
            "Max runs per UTC day",
            "discovery_max_runs_per_utc_day",
            secret=False,
        )
        read_budget = add_entry(
            card,
            3,
            "Daily X read budget, USD",
            "daily_x_read_limit_usd",
            secret=False,
        )
        focus_entries.setdefault("daily_x_read_limit_usd", read_budget)
        ttk.Label(
            card,
            text=(
                "Presets: 1h, 3h, 6h, 12h, or custom. Custom supports 10-10080 minutes. "
                "The local budget is separate from X Console Spend Cap."
            ),
            style="Muted.Card.TLabel",
            wraplength=330,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(2, 0))

    def build_oauth_card(parent: tk.Frame, row: int, column: int, *, columnspan: int = 1) -> None:
        card = make_card(parent, "Home Timeline & OAuth", row, column, columnspan=columnspan)
        ttk.Checkbutton(
            card,
            text="Configure OAuth 1.0a for timeline and publishing",
            variable=enable_home,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        add_entry(card, 1, "X consumer key", "x_consumer_key", secret=True, oauth=True)
        add_entry(card, 2, "X consumer secret", "x_consumer_secret", secret=True, oauth=True)
        add_entry(card, 3, "X access token", "x_access_token", secret=True, oauth=True)
        add_entry(
            card,
            4,
            "X access token secret",
            "x_access_token_secret",
            secret=True,
            oauth=True,
        )
        ttk.Label(
            card,
            text=(
                "OAuth 1.0a publishing uses these four fields; X_CLIENT_ID/X_CLIENT_SECRET "
                "are not required in OAuth 1.0a mode. Set the app to Read and write, then "
                "regenerate Access Token and Secret."
            ),
            style="Muted.Card.TLabel",
            wraplength=330,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(2, 0))

    def build_publishing_card(
        parent: tk.Frame, row: int, column: int, *, columnspan: int = 1
    ) -> None:
        card = make_card(parent, "Manual Publishing", row, column, columnspan=columnspan)
        ttk.Checkbutton(
            card,
            text="Enable manual X publishing after Telegram confirmation",
            variable=variables["manual_x_publish_enabled"],
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        enterprise = ttk.Checkbutton(
            card,
            text="This account has X Enterprise quote-post API access",
            variable=variables["enterprise_quote_posts_enabled"],
        )
        enterprise.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))
        enterprise_quote_checkboxes.append(enterprise)
        ttk.Checkbutton(
            card,
            text="Use the same local budget as X read",
            variable=variables["x_write_budget_matches_read"],
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 4))
        write_budget = add_entry(
            card,
            3,
            "Daily X write budget, USD",
            "daily_x_write_limit_usd",
            secret=False,
        )
        write_budget_entries.append(write_budget)
        ttk.Label(
            card,
            text="Publishing remains manual and always requires a separate Telegram confirmation.",
            style="Muted.Card.TLabel",
            wraplength=330,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(2, 0))

    general_page = create_page("General")
    build_ai_card(general_page, 0, 0)
    build_generation_card(general_page, 0, 1)
    build_x_read_card(general_page, 1, 0)
    build_discovery_card(general_page, 1, 1)
    build_oauth_card(general_page, 2, 0)
    build_publishing_card(general_page, 2, 1)

    ai_page = create_page("AI & Telegram")
    build_ai_card(ai_page, 0, 0, columnspan=2)

    email_page = create_page("Email Sign-in")
    build_email_card(email_page, 0, 0, columnspan=2)

    x_page = create_page("X (Twitter) Access")
    build_x_read_card(x_page, 0, 0)
    build_oauth_card(x_page, 0, 1)

    discovery_page = create_page("Discovery & Budget")
    build_generation_card(discovery_page, 0, 0)
    build_discovery_card(discovery_page, 0, 1)

    publishing_page = create_page("Publishing")
    build_oauth_card(publishing_page, 0, 0)
    build_publishing_card(publishing_page, 0, 1)

    security_page = create_page("Security & Tokens")
    build_ai_card(security_page, 0, 0)
    build_oauth_card(security_page, 0, 1)
    build_email_card(security_page, 1, 0, columnspan=2)
    ttk.Label(
        security_page,
        text=(
            "Secret values remain masked unless Show values is enabled. They are stored only "
            "in the local .env file and are never included in release archives."
        ),
        style="PageSubtitle.TLabel",
        wraplength=720,
    ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))

    advanced_page = create_page("Advanced")
    build_x_read_card(advanced_page, 0, 0)
    build_generation_card(advanced_page, 0, 1)
    build_discovery_card(advanced_page, 1, 0, columnspan=2)

    footer = tk.Frame(main, bg=bg)
    footer.pack(fill="x", padx=18, pady=(8, 14))
    ttk.Checkbutton(
        footer,
        text="Show saved values",
        variable=show_secrets,
    ).pack(side="left")
    ttk.Label(
        footer,
        text="Ctrl+V, Ctrl+Shift+V, Shift+Insert, and the Windows context menu are supported.",
        style="PageSubtitle.TLabel",
    ).pack(side="left", padx=(12, 0))
    buttons = ttk.Frame(footer, style="Footer.TFrame")
    buttons.pack(side="right")

    def update_secret_visibility(*_args: object) -> None:
        marker = "" if show_secrets.get() else "*"
        for entry in secret_entries:
            entry.configure(show=marker)

    def update_oauth_state(*_args: object) -> None:
        state = "normal" if enable_home.get() else "disabled"
        for entry in oauth_entries:
            entry.configure(state=state)

    def update_custom_interval_state(*_args: object) -> None:
        state = "normal" if variables["discovery_interval_preset"].get() == "custom" else "disabled"
        for entry in custom_interval_entries:
            entry.configure(state=state)

    def update_publication_state(*_args: object) -> None:
        enabled = bool(variables["manual_x_publish_enabled"].get())
        if enabled:
            enable_home.set(True)
        state = "normal" if enabled else "disabled"
        for checkbox in enterprise_quote_checkboxes:
            checkbox.configure(state=state)
        match_read = bool(variables["x_write_budget_matches_read"].get())
        entry_state = "normal" if enabled and not match_read else "disabled"
        for entry in write_budget_entries:
            entry.configure(state=entry_state)
        if enabled and match_read:
            variables["daily_x_write_limit_usd"].set(variables["daily_x_read_limit_usd"].get())
        if not enabled:
            variables["enterprise_quote_posts_enabled"].set(False)

    def update_premium_badge(*_args: object) -> None:
        enabled = variables["premium_long_posts_enabled"].get()
        premium_badge.configure(text="Active" if enabled else "Off")

    show_secrets.trace_add("write", update_secret_visibility)
    enable_home.trace_add("write", update_oauth_state)
    variables["discovery_interval_preset"].trace_add("write", update_custom_interval_state)
    variables["manual_x_publish_enabled"].trace_add("write", update_publication_state)
    variables["x_write_budget_matches_read"].trace_add("write", update_publication_state)
    variables["daily_x_read_limit_usd"].trace_add("write", update_publication_state)
    variables["premium_long_posts_enabled"].trace_add("write", update_premium_badge)
    update_secret_visibility()
    update_oauth_state()
    update_custom_interval_state()
    update_publication_state()
    update_premium_badge()

    first_entry = focus_entries["openai_key"]
    daily_x_read_budget_entry = focus_entries["daily_x_read_limit_usd"]
    daily_x_write_budget_entry = focus_entries["daily_x_write_limit_usd"]

    show_page("General")

    def save() -> None:
        candidate = ConfigurationValues(
            openai_key=variables["openai_key"].get().strip(),
            telegram_token=variables["telegram_token"].get().strip(),
            smtp_host=variables["smtp_host"].get().strip(),
            smtp_port=existing.smtp_port,
            smtp_username=variables["smtp_username"].get().strip(),
            smtp_password=variables["smtp_password"].get().strip(),
            smtp_from_email=variables["smtp_from_email"].get().strip(),
            smtp_starttls=bool(variables["smtp_starttls"].get()),
            xai_key=variables["xai_key"].get().strip(),
            telegram_user_id=variables["telegram_user_id"].get().strip(),
            generation_provider=variables["generation_provider"].get().strip(),
            humanizer_skill_path=variables["humanizer_skill_path"].get().strip(),
            x_bearer=variables["x_bearer"].get().strip(),
            x_user_id=variables["x_user_id"].get().strip(),
            x_consumer_key=(
                variables["x_consumer_key"].get().strip()
                if enable_home.get() or variables["manual_x_publish_enabled"].get()
                else ""
            ),
            x_consumer_secret=(
                variables["x_consumer_secret"].get().strip()
                if enable_home.get() or variables["manual_x_publish_enabled"].get()
                else ""
            ),
            x_access_token=(
                variables["x_access_token"].get().strip()
                if enable_home.get() or variables["manual_x_publish_enabled"].get()
                else ""
            ),
            x_access_token_secret=(
                variables["x_access_token_secret"].get().strip()
                if enable_home.get() or variables["manual_x_publish_enabled"].get()
                else ""
            ),
            outbound_proxy=variables["outbound_proxy"].get().strip(),
            post_length_mode=variables["post_length_mode"].get().strip(),
            premium_long_posts_enabled=bool(variables["premium_long_posts_enabled"].get()),
            manual_x_publish_enabled=bool(variables["manual_x_publish_enabled"].get()),
            enterprise_quote_posts_enabled=bool(variables["enterprise_quote_posts_enabled"].get()),
            x_write_budget_matches_read=bool(variables["x_write_budget_matches_read"].get()),
            daily_x_read_limit_usd=existing.daily_x_read_limit_usd,
            daily_x_write_limit_usd=existing.daily_x_write_limit_usd,
            x_request_delay_seconds=existing.x_request_delay_seconds,
            llm_minimum_interval_seconds=existing.llm_minimum_interval_seconds,
            discovery_interval_preset=variables["discovery_interval_preset"].get().strip(),
            discovery_custom_interval_minutes=existing.discovery_custom_interval_minutes,
            discovery_max_runs_per_utc_day=existing.discovery_max_runs_per_utc_day,
        )
        missing: list[str] = []
        if not candidate.telegram_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not candidate.telegram_user_id.isdigit() or int(candidate.telegram_user_id) <= 0:
            missing.append("Telegram user ID")
        if candidate.generation_provider == "openai" and not candidate.openai_key:
            missing.append("OPENAI_API_KEY")
        if candidate.generation_provider == "xai" and not candidate.xai_key:
            missing.append("XAI_API_KEY")
        if candidate.generation_provider not in GENERATION_PROVIDERS:
            missing.append("AI provider")
        if not candidate.smtp_host:
            missing.append("SMTP host")
        if not candidate.smtp_from_email:
            missing.append("From email")
        if bool(candidate.smtp_username) != bool(candidate.smtp_password):
            missing.append("both SMTP username and password")
        if missing:
            messagebox.showerror(
                "Required fields are missing",
                "Check: " + ", ".join(missing) + ".",
                parent=root,
            )
            return
        if enable_home.get() and not candidate.has_home_credentials:
            messagebox.showerror(
                "OAuth configuration is incomplete",
                "For home timeline, set X_USER_ID and all four OAuth 1.0a fields.",
                parent=root,
            )
            return
        if candidate.manual_x_publish_enabled and not candidate.has_oauth1_credentials:
            messagebox.showerror(
                "X publishing credentials are incomplete",
                "Manual publishing requires the four OAuth 1.0a key/token fields. "
                "X_CLIENT_ID and X_CLIENT_SECRET are not required in OAuth 1.0a mode. "
                "The X app must also use Read and write permissions.",
                parent=root,
            )
            return
        if candidate.post_length_mode not in POST_LENGTH_MODES:
            messagebox.showerror(
                "Invalid post length mode",
                "Choose short, medium, longread, or random.",
                parent=root,
            )
            return
        if (
            candidate.post_length_mode in {"medium", "longread", "random"}
            and not candidate.premium_long_posts_enabled
        ):
            messagebox.showerror(
                "X Premium long-post access is required",
                "Enable the Premium long-post checkbox or choose short.",
                parent=root,
            )
            return
        try:
            x_delay = float(variables["x_request_delay_seconds"].get().replace(",", "."))
            smtp_port = int(variables["smtp_port"].get())
            llm_interval = float(variables["llm_minimum_interval_seconds"].get().replace(",", "."))
            discovery_custom_minutes = int(variables["discovery_custom_interval_minutes"].get())
            discovery_max_runs = int(variables["discovery_max_runs_per_utc_day"].get())
            daily_x_read_limit_usd = float(
                variables["daily_x_read_limit_usd"].get().replace(",", ".")
            )
            daily_x_write_limit_usd = float(
                variables["daily_x_write_limit_usd"].get().replace(",", ".")
            )
        except ValueError:
            messagebox.showerror(
                "Invalid delay values",
                "Enter delays and budgets as numbers, and discovery values as whole numbers.",
                parent=root,
            )
            return
        if not 0 <= x_delay <= 60 or not 0 <= llm_interval <= 3600:
            messagebox.showerror(
                "Delay values are out of range",
                "X: 0-60 seconds; AI: 0-3600 seconds.",
                parent=root,
            )
            return
        if not 1 <= smtp_port <= 65_535:
            messagebox.showerror(
                "Invalid SMTP port",
                "SMTP port must be between 1 and 65535.",
                parent=root,
            )
            return
        if candidate.x_write_budget_matches_read:
            daily_x_write_limit_usd = daily_x_read_limit_usd
            variables["daily_x_write_limit_usd"].set(f"{daily_x_write_limit_usd:g}")
        if daily_x_read_limit_usd <= 0:
            messagebox.showerror(
                "Invalid X read budget",
                "The local daily X read safety budget must be greater than zero. "
                "It is separate from the X Console Manage Spend Cap.",
                parent=root,
            )
            daily_x_read_budget_entry.focus_set()
            daily_x_read_budget_entry.selection_range(0, tk.END)
            return
        if candidate.discovery_interval_preset not in DISCOVERY_INTERVAL_PRESETS:
            messagebox.showerror(
                "Invalid discovery interval",
                "Choose 1h, 3h, 6h, 12h, or custom.",
                parent=root,
            )
            return
        if not 10 <= discovery_custom_minutes <= 10_080:
            messagebox.showerror(
                "Invalid custom interval",
                "Custom discovery interval must be 10-10080 minutes.",
                parent=root,
            )
            return
        try:
            resolved_x_write_budget = _resolve_manual_publish_budget(
                enabled=candidate.manual_x_publish_enabled,
                budget_usd=daily_x_write_limit_usd,
                confirm_increase=lambda minimum: messagebox.askyesno(
                    "Increase X write budget?",
                    (
                        "Manual publishing needs enough daily budget for at least one "
                        f"configured write ({minimum:.3f} USD).\n\n"
                        f"Current value: {daily_x_write_limit_usd:g} USD.\n"
                        f"Set the budget to {minimum:.3f} USD and save these settings?"
                    ),
                    parent=root,
                    icon="warning",
                ),
            )
        except ValueError as exc:
            messagebox.showerror(
                "Invalid X write budget",
                str(exc),
                parent=root,
            )
            daily_x_write_budget_entry.focus_set()
            daily_x_write_budget_entry.selection_range(0, tk.END)
            return
        if resolved_x_write_budget is None:
            daily_x_write_budget_entry.focus_set()
            daily_x_write_budget_entry.selection_range(0, tk.END)
            return
        daily_x_write_limit_usd = resolved_x_write_budget
        variables["daily_x_write_limit_usd"].set(f"{daily_x_write_limit_usd:g}")
        if not 1 <= discovery_max_runs <= 144:
            messagebox.showerror(
                "Invalid daily run limit",
                "Maximum automatic discovery runs must be 1-144 per UTC day.",
                parent=root,
            )
            return
        candidate = replace(
            candidate,
            smtp_port=smtp_port,
            x_request_delay_seconds=x_delay,
            llm_minimum_interval_seconds=llm_interval,
            discovery_custom_interval_minutes=discovery_custom_minutes,
            discovery_max_runs_per_utc_day=discovery_max_runs,
            daily_x_read_limit_usd=daily_x_read_limit_usd,
            daily_x_write_limit_usd=daily_x_write_limit_usd,
            x_write_budget_matches_read=candidate.x_write_budget_matches_read,
        )
        if candidate.outbound_proxy:
            try:
                normalize_proxy_url(candidate.outbound_proxy)
            except ProxyConfigurationError as exc:
                messagebox.showerror(
                    "Invalid proxy value",
                    str(exc),
                    parent=root,
                )
                return
        result.append(candidate)
        root.destroy()

    def cancel() -> None:
        root.destroy()

    ttk.Button(buttons, text="Cancel", command=cancel, style="Secondary.TButton").grid(
        row=0, column=0, padx=(0, 8)
    )
    ttk.Button(buttons, text="Save changes", command=save, style="Accent.TButton").grid(
        row=0, column=1
    )
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.bind("<Escape>", lambda _event: cancel())
    root.bind("<Control-Return>", lambda _event: save())
    first_entry.focus_set()
    root.update_idletasks()
    width = min(max(root.winfo_reqwidth(), 1080), max(root.winfo_screenwidth() - 40, 980))
    height = min(max(root.winfo_reqheight(), 720), max(root.winfo_screenheight() - 70, 680))
    x = max((root.winfo_screenwidth() - width) // 2, 0)
    y = max((root.winfo_screenheight() - height) // 2, 0)
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.mainloop()
    return result[0] if result else None


def collect_configuration(existing: ConfigurationValues) -> ConfigurationValues:
    """Collect settings through a GUI on Windows and a pasteable fallback elsewhere."""

    if os.name == "nt":
        try:
            selected = _collect_configuration_gui(existing)
        except LauncherError as exc:
            _log(f"GUI configuration unavailable: {exc}")
            print(f"Could not open graphical setup: {exc}")
        else:
            if selected is None:
                raise LauncherError("Setup was cancelled by the user")
            return selected
    return _collect_configuration_console(existing)


def _build_env_content(
    configuration: ConfigurationValues,
    existing: Mapping[str, str] | None = None,
) -> str:
    """Build the local dotenv while preserving unrelated optional provider secrets."""

    current = existing or {}
    values: list[tuple[str, str]] = [
        ("APP_ENV", "development"),
        ("APP_HOST", "127.0.0.1"),
        ("APP_PORT", "8000"),
        ("DATABASE_URL", "sqlite:///./data/app.db"),
        ("DATA_DIR", "./data"),
        ("DRAFTS_DIR", "./drafts"),
        ("LOGS_DIR", "./logs"),
        ("CONFIG_DIR", "./config"),
        ("MOCK_MODE", "false"),
        ("AUTH_MODE", "local"),
        ("LOCAL_OTP_DELIVERY", "smtp"),
        ("SMTP_HOST", configuration.smtp_host),
        ("SMTP_PORT", str(configuration.smtp_port)),
        ("SMTP_USERNAME", configuration.smtp_username),
        ("SMTP_PASSWORD", configuration.smtp_password),
        ("SMTP_FROM_EMAIL", configuration.smtp_from_email),
        ("SMTP_STARTTLS", "true" if configuration.smtp_starttls else "false"),
        ("OPENAI_API_KEY", configuration.openai_key or current.get("OPENAI_API_KEY", "")),
        ("TELEGRAM_BOT_TOKEN", configuration.telegram_token),
        ("OUTBOUND_PROXY_ENABLED", "true" if configuration.outbound_proxy else "false"),
        ("OUTBOUND_PROXY", configuration.outbound_proxy),
        ("X_AUTH_MODE", "oauth1"),
        ("X_CONSUMER_KEY", configuration.x_consumer_key),
        ("X_CONSUMER_SECRET", configuration.x_consumer_secret),
        ("X_ACCESS_TOKEN", configuration.x_access_token),
        ("X_ACCESS_TOKEN_SECRET", configuration.x_access_token_secret),
        ("X_BEARER_TOKEN", configuration.x_bearer),
        ("X_USER_ID", configuration.x_user_id),
        ("X_CLIENT_ID", current.get("X_CLIENT_ID", "")),
        ("X_CLIENT_SECRET", current.get("X_CLIENT_SECRET", "")),
        ("X_REDIRECT_URI", current.get("X_REDIRECT_URI", "")),
        (
            "X_OAUTH2_SCOPES",
            current.get(
                "X_OAUTH2_SCOPES",
                "tweet.read tweet.write users.read offline.access media.write",
            ),
        ),
        ("XAI_API_KEY", configuration.xai_key or current.get("XAI_API_KEY", "")),
        ("HEYGEN_API_KEY", current.get("HEYGEN_API_KEY", "")),
        ("PUBLISH_ENABLED", "true" if configuration.manual_x_publish_enabled else "false"),
        ("AUTO_PUBLISH", "false"),
        ("STORE_LLM_PAYLOADS", "false"),
        ("LOG_LEVEL", current.get("LOG_LEVEL", "INFO")),
        ("COLOR_LOGS", current.get("COLOR_LOGS", "true")),
        ("DAILY_X_READ_LIMIT_USD", f"{configuration.daily_x_read_limit_usd:g}"),
        ("DAILY_X_WRITE_LIMIT_USD", f"{configuration.daily_x_write_limit_usd:g}"),
        (
            "X_WRITE_BUDGET_MATCH_READ",
            "true" if configuration.x_write_budget_matches_read else "false",
        ),
        ("DAILY_OPENAI_LIMIT_USD", current.get("DAILY_OPENAI_LIMIT_USD", "")),
        ("DAILY_XAI_LIMIT_USD", current.get("DAILY_XAI_LIMIT_USD", "")),
        ("DAILY_HEYGEN_LIMIT_USD", current.get("DAILY_HEYGEN_LIMIT_USD", "")),
    ]
    return "\n".join(f"{key}={_dotenv_value(value)}" for key, value in values) + "\n"


def configure_env(*, force: bool = False) -> None:
    """Create or repair local configuration through a paste-friendly Windows form."""

    existing_mapping = _read_existing_env()
    missing = _missing_configuration_fields(existing_mapping) if ENV_FILE.exists() else ()
    if ENV_FILE.exists() and not force and not missing:
        return
    existing = _configuration_from_mapping(existing_mapping)
    if ENV_FILE.exists():
        if missing and not force:
            print("Existing configuration needs an update: " + ", ".join(missing) + ".")
        backup = ENV_FILE.with_name(f".env.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(ENV_FILE, backup)
        print(f"Previous .env saved as {backup.name}")

    print("\nOpening the credential setup window...")
    print("Use only newly created credentials that have never been shared in chat.")
    configuration = collect_configuration(existing)
    if configuration.outbound_proxy:
        try:
            normalize_proxy_url(configuration.outbound_proxy)
        except ProxyConfigurationError as exc:
            raise LauncherError(f"Invalid proxy value: {exc}") from exc
    ENV_FILE.write_text(
        _build_env_content(configuration, existing_mapping),
        encoding="utf-8",
    )
    _write_post_length_mode(configuration.post_length_mode)
    _write_premium_long_posts_enabled(configuration.premium_long_posts_enabled)
    _write_generation_provider(configuration.generation_provider)
    _write_runtime_scalars(
        "generation",
        {
            "humanizer_skill_path": (
                json.dumps(configuration.humanizer_skill_path)
                if configuration.humanizer_skill_path
                else "null"
            ),
            "humanizer_include_references": "true",
        },
    )
    _write_telegram_operator_id(configuration.telegram_user_id)
    _write_request_pacing(
        configuration.x_request_delay_seconds,
        configuration.llm_minimum_interval_seconds,
    )
    _write_runtime_scalars(
        "publication",
        {
            "manual_x_publish_enabled": (
                "true" if configuration.manual_x_publish_enabled else "false"
            ),
            "enterprise_quote_posts_enabled": (
                "true" if configuration.enterprise_quote_posts_enabled else "false"
            ),
            "automatic_x_publish_enabled": "false",
        },
    )
    _write_discovery_schedule(
        configuration.discovery_interval_preset,
        configuration.discovery_custom_interval_minutes,
        configuration.discovery_max_runs_per_utc_day,
    )
    with suppress(OSError):
        os.chmod(ENV_FILE, 0o600)
    print("Local .env created. It is excluded from Git and release ZIPs.")
    if not configuration.x_bearer:
        print("Warning: automatic trends/search is skipped without X_BEARER_TOKEN.")
    if not configuration.has_home_credentials:
        print("Warning: home timeline is skipped without OAuth 1.0a and X_USER_ID.")


def _resolve_runtime_path(value: str, *, default: str) -> Path:
    raw = value.strip() or default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _sqlite_database_path(database_url: str) -> Path | None:
    """Resolve a file-backed SQLite URL without importing third-party packages."""

    prefix = "sqlite:///"
    if not database_url.casefold().startswith(prefix):
        return None
    raw_path = unquote(database_url[len(prefix) :].split("?", 1)[0])
    if not raw_path or raw_path == ":memory:" or raw_path.startswith("file:"):
        return None
    if os.name == "nt" and len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
        raw_path = raw_path[1:]
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _ensure_writable_directory(path: Path, *, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".xbot-write-check"
        probe.write_text("ok\n", encoding="ascii")
        probe.unlink()
    except OSError as exc:
        raise LauncherError(
            f"Could not prepare {label}: {path}. "
            "Check folder permissions, Windows controlled-folder access, "
            "and available disk space."
        ) from exc


def ensure_runtime_layout() -> None:
    """Create all local writable directories before Alembic or the app starts."""

    values = _read_existing_env()
    directories = (
        (_resolve_runtime_path(values.get("DATA_DIR", ""), default="data"), "data directory"),
        (_resolve_runtime_path(values.get("DRAFTS_DIR", ""), default="drafts"), "drafts directory"),
        (_resolve_runtime_path(values.get("LOGS_DIR", ""), default="logs"), "logs directory"),
    )
    for path, label in directories:
        _ensure_writable_directory(path, label=label)

    database_url = values.get("DATABASE_URL", "sqlite:///./data/app.db")
    database_path = _sqlite_database_path(database_url)
    if database_path is not None:
        _ensure_writable_directory(database_path.parent, label="database directory")
        _log(f"SQLite database path prepared: {database_path}")


def migrate_and_check(python: Path) -> None:
    ensure_runtime_layout()
    print("Updating the local database...")
    _run_streaming([str(python), "-m", "alembic", "upgrade", "head"], label="migrations")
    print("Checking configuration...")
    _run_streaming([str(python), "-m", "app.cli", "doctor"], label="doctor")


def start_desktop(python: Path) -> int:
    """Start the native shell with the shared local dashboard backend."""

    executable = python
    if os.name == "nt":
        pythonw = python.with_name("pythonw.exe")
        if pythonw.is_file():
            executable = pythonw
    _log("START desktop")
    desktop_log = ROOT / "logs" / "desktop.log"
    desktop_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with desktop_log.open("a", encoding="utf-8") as output:
            return subprocess.call(
                [str(executable), str(ROOT / "desktop.py")],
                cwd=ROOT,
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
                stdout=output,
                stderr=subprocess.STDOUT,
            )
    finally:
        _log("END desktop")


def start_server(python: Path) -> int:
    print("\nVouch is running.")
    print("The bot will check a bounded sample of trends and home timeline automatically.")
    print("In Telegram, use /discover or the optional /new <topic> command.")
    print("X publishing is disabled. Stop with Ctrl+C.")
    print(f"Detailed log: {ROOT / 'logs' / 'vouch.log'}\n")
    _log("START server")
    try:
        return subprocess.call(
            [str(python), "-m", "app.cli", "serve"],
            cwd=ROOT,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
        )
    finally:
        _log("END server")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configure",
        action="store_true",
        help="recreate .env through the paste-friendly local configuration form",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="create venv, install dependencies, configure and migrate, then exit",
    )
    parser.add_argument(
        "--desktop",
        action="store_true",
        help="start the native desktop shell instead of the standalone web server",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    os.chdir(ROOT)
    _log("Launcher invoked")
    try:
        base_python = find_base_python()
        python = ensure_venv(base_python)
        ensure_dependencies(python)
        if arguments.desktop:
            ensure_desktop_dependencies(python)
        configure_env(force=arguments.configure)
        migrate_and_check(python)
        if arguments.setup_only or arguments.configure:
            print("Setup complete. Use START_WEB.bat or START_DESKTOP.bat to run the bot.")
            return 0
        if arguments.desktop:
            return start_desktop(python)
        return start_server(python)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130
    except Exception as exc:
        _log(f"FATAL {type(exc).__name__}: {exc}")
        print(f"\nLaunch error: {type(exc).__name__}: {exc}")
        print(f"Details: {LAUNCHER_LOG}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
