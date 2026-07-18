"""Read-only local diagnostics; no provider write endpoint is ever called."""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from sqlalchemy import Engine, inspect, text

from app.config import ConfigurationError, Settings
from app.services.configuration import ConfigurationService
from app.services.humanizer_runtime import HumanizerRuntimeLoader

CheckStatus = Literal["PASS", "WARN", "FAIL", "SKIP"]


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    name: str
    status: CheckStatus
    message: str


class DoctorService:
    def __init__(self, settings: Settings, engine: Engine) -> None:
        self.settings = settings
        self.engine = engine

    def _runtime_checks(self) -> list[DoctorCheck]:
        try:
            configuration = ConfigurationService(self.settings.config_dir).load()
            runtime = configuration.runtime
        except ConfigurationError as exc:
            return [DoctorCheck("runtime_config", "FAIL", str(exc))]

        checks = [DoctorCheck("runtime_config", "PASS", "runtime and source configs are valid")]
        if self.settings.outbound_proxy_enabled:
            checks.append(
                DoctorCheck(
                    "outbound_proxy",
                    "PASS",
                    (
                        "enabled for OpenAI, xAI, X API, X source-media downloads, "
                        "Telegram, and image requests"
                    ),
                )
            )
        else:
            checks.append(DoctorCheck("outbound_proxy", "SKIP", "disabled"))
        generation = runtime.generation
        length_mode = generation.post_length_mode.value
        long_mode = length_mode in {"medium", "longread", "random"}
        premium_ready = (
            configuration.profile.account.x_account_tier == "premium"
            and configuration.profile.account.premium_long_posts_enabled
        )
        checks.append(
            DoctorCheck(
                "post_length_mode",
                "PASS" if not long_mode or premium_ready else "FAIL",
                (
                    f"{length_mode}; short<=280, medium=300-500, longread=500-1000"
                    if not long_mode or premium_ready
                    else (
                        f"{length_mode} requires Premium long-post access. "
                        "Run CONFIGURE_VOUCH.bat and enable the Premium checkbox, "
                        "or set account.x_account_tier=premium and "
                        "account.premium_long_posts_enabled=true in config/content_profile.yml"
                    )
                ),
            )
        )
        pacing = runtime.request_pacing
        checks.append(
            DoctorCheck(
                "request_pacing",
                "PASS",
                (
                    f"X delay={pacing.x_request_delay_seconds:g}s; "
                    f"X temporary retries={pacing.x_temporary_error_max_retries}; "
                    f"AI minimum interval={pacing.llm_minimum_interval_seconds:g}s; "
                    f"temporary 429 retries={pacing.llm_rate_limit_max_retries}; "
                    "insufficient_quota retries=0"
                ),
            )
        )
        if not generation.humanizer_enabled or generation.humanizer_mode == "disabled":
            checks.append(DoctorCheck("humanizer", "SKIP", "disabled in config/runtime.yml"))
        else:
            configured = generation.humanizer_skill_path
            configured_path = Path(configured).expanduser() if configured else None
            expected_skill = (
                configured_path / "SKILL.md"
                if configured_path is not None and configured_path.is_dir()
                else configured_path
            )
            if configured and (expected_skill is None or not expected_skill.is_file()):
                checks.append(
                    DoctorCheck(
                        "humanizer",
                        "FAIL",
                        f"configured skill path does not exist: {expected_skill}",
                    )
                )
            else:
                try:
                    active = HumanizerRuntimeLoader(
                        configured_path=configured,
                        include_references=generation.humanizer_include_references,
                    ).load()
                    required = {
                        "runtime-contract.md",
                        "semantic-lint.md",
                        "micro-reply-examples.md",
                        "x-post-contract.md",
                    }
                    missing_refs = sorted(required - set(active.reference_files))
                    problems: list[str] = []
                    if configured and active.source != "external":
                        problems.append("configured external skill could not be loaded")
                    if active.version == "unknown":
                        problems.append("skill version is missing from frontmatter and VERSION.md")
                    if missing_refs:
                        problems.append("missing required references: " + ", ".join(missing_refs))
                    humanizer_status: CheckStatus = "PASS" if not problems else "FAIL"
                    checks.append(
                        DoctorCheck(
                            "humanizer",
                            humanizer_status,
                            (
                                f"source={active.source}; version={active.version}; "
                                f"hash={active.skill_hash[:16]}; initial/revision=same-version; "
                                f"semantic_references={active.semantic_references_enabled}"
                                if not problems
                                else "; ".join(problems)
                            ),
                        )
                    )
                except (OSError, UnicodeError, ValueError) as exc:
                    checks.append(DoctorCheck("humanizer", "FAIL", str(exc)))
        provider = "mock" if self.settings.mock_mode else runtime.generation.provider
        if provider == "mock":
            checks.append(DoctorCheck("generation_provider", "SKIP", "mock provider selected"))
        elif provider == "openai":
            enabled = runtime.providers.openai.enabled
            has_key = self.settings.openai_api_key is not None
            status: CheckStatus = "PASS" if enabled and has_key else "FAIL"
            detail = (
                f"OpenAI enabled; model={runtime.providers.openai.model}"
                if status == "PASS"
                else "enable providers.openai and set OPENAI_API_KEY"
            )
            checks.append(DoctorCheck("generation_provider", status, detail))
        elif provider == "xai":
            enabled = runtime.providers.xai.enabled
            has_key = self.settings.xai_api_key is not None
            status = "PASS" if enabled and has_key else "FAIL"
            detail = (
                f"xAI enabled; model={runtime.providers.xai.model}"
                if status == "PASS"
                else "enable providers.xai and set XAI_API_KEY"
            )
            checks.append(DoctorCheck("generation_provider", status, detail))

        checks.append(
            DoctorCheck(
                "llm_request_policy",
                "PASS",
                (
                    "remote semantic validation enabled; additional provider-backed validation "
                    "requests may run per draft; doctor/startup send no provider probe"
                    if runtime.generation.quality.remote_semantic_validation_enabled
                    else "local deterministic validation; doctor/startup send no provider probe"
                ),
            )
        )

        if runtime.providers.heygen.enabled:
            checks.append(
                DoctorCheck(
                    "heygen",
                    "FAIL",
                    "enabled in config, but the HeyGen runtime client is not "
                    "implemented in this revision",
                )
            )
        else:
            checks.append(DoctorCheck("heygen", "SKIP", "disabled in config/runtime.yml"))

        if runtime.images.enabled:
            if self.settings.mock_mode:
                checks.append(
                    DoctorCheck(
                        "image_generation",
                        "SKIP",
                        "configured but disabled while MOCK_MODE=true",
                    )
                )
            else:
                image_ready = (
                    runtime.images.provider == "openai"
                    and runtime.providers.openai.enabled
                    and self.settings.openai_api_key is not None
                )
                checks.append(
                    DoctorCheck(
                        "image_generation",
                        "PASS" if image_ready else "FAIL",
                        (
                            f"enabled; model={runtime.providers.openai.image_model}"
                            if image_ready
                            else "enable OpenAI and set OPENAI_API_KEY for image generation"
                        ),
                    )
                )
        else:
            checks.append(DoctorCheck("image_generation", "SKIP", "disabled in config/runtime.yml"))

        telegram = runtime.telegram
        if not telegram.enabled:
            checks.append(DoctorCheck("telegram", "SKIP", "disabled in config/runtime.yml"))
        else:
            missing: list[str] = []
            if self.settings.telegram_bot_token is None:
                missing.append("TELEGRAM_BOT_TOKEN")
            if not telegram.effective_allowed_user_ids or telegram.effective_review_chat_id is None:
                missing.append("telegram.operator_user_id")
            checks.append(
                DoctorCheck(
                    "telegram",
                    "PASS" if not missing else "FAIL",
                    (
                        f"enabled for {len(telegram.effective_allowed_user_ids)} allowed user(s)"
                        if not missing
                        else (
                            "missing "
                            + ", ".join(missing)
                            + "; run CONFIGURE_VOUCH.bat to complete Telegram setup"
                        )
                    ),
                )
            )

        discovery = configuration.sources.automatic_discovery
        if not discovery.enabled:
            checks.append(
                DoctorCheck("automatic_discovery", "SKIP", "disabled in content_sources.yml")
            )
        elif self.settings.mock_mode:
            checks.append(DoctorCheck("automatic_discovery", "SKIP", "mock sources selected"))
        else:
            costs = configuration.costs.x
            source_catalog = configuration.sources.sources
            active_accounts = sum(
                1
                for account in source_catalog.selected_accounts.tracked_accounts
                if account.enabled
            )
            post_units = (
                source_catalog.home_timeline.max_posts
                if source_catalog.home_timeline.enabled
                else 0
            )
            if source_catalog.selected_accounts.enabled:
                post_units += (
                    active_accounts * source_catalog.selected_accounts.max_posts_per_account
                )
            trend_attempts = configuration.runtime.request_pacing.x_temporary_error_max_retries + 1
            estimate = (
                costs.read_usd * post_units + costs.trends_request_usd * trend_attempts
                if costs.read_usd is not None and costs.trends_request_usd is not None
                else None
            )
            if estimate is None:
                checks.append(
                    DoctorCheck(
                        "automatic_discovery_budget",
                        "FAIL",
                        "configure X read and trends estimates in cost_estimates.yml",
                    )
                )
            elif self.settings.daily_x_read_limit_usd is None:
                checks.append(
                    DoctorCheck(
                        "automatic_discovery_budget",
                        "FAIL",
                        "set DAILY_X_READ_LIMIT_USD to enable bounded automatic reads",
                    )
                )
            else:
                daily_limit = Decimal(str(self.settings.daily_x_read_limit_usd))
                checks.append(
                    DoctorCheck(
                        "automatic_discovery_budget",
                        "PASS" if daily_limit >= estimate else "FAIL",
                        (
                            f"max ${estimate} per run; local daily limit ${daily_limit}; "
                            f"max {discovery.max_runs_per_utc_day} run(s)/UTC day; "
                            "separate from X Console spend cap"
                        ),
                    )
                )
            home_ready = (
                source_catalog.home_timeline.enabled
                and bool(self.settings.x_user_id)
                and not self.settings.missing_x_home_timeline_credentials()
            )
            selected_ready = (
                source_catalog.selected_accounts.enabled
                and active_accounts > 0
                and not self.settings.missing_x_read_credentials()
            )
            discovery_status: CheckStatus = "PASS" if home_ready or selected_ready else "WARN"
            if discovery_status == "PASS":
                message = (
                    f"enabled: startup scan, then every {discovery.effective_interval_minutes}m; "
                    f"home≤{source_catalog.home_timeline.max_posts}; "
                    f"selected={active_accounts} account(s), "
                    f"≤{source_catalog.selected_accounts.max_posts_per_account} each; "
                    f"trends≤{discovery.max_trends}; lookback={discovery.lookback_hours}h; "
                    "Recent Search disabled"
                )
            else:
                message = (
                    "no readable post source: configure the home timeline or enable at least "
                    "one Selected account; Recent Search is disabled"
                )
            checks.append(
                DoctorCheck(
                    "automatic_discovery",
                    discovery_status,
                    message,
                )
            )

        activity = runtime.x_activity
        if not activity.enabled:
            checks.append(DoctorCheck("x_activity", "SKIP", "disabled in config/runtime.yml"))
        elif self.settings.mock_mode:
            checks.append(
                DoctorCheck("x_activity", "SKIP", "configured but disabled while MOCK_MODE=true")
            )
        else:
            private = any(
                item.event_type == "post.mention.create" for item in activity.subscriptions
            )
            missing = self.settings.missing_x_activity_credentials(private=private)
            checks.append(
                DoctorCheck(
                    "x_activity",
                    "PASS" if not missing else "FAIL",
                    (
                        f"enabled with {len(activity.subscriptions)} desired subscription(s); "
                        "remote state was not changed"
                        if not missing
                        else "missing " + ", ".join(missing)
                    ),
                )
            )

        manual_enabled = (
            self.settings.publish_enabled and runtime.publication.manual_x_publish_enabled
        )
        if not manual_enabled:
            checks.append(
                DoctorCheck(
                    "publishing",
                    "PASS",
                    "manual X publication is disabled (premoderation-only mode)",
                )
            )
        elif self.settings.mock_mode:
            checks.append(DoctorCheck("publishing", "WARN", "mock publication gate enabled"))
        else:
            missing = self.settings.missing_x_write_credentials()
            checks.append(
                DoctorCheck(
                    "publishing",
                    "PASS" if not missing else "FAIL",
                    (
                        "manual live X writer credentials are present; identity reads do not "
                        "prove write access. For OAuth 1.0a, the app must be Read and write and "
                        "the Access Token and Secret must be regenerated after permission changes; "
                        "no write probe was sent"
                        if not missing
                        else "missing " + ", ".join(missing)
                    ),
                )
            )
        return checks

    def run(self) -> list[DoctorCheck]:
        checks: list[DoctorCheck] = []
        version_ok = (3, 12) <= sys.version_info[:2] < (3, 14)
        checks.append(
            DoctorCheck(
                "python",
                "PASS" if version_ok else "FAIL",
                f"Python {sys.version_info.major}.{sys.version_info.minor}",
            )
        )
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            checks.append(DoctorCheck("database", "PASS", "connection available"))
        except Exception:
            checks.append(DoctorCheck("database", "FAIL", "connection unavailable"))
        try:
            tables = set(inspect(self.engine).get_table_names())
            migration_ready = "alembic_version" in tables
            checks.append(
                DoctorCheck(
                    "migrations",
                    "PASS" if migration_ready else "WARN",
                    "Alembic revision present" if migration_ready else "run alembic upgrade head",
                )
            )
        except Exception:
            checks.append(DoctorCheck("migrations", "WARN", "revision could not be read"))
        for name, path in (
            ("data_directory", self.settings.data_dir),
            ("drafts_directory", self.settings.drafts_dir),
        ):
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
                descriptor, probe = tempfile.mkstemp(dir=path, prefix=".doctor-")
                os.close(descriptor)
                os.unlink(probe)
                checks.append(DoctorCheck(name, "PASS", "writable"))
            except OSError:
                checks.append(DoctorCheck(name, "FAIL", "not writable"))

        checks.extend(self._runtime_checks())

        if self.settings.mock_mode:
            checks.extend(
                (
                    DoctorCheck("x_read", "SKIP", "mock mode; no network request"),
                    DoctorCheck("x_write", "SKIP", "mock mode; no write request"),
                )
            )
        else:
            capabilities = self.settings.x_capability_report()
            for capability in ("app_read", "home_timeline", "mentions", "create_post"):
                available = capabilities.get(capability, False)
                checks.append(
                    DoctorCheck(
                        f"x_{capability}",
                        "PASS" if available else "WARN",
                        (
                            "credentials/scopes configured"
                            if available
                            else "credentials/scopes missing"
                        ),
                    )
                )
        budget_names = (
            "daily_x_read_limit_usd",
            "daily_x_write_limit_usd",
            "daily_openai_limit_usd",
            "daily_xai_limit_usd",
            "daily_heygen_limit_usd",
        )
        configured = sum(getattr(self.settings, name) is not None for name in budget_names)
        checks.append(
            DoctorCheck("budgets", "PASS" if configured else "WARN", f"{configured}/5 configured")
        )
        return checks
