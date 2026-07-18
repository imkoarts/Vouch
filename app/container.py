"""Composition root for mock or human-approved live generation services."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

import httpx
from openai import AsyncOpenAI
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import ConfigurationError, Settings
from app.models import UserAccount, VoiceProfile
from app.providers.base import LLMProvider
from app.providers.mock_provider import MockLLMProvider
from app.providers.responses_provider import ResponsesLLMProvider
from app.schemas.configuration import VoiceProfileConfiguration
from app.services.approval_service import ApprovalService
from app.services.artifact_projection import ArtifactProjectionService
from app.services.artifact_projection_coordinator import ArtifactProjectionCoordinator
from app.services.configuration import ConfigurationService, ContentConfiguration
from app.services.cost_service import BudgetKind, CostService
from app.services.draft_files import DraftArtifactStore
from app.services.draft_service import DraftService
from app.services.humanizer import HumanizerService
from app.services.humanizer_runtime import HumanizerRuntimeLoader
from app.services.image_generation import ImageGenerationService
from app.services.publishing_service import (
    PublicationFaultInjector,
    PublishingService,
    XWritePort,
)
from app.services.tenant_context import active_voice_profile
from app.x_api.live import XApiClient
from app.x_api.mock import MockXClient


@dataclass(slots=True)
class Services:
    drafts: DraftService
    approvals: ApprovalService
    publishing: PublishingService
    writer: XWritePort
    costs: CostService
    configuration: ContentConfiguration
    projections: ArtifactProjectionService
    projection_coordinator: ArtifactProjectionCoordinator
    images: ImageGenerationService


# Compatibility name retained for existing imports and tests.
MockServices = Services


def _decimal(value: float | None) -> Decimal | None:
    return Decimal(str(value)) if value is not None else None


def _secret(value: SecretStr | None) -> str:
    return value.get_secret_value() if value is not None else ""


def _build_provider(
    settings: Settings,
    configuration: ContentConfiguration,
    runtime_loader: HumanizerRuntimeLoader | None = None,
) -> LLMProvider:
    runtime = configuration.runtime
    runtime_loader = runtime_loader or HumanizerRuntimeLoader(
        configured_path=runtime.generation.humanizer_skill_path,
        include_references=runtime.generation.humanizer_include_references,
    )
    selected = "mock" if settings.mock_mode else runtime.generation.provider
    if selected == "mock":
        return MockLLMProvider()
    if selected == "openai":
        if not runtime.providers.openai.enabled:
            raise ConfigurationError("OpenAI is selected but disabled in config/runtime.yml")
        pacing = runtime.request_pacing
        return ResponsesLLMProvider(
            name="openai",
            model=runtime.providers.openai.model,
            api_key=_secret(settings.openai_api_key),
            timeout_seconds=runtime.providers.openai.timeout_seconds,
            proxy_url=settings.outbound_proxy_url,
            pre_request_delay_seconds=pacing.llm_pre_request_delay_seconds,
            minimum_interval_seconds=pacing.llm_minimum_interval_seconds,
            rate_limit_max_retries=pacing.llm_rate_limit_max_retries,
            rate_limit_initial_backoff_seconds=(pacing.llm_rate_limit_initial_backoff_seconds),
            rate_limit_max_backoff_seconds=pacing.llm_rate_limit_max_backoff_seconds,
            insufficient_quota_cooldown_seconds=(pacing.insufficient_quota_cooldown_minutes * 60),
            structured_output_max_retries=pacing.llm_structured_output_max_retries,
            structured_output_retry_delay_seconds=(
                pacing.llm_structured_output_retry_delay_seconds
            ),
            humanizer_runtime_loader=runtime_loader,
        )
    if selected == "xai":
        if not runtime.providers.xai.enabled:
            raise ConfigurationError("xAI is selected but disabled in config/runtime.yml")
        pacing = runtime.request_pacing
        return ResponsesLLMProvider(
            name="xai",
            model=runtime.providers.xai.model,
            api_key=_secret(settings.xai_api_key),
            base_url=runtime.providers.xai.base_url,
            timeout_seconds=runtime.providers.xai.timeout_seconds,
            proxy_url=settings.outbound_proxy_url,
            pre_request_delay_seconds=pacing.llm_pre_request_delay_seconds,
            minimum_interval_seconds=pacing.llm_minimum_interval_seconds,
            rate_limit_max_retries=pacing.llm_rate_limit_max_retries,
            rate_limit_initial_backoff_seconds=(pacing.llm_rate_limit_initial_backoff_seconds),
            rate_limit_max_backoff_seconds=pacing.llm_rate_limit_max_backoff_seconds,
            insufficient_quota_cooldown_seconds=(pacing.insufficient_quota_cooldown_minutes * 60),
            structured_output_max_retries=pacing.llm_structured_output_max_retries,
            structured_output_retry_delay_seconds=(
                pacing.llm_structured_output_retry_delay_seconds
            ),
            humanizer_runtime_loader=runtime_loader,
        )
    raise ConfigurationError(f"Unsupported generation provider: {selected}")


def _build_writer(settings: Settings, configuration: ContentConfiguration) -> XWritePort:
    runtime = configuration.runtime.publication
    live_enabled = (
        not settings.mock_mode and settings.publish_enabled and runtime.manual_x_publish_enabled
    )
    if not live_enabled:
        return MockXClient()
    missing = settings.missing_x_write_credentials()
    if missing:
        raise ConfigurationError(
            "Manual X publication is enabled but credentials are incomplete: " + ", ".join(missing)
        )
    return XApiClient(
        base_url=settings.x_api_base_url,
        auth_mode=settings.x_auth_mode,
        access_token=_secret(settings.x_access_token),
        consumer_key=_secret(settings.x_consumer_key),
        consumer_secret=_secret(settings.x_consumer_secret),
        access_token_secret=_secret(settings.x_access_token_secret),
        oauth2_scopes=settings.x_oauth2_scope_set if settings.x_auth_mode == "oauth2" else None,
        timeout_seconds=settings.request_timeout_seconds,
        proxy_url=settings.outbound_proxy_url,
    )


def build_services(
    session: Session,
    settings: Settings,
    *,
    writer: XWritePort | None = None,
    publication_fault_injector: PublicationFaultInjector | None = None,
) -> Services:
    """Build services without granting any LLM or Telegram component X-write access."""

    settings.ensure_directories()
    configuration = ConfigurationService(settings.config_dir).load()
    artifacts = DraftArtifactStore(settings.drafts_dir)
    projection_coordinator = ArtifactProjectionCoordinator.install(session, artifacts)
    projections = ArtifactProjectionService(session, artifacts)
    runtime_loader = HumanizerRuntimeLoader(
        configured_path=configuration.runtime.generation.humanizer_skill_path,
        include_references=configuration.runtime.generation.humanizer_include_references,
    )
    provider = _build_provider(settings, configuration, runtime_loader)
    humanizer = HumanizerService(
        enabled=configuration.runtime.generation.humanizer_enabled,
        mode=configuration.runtime.generation.humanizer_mode,
        external_skill_path=configuration.runtime.generation.humanizer_skill_path,
        include_references=(configuration.runtime.generation.humanizer_include_references),
        runtime_loader=runtime_loader,
    )
    configured_voice = active_voice_profile(configuration.runtime.generation.voice_profile)
    if configured_voice is configuration.runtime.generation.voice_profile:
        local_profile = session.scalar(
            select(VoiceProfile)
            .join(UserAccount, UserAccount.id == VoiceProfile.user_id)
            .where(UserAccount.auth_provider == "local")
            .order_by(VoiceProfile.updated_at.desc())
        )
        if local_profile is not None:
            configured_voice = VoiceProfileConfiguration(
                account=local_profile.account_type,
                language=local_profile.language,
                tone=tuple(local_profile.tone),
                response_preferences=tuple(local_profile.response_preferences),
                guidance=local_profile.guidance,
                banned_tendencies=tuple(local_profile.banned_tendencies),
            )
    drafts = DraftService(
        session,
        artifacts,
        provider,
        profile=configuration.profile,
        max_weighted_length=configuration.profile.account.default_post_max_chars,
        premium_long_posts_enabled=(
            configuration.profile.account.x_account_tier == "premium"
            and configuration.profile.account.premium_long_posts_enabled
        ),
        premium_long_post_max_chars=(configuration.profile.account.premium_long_post_max_chars),
        post_length_mode=configuration.runtime.generation.post_length_mode,
        similarity_threshold=settings.similarity_threshold,
        humanizer=humanizer,
        editorial_quality_retry_count=(
            configuration.runtime.generation.quality.max_humanizer_revisions
        ),
        minimum_specificity_score=(
            configuration.runtime.generation.quality.minimum_specificity_score
        ),
        minimum_evidence_score=(configuration.runtime.generation.quality.minimum_evidence_score),
        minimum_naturalness_score=(
            configuration.runtime.generation.quality.minimum_naturalness_score
        ),
        maximum_recent_similarity=(
            configuration.runtime.generation.quality.maximum_recent_similarity
        ),
        max_evidence_items=configuration.runtime.generation.evidence.max_items,
        angle_selection_enabled=configuration.runtime.generation.angle_selection_enabled,
        voice_profile=configured_voice,
        style_examples_enabled=configuration.runtime.generation.style_examples.enabled,
        max_approved_examples=(
            configuration.runtime.generation.style_examples.max_approved_examples
        ),
        max_rejected_examples=(
            configuration.runtime.generation.style_examples.max_rejected_examples
        ),
        recent_corpus_limit=(configuration.runtime.generation.quality.recent_corpus_limit),
        style_examples_path=settings.config_dir / "style_examples.yml",
        outbound_proxy_url=settings.outbound_proxy_url,
        maximum_variants=configuration.runtime.generation.variants.count,
        signal_candidate_attempts=(
            configuration.runtime.generation.signal_selection.max_candidate_attempts
        ),
        automatic_multi_source_synthesis_enabled=(
            configuration.runtime.generation.signal_selection.automatic_multi_source_synthesis_enabled
        ),
        remote_semantic_validation_enabled=(
            configuration.runtime.generation.quality.remote_semantic_validation_enabled
        ),
        quality_enabled=configuration.runtime.generation.quality.enabled,
        require_distinct_angles=(configuration.runtime.generation.variants.require_distinct_angles),
    )
    approvals = ApprovalService(session, drafts, artifacts)
    limits: dict[BudgetKind, Decimal | None] = {
        "x_read": _decimal(settings.daily_x_read_limit_usd),
        "x_write": _decimal(settings.daily_x_write_limit_usd),
        "openai": _decimal(settings.daily_openai_limit_usd),
        "xai": _decimal(settings.daily_xai_limit_usd),
        "heygen": _decimal(settings.daily_heygen_limit_usd),
    }
    estimates: dict[str, Decimal] = {}
    if configuration.costs.x.write_usd is not None:
        estimates["x_write"] = configuration.costs.x.write_usd
    if configuration.costs.x.read_usd is not None:
        estimates["x_read"] = configuration.costs.x.read_usd
    costs = CostService(
        session,
        limits=limits,
        estimates=estimates,
        # Mock mode may run without configured prices. Live manual writes use
        # fail-closed cost semantics whenever an operator sets a daily budget.
        allow_unknown_estimates=settings.mock_mode,
    )
    selected_writer = writer or _build_writer(settings, configuration)
    manual_publish_enabled = settings.publish_enabled and (
        settings.mock_mode or configuration.runtime.publication.manual_x_publish_enabled
    )
    publishing = PublishingService(
        session,
        drafts,
        approvals,
        selected_writer,
        costs,
        publish_enabled=manual_publish_enabled,
        # X_USER_ID selects the account whose home timeline and mentions are read. It is not a
        # publishing-identity pin: manual /new sources may come from any public account, while the
        # authenticated OAuth writer returned by /2/users/me is the authoritative write identity
        # displayed in the final confirmation preview.
        expected_account_id=None,
        fault_injector=publication_fault_injector,
        live_writes_enabled=(
            not settings.mock_mode
            and settings.publish_enabled
            and configuration.runtime.publication.manual_x_publish_enabled
        ),
        enterprise_quote_posts_enabled=(
            configuration.runtime.publication.enterprise_quote_posts_enabled
        ),
    )
    image_client_factory: Callable[[], AsyncOpenAI] | None = None
    images_enabled = (
        not settings.mock_mode
        and configuration.runtime.images.enabled
        and configuration.runtime.images.provider == "openai"
        and configuration.runtime.providers.openai.enabled
        and settings.openai_api_key is not None
    )
    if images_enabled:

        def create_image_client() -> AsyncOpenAI:
            http_client = httpx.AsyncClient(
                proxy=settings.outbound_proxy_url,
                timeout=settings.request_timeout_seconds,
                follow_redirects=False,
                trust_env=False,
            )
            return AsyncOpenAI(
                api_key=_secret(settings.openai_api_key),
                timeout=settings.request_timeout_seconds,
                max_retries=2,
                http_client=http_client,
            )

        image_client_factory = create_image_client
    images = ImageGenerationService(
        drafts=drafts,
        client_factory=image_client_factory,
        enabled=images_enabled,
        model=configuration.runtime.providers.openai.image_model,
        size=configuration.runtime.images.size,
        quality=configuration.runtime.images.quality,
        output_format=configuration.runtime.images.output_format,
    )
    return Services(
        drafts,
        approvals,
        publishing,
        selected_writer,
        costs,
        configuration,
        projections,
        projection_coordinator,
        images,
    )


def build_mock_services(
    session: Session,
    settings: Settings,
    *,
    writer: XWritePort | None = None,
    publication_fault_injector: PublicationFaultInjector | None = None,
) -> Services:
    """Backward-compatible wrapper used by the existing mock tests and CLI."""

    return build_services(
        session,
        settings,
        writer=writer,
        publication_fault_injector=publication_fault_injector,
    )
