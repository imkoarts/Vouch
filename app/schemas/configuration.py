"""Strict contracts for the checked-in content configuration files."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.enums import ContentSourceKind, ContentType, PostLengthMode


class ConfigurationModel(BaseModel):
    """Fail closed when configuration contains misspelled or unknown fields."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class WeightedSourceConfiguration(ConfigurationModel):
    enabled: bool
    weight: float = Field(ge=0.0, le=1.0)


class HomeTimelineConfiguration(WeightedSourceConfiguration):
    max_posts: int = Field(default=5, ge=1, le=10)
    exclude_replies: bool = True
    exclude_retweets: bool = True


class RecentSearchConfiguration(WeightedSourceConfiguration):
    queries: tuple[str, ...]
    max_posts_per_query: int = Field(default=10, ge=10, le=10)


class TrackedAccountConfiguration(ConfigurationModel):
    """One X account that can be paused without removing it from the watch list."""

    username: str = Field(min_length=1, max_length=50)
    enabled: bool = True
    user_id: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        normalized = value.strip().removeprefix("@").casefold()
        if not normalized or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in normalized
        ):
            raise ValueError("username may contain only letters, numbers, and underscores")
        return normalized


class SelectedAccountsConfiguration(WeightedSourceConfiguration):
    usernames: tuple[str, ...] = ()
    user_ids: tuple[str, ...] = ()
    list_ids: tuple[str, ...] = ()
    preferred_categories: tuple[str, ...] = ()
    accounts: tuple[TrackedAccountConfiguration, ...] = ()
    max_posts_per_account: int = Field(default=5, ge=5, le=100)

    @property
    def tracked_accounts(self) -> tuple[TrackedAccountConfiguration, ...]:
        """Return new structured accounts plus legacy username entries without duplicates."""

        items = list(self.accounts)
        known = {item.username.casefold() for item in items}
        for index, username in enumerate(self.usernames):
            normalized = username.strip().removeprefix("@").casefold()
            if not normalized or normalized in known:
                continue
            user_id = self.user_ids[index] if index < len(self.user_ids) else None
            items.append(
                TrackedAccountConfiguration(username=normalized, enabled=True, user_id=user_id)
            )
            known.add(normalized)
        return tuple(items)

    @field_validator("accounts")
    @classmethod
    def accounts_are_unique(
        cls, values: tuple[TrackedAccountConfiguration, ...]
    ) -> tuple[TrackedAccountConfiguration, ...]:
        usernames = [item.username.casefold() for item in values]
        if len(usernames) != len(set(usernames)):
            raise ValueError("selected accounts must be unique by username")
        return values


class ManualSourceConfiguration(WeightedSourceConfiguration):
    creator_inspiration_reference_url: str


class EvergreenSourceConfiguration(WeightedSourceConfiguration):
    topics: tuple[str, ...]


class SourceCatalogConfiguration(ConfigurationModel):
    home_timeline: HomeTimelineConfiguration
    recent_search: RecentSearchConfiguration
    x_activity: WeightedSourceConfiguration
    selected_accounts: SelectedAccountsConfiguration
    manual: ManualSourceConfiguration
    evergreen: EvergreenSourceConfiguration


class AutomaticDiscoveryConfiguration(ConfigurationModel):
    enabled: bool = True
    # Retained only to load older config files. The UI does not expose this legacy switch;
    # enabled discovery always performs one startup read and then follows the interval.
    run_on_start: bool = False
    interval_preset: Literal["1h", "3h", "6h", "12h", "custom"] = "12h"
    custom_interval_minutes: int = Field(default=10, ge=10, le=10_080)
    max_runs_per_utc_day: int = Field(default=5, ge=1, le=144)
    lookback_hours: int = Field(default=24, ge=1, le=168)
    trends_woeid: int = Field(default=1, ge=1)
    max_trends: int = Field(default=3, ge=1, le=10)
    trend_topics_per_run: int = Field(default=1, ge=1, le=3)
    max_total_posts: int = Field(default=15, ge=5, le=30)
    final_candidates: int = Field(default=5, ge=1, le=10)
    generation_candidates_per_run: int = Field(default=5, ge=1, le=10)
    drafts_per_run: int = Field(default=1, ge=1, le=1)
    notify_when_no_candidate: bool = True

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_interval_hours(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "interval_hours" not in value:
            return value
        migrated = dict(value)
        raw_hours = migrated.pop("interval_hours")
        if "interval_preset" not in migrated:
            try:
                hours = int(raw_hours)
            except (TypeError, ValueError):
                hours = 12
            preset = f"{hours}h"
            if preset in {"1h", "3h", "6h", "12h"}:
                migrated["interval_preset"] = preset
            else:
                migrated["interval_preset"] = "custom"
                migrated.setdefault("custom_interval_minutes", max(10, hours * 60))
        return migrated

    @property
    def effective_interval_minutes(self) -> int:
        if self.interval_preset == "custom":
            return self.custom_interval_minutes
        return int(self.interval_preset.removesuffix("h")) * 60

    @model_validator(mode="after")
    def discovery_limits_are_consistent(self) -> AutomaticDiscoveryConfiguration:
        if self.trend_topics_per_run > self.max_trends:
            raise ValueError("trend_topics_per_run must be <= max_trends")
        return self


class RankingConfiguration(ConfigurationModel):
    source_priority: float = Field(ge=0.0)
    recency_score: float = Field(ge=0.0)
    engagement_velocity: float = Field(ge=0.0)
    topic_relevance: float = Field(ge=0.0)
    novelty_score: float = Field(ge=0.0)
    media_potential: float = Field(ge=0.0)
    plagiarism_risk: float = Field(ge=0.0)
    safety_risk: float = Field(ge=0.0)
    engagement_age_floor_hours: float = Field(gt=0.0)


class ContentSourcesConfiguration(ConfigurationModel):
    sources: SourceCatalogConfiguration
    automatic_discovery: AutomaticDiscoveryConfiguration = Field(
        default_factory=AutomaticDiscoveryConfiguration
    )
    ranking: RankingConfiguration

    def source_for_kind(self, kind: ContentSourceKind) -> WeightedSourceConfiguration:
        """Return the authoritative configured source bucket for a domain kind."""

        if kind is ContentSourceKind.HOME_TIMELINE:
            return self.sources.home_timeline
        if kind is ContentSourceKind.RECENT_SEARCH:
            return self.sources.recent_search
        if kind is ContentSourceKind.X_ACTIVITY:
            return self.sources.x_activity
        if kind in {ContentSourceKind.SELECTED_ACCOUNT, ContentSourceKind.X_LIST}:
            return self.sources.selected_accounts
        if kind in {ContentSourceKind.MANUAL_URL, ContentSourceKind.IMPORT_FILE}:
            return self.sources.manual
        return self.sources.evergreen


class AccountConfiguration(ConfigurationModel):
    language: str = Field(min_length=2, max_length=16)
    secondary_language: str | None = Field(default=None, min_length=2, max_length=16)
    x_account_tier: Literal["standard", "premium"] = "standard"
    default_post_max_chars: int = Field(default=280, ge=1, le=280)
    premium_long_posts_enabled: bool = False
    premium_long_post_max_chars: int = Field(default=25_000, ge=281, le=25_000)

    @model_validator(mode="after")
    def premium_settings_are_consistent(self) -> AccountConfiguration:
        if self.premium_long_posts_enabled and self.x_account_tier != "premium":
            raise ValueError("premium_long_posts_enabled requires x_account_tier=premium")
        return self


class BrandConfiguration(ConfigurationModel):
    name: str
    description: str
    target_audience: str
    expertise: tuple[str, ...]
    values: tuple[str, ...]
    prohibited_topics: tuple[str, ...]
    preferred_topics: tuple[str, ...]
    preferred_tone: tuple[str, ...]
    avoid: tuple[str, ...]


class GenerationConfiguration(ConfigurationModel):
    formats: tuple[ContentType, ...] = Field(min_length=1)
    variants_per_idea: int = Field(default=3, ge=1, le=3)
    hashtags_max: int = Field(ge=0)
    emoji_max: int = Field(ge=0)

    @field_validator("formats")
    @classmethod
    def formats_are_unique(cls, formats: tuple[ContentType, ...]) -> tuple[ContentType, ...]:
        if len(formats) != len(set(formats)):
            raise ValueError("generation formats must be unique")
        return formats


class ContentProfileConfiguration(ConfigurationModel):
    account: AccountConfiguration
    brand: BrandConfiguration
    generation: GenerationConfiguration


class XCostEstimates(ConfigurationModel):
    read_usd: Decimal | None = Field(default=None, ge=0)
    trends_request_usd: Decimal | None = Field(default=None, ge=0)
    write_usd: Decimal | None = Field(default=None, ge=0)


class TokenCostEstimates(ConfigurationModel):
    input_per_million_usd: Decimal | None = Field(default=None, ge=0)
    output_per_million_usd: Decimal | None = Field(default=None, ge=0)


class HeyGenCostEstimates(ConfigurationModel):
    per_minute_usd: Decimal | None = Field(default=None, ge=0)


class CostEstimatesConfiguration(ConfigurationModel):
    """User-maintained estimates; ``None`` means explicitly not configured."""

    x: XCostEstimates
    openai: TokenCostEstimates
    xai: TokenCostEstimates
    heygen: HeyGenCostEstimates


class ProviderFeatureConfiguration(ConfigurationModel):
    enabled: bool
    model: str = Field(min_length=1)
    timeout_seconds: float = Field(default=120.0, gt=0.0, le=3600.0)


class OpenAIProviderFeatureConfiguration(ProviderFeatureConfiguration):
    image_model: str = Field(default="gpt-image-2", min_length=1)


class XAIProviderFeatureConfiguration(ProviderFeatureConfiguration):
    base_url: str = Field(default="https://api.x.ai/v1", min_length=1)


class HeyGenProviderFeatureConfiguration(ProviderFeatureConfiguration):
    mode: str = Field(default="api", pattern="^(api|plugin_manual)$")


class ProvidersConfiguration(ConfigurationModel):
    openai: OpenAIProviderFeatureConfiguration
    xai: XAIProviderFeatureConfiguration
    heygen: HeyGenProviderFeatureConfiguration


class GenerationEvidenceConfiguration(ConfigurationModel):
    max_items: int = Field(default=12, ge=1, le=25)
    prefer_primary_sources: bool = True
    deduplicate: bool = True


class GenerationVariantsConfiguration(ConfigurationModel):
    count: int = Field(default=3, ge=1, le=3)
    require_distinct_angles: bool = True


class SignalSelectionConfiguration(ConfigurationModel):
    enabled: bool = True
    max_candidate_attempts: int = Field(default=5, ge=1, le=10)
    automatic_multi_source_synthesis_enabled: bool = False


class GenerationQualityConfiguration(ConfigurationModel):
    enabled: bool = True
    remote_semantic_validation_enabled: bool = False
    max_draft_attempts: int = Field(default=2, ge=1, le=5)
    max_humanizer_revisions: int = Field(default=1, ge=0, le=2)
    reject_insufficient_context: bool = True
    minimum_specificity_score: float = Field(default=0.70, ge=0.0, le=1.0)
    minimum_evidence_score: float = Field(default=0.90, ge=0.0, le=1.0)
    minimum_naturalness_score: float = Field(default=0.70, ge=0.0, le=1.0)
    maximum_recent_similarity: float = Field(default=0.88, ge=0.0, le=1.0)
    recent_corpus_limit: int = Field(default=12, ge=0, le=50)
    topic_substitution_check: bool = True
    final_sentence_deletion_check: bool = True


class VoiceDefaultsConfiguration(ConfigurationModel):
    hashtags: bool = False
    emojis: bool = False
    max_emojis: int = Field(default=1, ge=0, le=5)
    lowercase_allowed: bool = True
    slang_allowed: bool = True
    forced_slang: bool = False
    forced_hot_take: bool = False
    forced_contrarianism: bool = False


class VoiceShortPostConfiguration(ConfigurationModel):
    preferred_sentences_min: int = Field(default=1, ge=1, le=5)
    preferred_sentences_max: int = Field(default=3, ge=1, le=6)
    preferred_characters_min: int = Field(default=60, ge=1, le=280)
    preferred_characters_max: int = Field(default=260, ge=1, le=280)

    @model_validator(mode="after")
    def ranges_are_valid(self) -> VoiceShortPostConfiguration:
        if self.preferred_sentences_min > self.preferred_sentences_max:
            raise ValueError("preferred sentence minimum must not exceed maximum")
        if self.preferred_characters_min > self.preferred_characters_max:
            raise ValueError("preferred character minimum must not exceed maximum")
        return self


class VoiceProfileConfiguration(ConfigurationModel):
    account: str | None = None
    language: str = Field(default="en", min_length=2, max_length=16)
    tone: tuple[str, ...] = (
        "casual",
        "observant",
        "slightly_skeptical",
        "internet_native",
    )
    response_preferences: tuple[str, ...] = ()
    guidance: str = ""
    defaults: VoiceDefaultsConfiguration = Field(default_factory=VoiceDefaultsConfiguration)
    short_post: VoiceShortPostConfiguration = Field(default_factory=VoiceShortPostConfiguration)
    banned_tendencies: tuple[str, ...] = (
        "corporate_voice",
        "creator_coach_voice",
        "generic_motivation",
        "fake_depth",
        "polished_slogan_ending",
        "vague_brand_praise",
    )


class StyleExamplesConfiguration(ConfigurationModel):
    enabled: bool = True
    max_approved_examples: int = Field(default=3, ge=0, le=5)
    max_rejected_examples: int = Field(default=2, ge=0, le=3)


class GenerationRuntimeConfiguration(ConfigurationModel):
    provider: str = Field(pattern="^(mock|openai|xai)$")
    post_length_mode: PostLengthMode = PostLengthMode.SHORT
    input_classification_enabled: bool = True
    angle_selection_enabled: bool = True
    evidence: GenerationEvidenceConfiguration = Field(
        default_factory=GenerationEvidenceConfiguration
    )
    variants: GenerationVariantsConfiguration = Field(
        default_factory=GenerationVariantsConfiguration
    )
    signal_selection: SignalSelectionConfiguration = Field(
        default_factory=SignalSelectionConfiguration
    )
    quality: GenerationQualityConfiguration = Field(default_factory=GenerationQualityConfiguration)
    voice_profile: VoiceProfileConfiguration = Field(default_factory=VoiceProfileConfiguration)
    style_examples: StyleExamplesConfiguration = Field(default_factory=StyleExamplesConfiguration)
    humanizer_enabled: bool = True
    humanizer_mode: str = Field(
        default="compact_conditional",
        pattern="^(compact_conditional|prompt_and_cleanup|prompt_only|disabled)$",
    )
    humanizer_skill_path: str | None = None
    humanizer_include_references: bool = True
    editorial_quality_retry_count: int | None = Field(default=None, ge=0, le=2)
    notify_telegram: bool = True

    @model_validator(mode="after")
    def generation_safety_flags_are_not_disableable(
        self,
    ) -> GenerationRuntimeConfiguration:
        if not self.signal_selection.enabled:
            raise ValueError(
                "generation.signal_selection.enabled=false is deprecated; grounded signal "
                "selection is now a mandatory safety stage"
            )
        if not self.angle_selection_enabled:
            raise ValueError(
                "generation.angle_selection_enabled=false is deprecated; grounded contribution "
                "planning is now a mandatory safety stage"
            )
        if (
            self.editorial_quality_retry_count is not None
            and self.editorial_quality_retry_count != self.quality.max_humanizer_revisions
        ):
            raise ValueError(
                "generation.editorial_quality_retry_count is deprecated; keep it equal to "
                "generation.quality.max_humanizer_revisions or remove it"
            )
        return self

    @field_validator("humanizer_skill_path", mode="before")
    @classmethod
    def empty_humanizer_path_is_none(cls, value: object) -> object:
        return None if value == "" else value


class RequestPacingConfiguration(ConfigurationModel):
    """Conservative delays and bounded retries for paid external APIs."""

    x_request_delay_seconds: float = Field(default=2.0, ge=0.0, le=60.0)
    x_temporary_error_max_retries: int = Field(default=2, ge=0, le=5)
    x_temporary_error_initial_backoff_seconds: float = Field(default=2.0, ge=0.0, le=120.0)
    x_temporary_error_max_backoff_seconds: float = Field(default=10.0, ge=0.0, le=600.0)
    llm_pre_request_delay_seconds: float = Field(default=3.0, ge=0.0, le=60.0)
    llm_minimum_interval_seconds: float = Field(default=30.0, ge=0.0, le=3600.0)
    llm_rate_limit_max_retries: int = Field(default=2, ge=0, le=5)
    llm_rate_limit_initial_backoff_seconds: float = Field(default=5.0, ge=0.1, le=120.0)
    llm_rate_limit_max_backoff_seconds: float = Field(default=30.0, ge=0.1, le=600.0)
    insufficient_quota_cooldown_minutes: int = Field(default=60, ge=1, le=1440)
    llm_structured_output_max_retries: int = Field(default=0, ge=0, le=2)
    llm_structured_output_retry_delay_seconds: float = Field(default=2.0, ge=0.0, le=60.0)

    @model_validator(mode="after")
    def backoff_window_is_valid(self) -> RequestPacingConfiguration:
        if (
            self.x_temporary_error_max_backoff_seconds
            < self.x_temporary_error_initial_backoff_seconds
        ):
            raise ValueError(
                "x_temporary_error_max_backoff_seconds must be >= "
                "x_temporary_error_initial_backoff_seconds"
            )
        if self.llm_rate_limit_max_backoff_seconds < self.llm_rate_limit_initial_backoff_seconds:
            raise ValueError(
                "llm_rate_limit_max_backoff_seconds must be >= "
                "llm_rate_limit_initial_backoff_seconds"
            )
        return self


class ImageGenerationConfiguration(ConfigurationModel):
    enabled: bool = False
    provider: str = Field(default="openai", pattern="^(openai|disabled)$")
    size: str = Field(default="1536x1024", min_length=3)
    quality: str = Field(default="medium", pattern="^(low|medium|high|auto)$")
    output_format: str = Field(default="png", pattern="^(png|jpeg|webp)$")


class TelegramRuntimeConfiguration(ConfigurationModel):
    enabled: bool = False
    autostart: bool = True
    operator_user_id: int | None = Field(default=None, gt=0)
    allowed_user_ids: tuple[int, ...] = ()
    review_chat_id: int | None = Field(default=None, gt=0)
    notify_on_new_draft: bool = True
    long_poll_timeout_seconds: int = Field(default=30, ge=1, le=50)
    request_timeout_seconds: int = Field(default=45, ge=5, le=120)
    drop_pending_updates_on_start: bool = False

    @field_validator("allowed_user_ids")
    @classmethod
    def allowed_user_ids_are_unique(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if len(values) != len(set(values)):
            raise ValueError("telegram allowed_user_ids must be unique")
        if any(value <= 0 for value in values):
            raise ValueError("telegram user IDs must be positive")
        return values

    @property
    def effective_allowed_user_ids(self) -> tuple[int, ...]:
        """Resolve the simple operator ID plus optional legacy/advanced allowlist."""

        values = list(self.allowed_user_ids)
        if self.operator_user_id is not None:
            values.insert(0, self.operator_user_id)
        return tuple(dict.fromkeys(values))

    @property
    def effective_review_chat_id(self) -> int | None:
        """Use the operator ID as the default private review chat."""

        return self.review_chat_id or self.operator_user_id


XActivityEventType = Literal["post.create", "post.delete", "post.mention.create"]


class XActivitySubscriptionConfiguration(ConfigurationModel):
    event_type: XActivityEventType
    user_id: str = Field(min_length=1, max_length=64)
    tag: str | None = Field(default=None, max_length=200)
    ingest_as_idea: bool = True
    generate_reply_draft: bool = False
    notify_telegram: bool = True

    @field_validator("user_id")
    @classmethod
    def user_id_is_numeric_or_self(cls, value: str) -> str:
        normalized = value.strip()
        if normalized != "self" and not normalized.isdigit():
            raise ValueError("x_activity subscription user_id must be numeric or 'self'")
        return normalized

    @model_validator(mode="after")
    def event_actions_are_consistent(self) -> XActivitySubscriptionConfiguration:
        if self.generate_reply_draft and self.event_type != "post.mention.create":
            raise ValueError("generate_reply_draft is only valid for post.mention.create")
        if self.event_type == "post.delete" and (self.ingest_as_idea or self.generate_reply_draft):
            raise ValueError("post.delete cannot create ideas or reply drafts")
        return self


class XActivityRuntimeConfiguration(ConfigurationModel):
    enabled: bool = False
    autostart: bool = False
    notify_telegram: bool = True
    backfill_minutes: int = Field(default=0, ge=0, le=5)
    reconnect_initial_seconds: float = Field(default=2.0, ge=0.5, le=60.0)
    reconnect_max_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    subscriptions: tuple[XActivitySubscriptionConfiguration, ...] = ()

    @field_validator("subscriptions")
    @classmethod
    def subscriptions_are_unique(
        cls, values: tuple[XActivitySubscriptionConfiguration, ...]
    ) -> tuple[XActivitySubscriptionConfiguration, ...]:
        keys = [(item.event_type, item.user_id) for item in values]
        if len(keys) != len(set(keys)):
            raise ValueError("x_activity subscriptions must be unique by event_type and user_id")
        return values

    @model_validator(mode="after")
    def reconnect_window_is_valid(self) -> XActivityRuntimeConfiguration:
        if self.reconnect_max_seconds < self.reconnect_initial_seconds:
            raise ValueError(
                "x_activity reconnect_max_seconds must be >= reconnect_initial_seconds"
            )
        return self


class PublicationRuntimeConfiguration(ConfigurationModel):
    manual_x_publish_enabled: bool = False
    enterprise_quote_posts_enabled: bool = False
    automatic_x_publish_enabled: bool = False

    @field_validator("automatic_x_publish_enabled")
    @classmethod
    def automatic_publication_is_forbidden(cls, value: bool) -> bool:
        if value:
            raise ValueError("automatic X publication is intentionally unsupported")
        return value


class RuntimeConfiguration(ConfigurationModel):
    providers: ProvidersConfiguration
    generation: GenerationRuntimeConfiguration
    request_pacing: RequestPacingConfiguration = Field(default_factory=RequestPacingConfiguration)
    images: ImageGenerationConfiguration
    telegram: TelegramRuntimeConfiguration
    x_activity: XActivityRuntimeConfiguration = Field(default_factory=XActivityRuntimeConfiguration)
    publication: PublicationRuntimeConfiguration
