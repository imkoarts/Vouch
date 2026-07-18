"""Shared content-mode routing before contribution planning."""

from __future__ import annotations

from app.domain.enums import ContentType, GenerationMode
from app.schemas.content import ContentModeDecision, EditorialIntent


def route_content_mode(
    *,
    editorial_intent: EditorialIntent,
    generation_mode: GenerationMode,
    requested_format: ContentType | None,
    preferred_length_min: int | None = None,
    preferred_length_max: int | None = None,
    hard_length_max: int | None = None,
) -> ContentModeDecision:
    """Return one shared mode policy; length preferences never authorize padding."""

    format_value = (requested_format or ContentType.SHORT_POST).value
    hard_max = hard_length_max
    if hard_max is None:
        hard_max = 280 if requested_format is not ContentType.LONG_POST else 25_000

    if editorial_intent == "report_event":
        return ContentModeDecision(
            mode="factual_update",
            subtype="direct_update",
            source_role="evidence",
            factual_inspection_required=True,
            attribution_required=generation_mode is GenerationMode.NEWS_CLAIM,
            requested_format=format_value,
            preferred_length_min=preferred_length_min,
            preferred_length_max=preferred_length_max,
            hard_length_max=hard_max,
            failure_conditions=(
                "unsupported factual relation",
                "invalid evidence trace",
                "hard length exceeded",
            ),
        )
    if editorial_intent == "rewrite_existing":
        return ContentModeDecision(
            mode="summary",
            subtype="meaning_preserving_rewrite",
            source_role="factual_premise",
            factual_inspection_required=True,
            attribution_required=False,
            requested_format=format_value,
            preferred_length_min=preferred_length_min,
            preferred_length_max=preferred_length_max,
            hard_length_max=hard_max,
            failure_conditions=("meaning changed", "unsupported claim", "hard length exceeded"),
        )
    if editorial_intent in {"reply_reaction", "quote_reaction"}:
        return ContentModeDecision(
            mode="commentary",
            subtype=(
                "reply_commentary" if editorial_intent == "reply_reaction" else "quote_commentary"
            ),
            source_role="quotation",
            factual_inspection_required=False,
            attribution_required=False,
            requested_format=format_value,
            preferred_length_min=None,
            preferred_length_max=min(preferred_length_max or 280, 280),
            hard_length_max=min(hard_max, 280),
            failure_conditions=(
                "source echo without a reaction",
                "invented factual relation",
                "no source-specific contribution",
            ),
        )
    if editorial_intent == "explain_topic" and generation_mode in {
        GenerationMode.TOPIC_ONLY,
        GenerationMode.USER_IDEA,
    }:
        return ContentModeDecision(
            mode="opinion_or_creative",
            subtype="topic_composition",
            source_role="inspiration",
            factual_inspection_required=False,
            attribution_required=False,
            requested_format=format_value,
            preferred_length_min=preferred_length_min,
            preferred_length_max=preferred_length_max,
            hard_length_max=hard_max,
            failure_conditions=("unsupported factual claim", "hard length exceeded"),
        )
    return ContentModeDecision(
        mode="commentary",
        subtype="analytical_commentary" if editorial_intent == "long_form_analysis" else None,
        source_role="subject_of_commentary",
        factual_inspection_required=False,
        attribution_required=False,
        requested_format=format_value,
        preferred_length_min=preferred_length_min,
        preferred_length_max=preferred_length_max,
        hard_length_max=hard_max,
        failure_conditions=(
            "no grounded contribution",
            "source echo",
            "forced novelty",
            "analytical packaging",
            "hard length exceeded",
        ),
    )
