from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.domain.enums import ContentType, GenerationMode
from app.providers.mock_provider import MockLLMProvider
from app.schemas.content import (
    GenerationRequest,
    GenerationResult,
    GenerationVariant,
    MediaPlan,
    UntrustedSourceData,
)
from app.services.draft_service import DraftService
from app.services.editorial_quality import assess_variant, inspect_reply_naturalness
from app.services.generation_pipeline import build_generation_context
from app.services.semantic_classifier import DeterministicSemanticInspector


def _context(source: str):
    return build_generation_context(
        idea_summary="Reply to source",
        idea_explanation={"origin": "reply_to_source", "editorial_intent": "reply_reaction"},
        language="en",
        sources=(
            UntrustedSourceData(
                source_id="source",
                source_type="x_post",
                content=source,
            ),
        ),
        content_type=ContentType.REPLY,
    )


def _variant(candidate, text: str) -> GenerationVariant:
    return GenerationVariant(
        label=candidate.angle_id,
        text=text,
        similarity_risk=0.0,
        angle_id=candidate.angle_id,
        angle_type=candidate.angle_type,
        evidence_ids=candidate.evidence_ids,
        confidence=candidate.confidence,
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
        contribution_id=candidate.angle_id,
        contribution_type=candidate.contribution_type,
    )


class _SensitiveReplyProvider:
    name = "sensitive-reply-provider"
    model = "offline"

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    async def generate_structured(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        text_by_family = {
            "plain_observation": "ten deaths after an outage needs a serious investigation",
            "direct_response": "the response should focus on what failed and who was affected",
            "specific_qualification": (
                "the response should focus on what failed and who was affected"
            ),
            "genuine_question": "what failed badly enough for an outage to cause ten deaths?",
            "dry_humor": "the outage is doing heroic amounts of work",
            "contextual_extension": "the outage now needs a serious failure review",
        }
        variants = tuple(
            _variant(candidate, text_by_family[candidate.contribution_type or "plain_observation"])
            for candidate in request.angle_candidates
        )
        return GenerationResult(
            idea_summary=request.idea_summary,
            recommended_format=request.content_type,
            variants=variants,
            recommended_variant=0,
            recommendation_reason="offline sensitive reply test",
            media_plan=MediaPlan(),
        )


async def _request_for(source: str, *, recent_posts: tuple[str, ...] = ()) -> GenerationRequest:
    context = _context(source)
    planner = DraftService.__new__(DraftService)
    planner.deterministic_semantic_inspector = DeterministicSemanticInspector()
    planner.semantic_inspector = planner.deterministic_semantic_inspector
    planner.recent_corpus_limit = 0
    filtered, _, _ = await planner._validate_context_semantics(context)
    return GenerationRequest(
        idea_summary="reply",
        content_type=ContentType.REPLY,
        language="en",
        generation_mode=GenerationMode.REPLY,
        editorial_intent=filtered.editorial_intent,
        evidence_packet=filtered.evidence,
        source_coverage=filtered.source_coverage,
        available_evidence=filtered.available_evidence,
        content_mode=filtered.content_mode,
        signal_decision=filtered.signal_decision,
        angle_candidates=filtered.angles,
        literal_plans=filtered.literal_plans,
        recent_posts=recent_posts,
    )


def _raw_sensitive_request(source: str) -> GenerationRequest:
    context = _context(source)
    selected_ids = {
        "reply-plain-observation",
        "reply-direct-response",
        "reply-dry-humor",
    }
    selected_pairs = tuple(
        (angle, plan)
        for angle, plan in zip(context.angles, context.literal_plans, strict=True)
        if angle.angle_id in selected_ids
    )
    angles = tuple(angle for angle, _ in selected_pairs)
    plans = tuple(plan for _, plan in selected_pairs)
    return GenerationRequest(
        idea_summary="reply",
        content_type=ContentType.REPLY,
        language="en",
        generation_mode=GenerationMode.REPLY,
        editorial_intent=context.editorial_intent,
        evidence_packet=context.evidence,
        source_coverage=context.source_coverage,
        available_evidence=context.available_evidence,
        content_mode=context.content_mode,
        signal_decision=context.signal_decision,
        angle_candidates=angles,
        literal_plans=plans,
    )


def _generation_service(provider) -> DraftService:
    service = DraftService.__new__(DraftService)
    service.provider = provider
    service.humanizer = SimpleNamespace(process_result=lambda result: result)
    service.editorial_quality_retry_count = 0
    service.quality_enabled = False
    service.minimum_specificity_score = 0.0
    service.minimum_evidence_score = 0.0
    service.minimum_naturalness_score = 0.0
    service.maximum_recent_similarity = 1.0
    service.require_distinct_angles = False
    service.similarity_threshold = 0.99
    service.deterministic_semantic_inspector = DeterministicSemanticInspector()
    service.semantic_inspector = service.deterministic_semantic_inspector
    service._semantic_inspection_cache = {}
    return service


@pytest.mark.asyncio
async def test_sensitive_source_keeps_non_humor_candidates_and_rejects_humor() -> None:
    request = await _request_for("A hospital reported 10 deaths after the outage.")
    assert {item.contribution_type for item in request.angle_candidates} >= {
        "plain_observation",
        "direct_response",
    }
    assert "dry_humor" not in {item.contribution_type for item in request.angle_candidates}

    request = _raw_sensitive_request("A hospital reported 10 deaths after the outage.")
    provider = _SensitiveReplyProvider()
    result = await provider.generate_structured(request)
    reports = tuple(
        assess_variant(
            variant,
            evidence_packet=request.evidence_packet,
            editorial_intent="reply_reaction",
            minimum_specificity=0.0,
            minimum_evidence=0.0,
            minimum_naturalness=0.0,
        )
        for variant in result.variants
    )
    status_by_family = {
        variant.contribution_type: report.passed
        for variant, report in zip(result.variants, reports, strict=True)
    }

    assert status_by_family["plain_observation"]
    assert status_by_family["direct_response"]
    assert not status_by_family["dry_humor"]


@pytest.mark.asyncio
async def test_tragedy_joke_is_blocked_before_recommendation_and_plain_wins() -> None:
    request = _raw_sensitive_request("A hospital reported 10 deaths after the outage.")
    service = _generation_service(_SensitiveReplyProvider())

    _, selected, _, reports = await service._generate_quality_checked(request, None)

    assert selected.contribution_type == "plain_observation"
    humor_index = next(
        index
        for index, candidate in enumerate(request.angle_candidates)
        if candidate.contribution_type == "dry_humor"
    )
    assert not reports[humor_index].passed
    assert "HUMOR_ON_TRAGEDY" in {issue.code for issue in reports[humor_index].issues}


@pytest.mark.asyncio
async def test_regeneration_quality_path_uses_same_humor_safety_gate() -> None:
    request = _raw_sensitive_request("Five people died in the crash.")
    provider = _SensitiveReplyProvider()
    service = _generation_service(provider)

    _, selected, _, reports = await service._generate_quality_checked(
        request.model_copy(update={"feedback": "Regenerate without losing safety."}),
        None,
    )

    assert provider.requests[-1].feedback == "Regenerate without losing safety."
    assert selected.contribution_type != "dry_humor"
    assert any(issue.code == "HUMOR_ON_TRAGEDY" for report in reports for issue in report.issues)


def test_recent_reductive_history_lowers_synonymous_shell() -> None:
    packet = _context("Onboarding added more paperwork.").evidence
    recent = ("fundraising is just sales with extra steps",)
    repeated = GenerationVariant(
        label="repeated",
        text="onboarding boils down to paperwork in a nicer font",
        similarity_risk=0.0,
        angle_id="repeated",
        angle_type="plain_update",
        evidence_ids=("source",),
        confidence="high",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
        contribution_id="repeated",
        contribution_type="plain_observation",
    )
    fresh = repeated.model_copy(
        update={
            "label": "fresh",
            "angle_id": "fresh",
            "contribution_id": "fresh",
            "text": "the extra paperwork makes onboarding harder to finish",
        }
    )
    repeated_shape = inspect_reply_naturalness(
        repeated, evidence_packet=packet, recent_posts=recent
    )
    fresh_shape = inspect_reply_naturalness(fresh, evidence_packet=packet, recent_posts=recent)

    assert repeated_shape.recent_syntax_reuse
    assert not fresh_shape.recent_syntax_reuse


def test_recent_inverse_praise_history_catches_paraphrase() -> None:
    candidate = GenerationVariant(
        label="inverse",
        text="few things communicate simplicity better than six more screens",
        similarity_risk=0.0,
        angle_id="inverse",
        angle_type="plain_update",
        evidence_ids=("source",),
        confidence="high",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
        contribution_id="inverse",
        contribution_type="plain_observation",
    )

    shape = inspect_reply_naturalness(
        candidate,
        recent_posts=("nothing says simplicity like six screens",),
    )

    assert shape.recent_syntax_reuse
    assert shape.quote_card_risk


@pytest.mark.asyncio
async def test_mock_mode_uses_same_humor_safety_policy() -> None:
    request = await _request_for("A hospital reported 10 deaths after the outage.")
    result = await MockLLMProvider().generate_structured(request)

    for variant in result.variants:
        report = assess_variant(
            variant,
            evidence_packet=request.evidence_packet,
            editorial_intent="reply_reaction",
            minimum_specificity=0.0,
            minimum_evidence=0.0,
            minimum_naturalness=0.0,
        )
        if variant.contribution_type == "dry_humor" and not report.passed:
            assert {issue.code for issue in report.issues} & {
                "HUMOR_ON_TRAGEDY",
                "REACTION_LABEL_WITHOUT_FUNCTION",
                "PUNCHLINE_COMPULSION",
            }
