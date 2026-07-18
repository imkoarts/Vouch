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
from app.services.editorial_quality import assess_variant
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


def _request(
    source: str,
    *,
    candidate_ids: tuple[str, ...] = (
        "reply-plain-observation",
        "reply-direct-response",
        "reply-dry-humor",
    ),
    recent_posts: tuple[str, ...] = (),
) -> GenerationRequest:
    context = _context(source)
    selected = tuple(
        (angle, plan)
        for angle, plan in zip(context.angles, context.literal_plans, strict=True)
        if angle.angle_id in set(candidate_ids)
    )
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
        angle_candidates=tuple(angle for angle, _ in selected),
        literal_plans=tuple(plan for _, plan in selected),
        recent_posts=recent_posts,
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


class _TextByFamilyProvider:
    name = "reply-label-safety-provider"
    model = "offline"

    def __init__(self, text_by_family: dict[str, str]) -> None:
        self.text_by_family = text_by_family
        self.requests: list[GenerationRequest] = []

    async def generate_structured(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        variants = tuple(
            _variant(
                candidate,
                self.text_by_family[candidate.contribution_type or "plain_observation"],
            )
            for candidate in request.angle_candidates
        )
        return GenerationResult(
            idea_summary=request.idea_summary,
            recommended_format=request.content_type,
            variants=variants,
            recommended_variant=0,
            recommendation_reason="offline test",
            media_plan=MediaPlan(),
        )


def _service(provider) -> DraftService:
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
async def test_provider_tragedy_joke_in_plain_family_is_blocked_and_direct_wins() -> None:
    request = _request(
        "A hospital reported 10 deaths after the outage.",
        candidate_ids=("reply-plain-observation", "reply-direct-response"),
    )
    provider = _TextByFamilyProvider(
        {
            "plain_observation": "the outage is doing heroic amounts of work",
            "direct_response": "the response should focus on what failed and who was affected",
        }
    )

    _, selected, _, reports = await _service(provider)._generate_quality_checked(request, None)

    assert selected.contribution_type == "direct_response"
    plain_index = next(
        index
        for index, angle in enumerate(request.angle_candidates)
        if angle.contribution_type == "plain_observation"
    )
    assert not reports[plain_index].passed
    assert "HUMOR_ON_TRAGEDY" in {issue.code for issue in reports[plain_index].issues}


@pytest.mark.asyncio
async def test_sensitive_source_keeps_respectful_non_humor_candidates() -> None:
    context = _context("Ten people lost their lives after the outage.")
    planner = DraftService.__new__(DraftService)
    planner.deterministic_semantic_inspector = DeterministicSemanticInspector()
    planner.semantic_inspector = planner.deterministic_semantic_inspector
    planner.recent_corpus_limit = 0

    filtered, _, _ = await planner._validate_context_semantics(context)
    families = {candidate.contribution_type for candidate in filtered.angles}

    assert "plain_observation" in families
    assert "direct_response" in families or "genuine_question" in families
    # The exploratory humor family may remain before prose exists; final text safety owns rejection.


def test_software_crash_humor_is_not_blocked_as_tragedy() -> None:
    request = _request("The app crash was fixed in version 2.1.")
    variant = _variant(
        next(item for item in request.angle_candidates if item.contribution_type == "dry_humor"),
        "finally an update that fixes the actual crash",
    )

    report = assess_variant(
        variant,
        evidence_packet=request.evidence_packet,
        editorial_intent="reply_reaction",
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert "HUMOR_ON_TRAGEDY" not in {issue.code for issue in report.issues}


def test_lost_lives_callback_is_blocked_even_in_plain_family() -> None:
    request = _request(
        "Ten people lost their lives after the outage.",
        candidate_ids=("reply-plain-observation",),
    )
    variant = _variant(request.angle_candidates[0], "the outage really committed to the bit")

    report = assess_variant(
        variant,
        evidence_packet=request.evidence_packet,
        editorial_intent="reply_reaction",
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert not report.passed
    assert "HUMOR_ON_TRAGEDY" in {issue.code for issue in report.issues}


@pytest.mark.asyncio
async def test_regeneration_uses_label_independent_humor_safety() -> None:
    request = _request(
        "A hospital reported 10 deaths after the outage.",
        candidate_ids=("reply-plain-observation", "reply-direct-response"),
    ).model_copy(update={"feedback": "Regenerate safely."})
    provider = _TextByFamilyProvider(
        {
            "plain_observation": "the outage is doing heroic amounts of work",
            "direct_response": "the response should focus on the failure and the people affected",
        }
    )

    _, selected, _, reports = await _service(provider)._generate_quality_checked(request, None)

    assert provider.requests[-1].feedback == "Regenerate safely."
    assert selected.contribution_type == "direct_response"
    assert any(issue.code == "HUMOR_ON_TRAGEDY" for report in reports for issue in report.issues)


@pytest.mark.asyncio
async def test_mock_mode_retains_non_humor_policy_for_sensitive_source() -> None:
    request = _request(
        "A hospital reported 10 deaths after the outage.",
        candidate_ids=("reply-plain-observation", "reply-direct-response"),
    )
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
        assert "HUMOR_ON_TRAGEDY" not in {issue.code for issue in report.issues}


@pytest.mark.asyncio
async def test_paraphrased_inverse_praise_loses_to_fresh_plain_reply() -> None:
    request = _request(
        "The company added eight screens to onboarding.",
        candidate_ids=("reply-plain-observation", "reply-direct-response"),
        recent_posts=("nothing says simplicity like six screens",),
    )
    provider = _TextByFamilyProvider(
        {
            "plain_observation": "nothing screams simplicity like eight screens",
            "direct_response": "eight extra screens make onboarding harder to finish",
        }
    )

    _, selected, _, reports = await _service(provider)._generate_quality_checked(request, None)

    assert selected.contribution_type == "direct_response"
    repeated_index = next(
        index
        for index, angle in enumerate(request.angle_candidates)
        if angle.contribution_type == "plain_observation"
    )
    assert "RECENT_SYNTAX_REUSE" in {issue.code for issue in reports[repeated_index].issues}


@pytest.mark.asyncio
async def test_decorated_identity_loses_to_structurally_fresh_plain_reply() -> None:
    request = _request(
        "Onboarding now includes more paperwork.",
        candidate_ids=("reply-plain-observation", "reply-direct-response"),
        recent_posts=("fundraising is just sales with extra steps",),
    )
    provider = _TextByFamilyProvider(
        {
            "plain_observation": "onboarding is paperwork with better branding",
            "direct_response": "the added paperwork gives people more steps before they can start",
        }
    )

    _, selected, _, reports = await _service(provider)._generate_quality_checked(request, None)

    assert selected.contribution_type == "direct_response"
    repeated_index = next(
        index
        for index, angle in enumerate(request.angle_candidates)
        if angle.contribution_type == "plain_observation"
    )
    assert "RECENT_SYNTAX_REUSE" in {issue.code for issue in reports[repeated_index].issues}
