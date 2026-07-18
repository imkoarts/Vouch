from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.domain.enums import ContentType, GenerationMode
from app.schemas.content import (
    GenerationRequest,
    GenerationResult,
    GenerationVariant,
    MediaPlan,
    UntrustedSourceData,
)
from app.services.claim_extraction import DeterministicClaimExtractor
from app.services.draft_service import DraftService
from app.services.editorial_quality import assess_variant
from app.services.final_text_inspection import FinalTextInspectionService
from app.services.generation_pipeline import build_generation_context
from app.services.semantic_classifier import DeterministicSemanticInspector


def _context(source: str):
    return build_generation_context(
        idea_summary="Reply to source",
        idea_explanation={"origin": "reply_to_source", "editorial_intent": "reply_reaction"},
        language="en",
        sources=(UntrustedSourceData(source_id="source", source_type="x_post", content=source),),
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
    selected_ids = set(candidate_ids)
    selected = tuple(
        (angle, plan)
        for angle, plan in zip(context.angles, context.literal_plans, strict=True)
        if angle.angle_id in selected_ids
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
    name = "semantic-relation-scope-v176-provider"
    model = "offline"

    def __init__(self, text_by_family: dict[str, str]) -> None:
        self.text_by_family = text_by_family

    async def generate_structured(self, request: GenerationRequest) -> GenerationResult:
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
            recommendation_reason="offline v0.17.6 regression test",
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


def _report(request: GenerationRequest, candidate_id: str, text: str):
    candidate = next(item for item in request.angle_candidates if item.angle_id == candidate_id)
    return assess_variant(
        _variant(candidate, text),
        evidence_packet=request.evidence_packet,
        editorial_intent="reply_reaction",
        recent_posts=request.recent_posts,
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )


@pytest.mark.asyncio
async def test_v176_generated_sarcastic_praise_mislabeled_plain_is_blocked() -> None:
    request = _request(
        "A hospital reported 10 deaths after the outage.",
        candidate_ids=("reply-plain-observation", "reply-direct-response"),
    )
    provider = _TextByFamilyProvider(
        {
            "plain_observation": "a round of applause for the outage",
            "direct_response": "the deaths require a serious review of the outage response",
        }
    )

    _, selected, _, reports = await _service(provider)._generate_quality_checked(request, None)

    assert selected.contribution_type == "direct_response"
    assert any(issue.code == "HUMOR_ON_TRAGEDY" for report in reports for issue in report.issues)


@pytest.mark.asyncio
async def test_v176_regeneration_uses_same_sarcasm_policy() -> None:
    request = _request(
        "A hospital reported 10 deaths after the outage.",
        candidate_ids=("reply-plain-observation", "reply-direct-response"),
    ).model_copy(update={"feedback": "Regenerate without trivializing the deaths."})
    provider = _TextByFamilyProvider(
        {
            "plain_observation": "the outage crushed it",
            "direct_response": "the outage response needs an independent safety review",
        }
    )

    _, selected, _, reports = await _service(provider)._generate_quality_checked(request, None)

    assert selected.contribution_type == "direct_response"
    assert any(issue.code == "HUMOR_ON_TRAGEDY" for report in reports for issue in report.issues)


def test_v176_human_harm_result_state_blocks_humor() -> None:
    request = _request(
        "The incident left ten people on life support.",
        candidate_ids=("reply-dry-humor",),
    )
    report = _report(request, "reply-dry-humor", "achievement unlocked")

    assert not report.passed
    assert "HUMOR_ON_TRAGEDY" in {issue.code for issue in report.issues}


def test_v176_clear_technical_death_source_can_retain_harmless_humor() -> None:
    request = _request("The pipeline died overnight.", candidate_ids=("reply-dry-humor",))
    report = _report(request, "reply-dry-humor", "the pipeline has left the chat")

    assert "HUMOR_ON_TRAGEDY" not in {issue.code for issue in report.issues}


@pytest.mark.asyncio
async def test_v176_unknown_referent_blocks_only_humor_candidate() -> None:
    request = _request(
        "It died after the change.",
        candidate_ids=("reply-direct-response", "reply-dry-humor"),
    )
    provider = _TextByFamilyProvider(
        {
            "direct_response": "the failed component is not identified in the source",
            "dry_humor": "achievement unlocked",
        }
    )

    _, selected, _, reports = await _service(provider)._generate_quality_checked(request, None)

    assert selected.contribution_type == "direct_response"
    humor_index = next(
        index
        for index, candidate in enumerate(request.angle_candidates)
        if candidate.contribution_type == "dry_humor"
    )
    assert "HUMOR_ON_TRAGEDY" in {issue.code for issue in reports[humor_index].issues}


@pytest.mark.asyncio
async def test_v176_manual_and_live_reinspection_share_sarcasm_gate() -> None:
    context = _context("A hospital reported 10 deaths after the outage.")
    contribution = next(
        candidate for candidate in context.angles if candidate.angle_id == "reply-plain-observation"
    )
    variant = _variant(contribution, "five stars for the outage")
    claims = DeterministicClaimExtractor()
    semantics = DeterministicSemanticInspector()
    service = FinalTextInspectionService(
        deterministic_claim_extractor=claims,
        claim_extractor=claims,
        deterministic_semantic_inspector=semantics,
        semantic_inspector=semantics,
    )

    local = service.inspect_local(
        variant.text,
        variant=variant,
        packet=context.evidence,
        content_mode=context.content_mode,
        contribution=contribution,
        coverage=context.source_coverage,
        semantic_reinspection_required=True,
    )
    live = await service.inspect(
        variant.text,
        variant=variant,
        packet=context.evidence,
        content_mode=context.content_mode,
        contribution=contribution,
        coverage=context.source_coverage,
        allow_live=True,
        semantic_reinspection_required=True,
    )

    local_codes = {issue.code for issue in local.editorial_report.issues}
    live_codes = {issue.code for issue in live.editorial_report.issues}
    assert "HUMOR_ON_TRAGEDY" in local_codes
    assert local_codes == live_codes


@pytest.mark.asyncio
async def test_v176_decorated_identity_paraphrase_loses_to_fresh_reply() -> None:
    request = _request(
        "Onboarding added more paperwork.",
        candidate_ids=("reply-plain-observation", "reply-direct-response"),
        recent_posts=("fundraising is just sales with extra steps",),
    )
    provider = _TextByFamilyProvider(
        {
            "plain_observation": "onboarding is glorified paperwork",
            "direct_response": "the added paperwork increases onboarding completion time",
        }
    )

    _, selected, _, reports = await _service(provider)._generate_quality_checked(request, None)

    assert selected.contribution_type == "direct_response"
    repeated_index = next(
        index
        for index, candidate in enumerate(request.angle_candidates)
        if candidate.contribution_type == "plain_observation"
    )
    assert "RECENT_SYNTAX_REUSE" in {issue.code for issue in reports[repeated_index].issues}
