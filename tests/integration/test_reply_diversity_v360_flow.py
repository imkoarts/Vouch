from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.enums import ContentType, GenerationMode
from app.schemas.content import (
    EvidenceItem,
    EvidencePacket,
    GenerationRequest,
    GenerationResult,
    GenerationVariant,
    MediaPlan,
    QualityReport,
    UntrustedSourceData,
)
from app.services.draft_service import DraftService
from app.services.editorial_quality import assess_variant
from app.services.generation_pipeline import build_generation_context
from app.services.humanizer_runtime import HumanizerRuntimeLoader
from app.services.semantic_classifier import DeterministicSemanticInspector


class ReplyRecordingProvider:
    name = "reply-recording"
    model = "reply-recording-v1"

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    async def generate_structured(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        variants: list[GenerationVariant] = []
        for candidate in request.angle_candidates:
            if candidate.contribution_type == "plain_observation":
                text = (
                    "half the time the social manager is good, they just gave them "
                    "10 other jobs and no budget"
                )
            elif candidate.contribution_type == "direct_response":
                text = "this is just understaffing with extra steps"
            elif candidate.contribution_type == "dry_humor":
                text = "nothing says social strategy like ten jobs and no budget"
            else:
                text = "ten other jobs and no budget explain the constraint"
            variants.append(
                GenerationVariant(
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
            )
        return GenerationResult(
            idea_summary=request.idea_summary,
            recommended_format=request.content_type,
            variants=tuple(variants),
            recommended_variant=0,
            recommendation_reason="offline reply diversity integration",
            media_plan=MediaPlan(),
        )


def _context():
    return build_generation_context(
        idea_summary="Reply to overloaded social manager",
        idea_explanation={"origin": "reply_to_source", "editorial_intent": "reply_reaction"},
        language="en",
        sources=(
            UntrustedSourceData(
                source_id="source",
                source_type="x_post",
                content="The social manager handles 10 other jobs and has no budget.",
            ),
        ),
        content_type=ContentType.REPLY,
    )


def _passing_report() -> QualityReport:
    return QualityReport(
        passed=True,
        specificity_score=1.0,
        evidence_score=1.0,
        naturalness_score=1.0,
        distinctiveness_score=1.0,
    )


@pytest.mark.asyncio
async def test_reply_generation_request_receives_recent_posts_and_non_humor_candidates() -> None:
    context = _context()
    service = DraftService.__new__(DraftService)
    service.deterministic_semantic_inspector = DeterministicSemanticInspector()
    service.semantic_inspector = service.deterministic_semantic_inspector
    filtered, _, _ = await service._validate_context_semantics(context)
    recent = (
        "fundraising is just sales with extra steps",
        "finally an official anti-yap switch",
    )
    request = GenerationRequest(
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
        recent_posts=recent,
    )
    provider = ReplyRecordingProvider()

    result = await provider.generate_structured(request)

    assert provider.requests[0].recent_posts == recent
    assert {item.contribution_type for item in request.angle_candidates} >= {
        "plain_observation",
        "direct_response",
    }
    scored = [
        (
            DraftService._variant_selection_score(variant, _passing_report(), request),
            variant,
        )
        for variant in result.variants
    ]
    selected = max(scored, key=lambda item: item[0])[1]
    assert selected.contribution_type == "plain_observation"


@pytest.mark.asyncio
async def test_semantic_filter_keeps_exploratory_humor_but_still_caps_grounded_families() -> None:
    context = _context()
    service = DraftService.__new__(DraftService)
    service.deterministic_semantic_inspector = DeterministicSemanticInspector()
    service.semantic_inspector = service.deterministic_semantic_inspector

    filtered, _, inspections = await service._validate_context_semantics(context)

    retained = {item.contribution_type for item in filtered.angles}
    assert "plain_observation" in retained
    assert "direct_response" in retained
    assert "dry_humor" in retained
    assert len(filtered.angles) <= 3
    assert any(item.contribution_id == "reply-dry-humor" and item.grounded for item in inspections)


def test_regeneration_contract_and_external_fallback_are_atomic(tmp_path: Path) -> None:
    runtime = HumanizerRuntimeLoader(configured_path=tmp_path / "missing" / "SKILL.md").load()

    assert runtime.source == "bundled"
    assert runtime.version == "3.6.0"
    assert "ordinary human reaction is the default" in runtime.initial_contract
    assert "ordinary human reaction is the default" in runtime.revision_contract
    reply_range = f"4{chr(8211)}35 words"
    assert reply_range in runtime.initial_contract
    assert reply_range in runtime.revision_contract


def test_provider_payload_carries_recent_history_and_new_reply_policy() -> None:
    from app.providers.responses_provider import ResponsesLLMProvider

    packet = EvidencePacket(
        mode=GenerationMode.REPLY,
        topic="reply",
        items=(
            EvidenceItem(
                evidence_id="source",
                source_type="x_post",
                text="The social manager has ten other jobs and no budget.",
            ),
        ),
        language="en",
        requested_format="reply",
    )
    request = GenerationRequest(
        idea_summary="reply",
        content_type=ContentType.REPLY,
        language="en",
        generation_mode=GenerationMode.REPLY,
        evidence_packet=packet,
        recent_posts=("finally an official anti-yap switch",),
    )

    payload = ResponsesLLMProvider._input_payload(request)

    assert '"recent_account_posts":["finally an official anti-yap switch"]' in payload
    assert '"reply_preferred_word_range":[4,35]' in payload
    assert '"prefer_least_constructed_supported_reply":true' in payload
    assert '"reply_recent_output_structural_diversity":true' in payload


def test_recent_syntax_reuse_lowers_reply_recommendation() -> None:
    packet = EvidencePacket(
        mode=GenerationMode.REPLY,
        topic="reply",
        items=(
            EvidenceItem(
                evidence_id="source",
                source_type="x_post",
                text="The team added more steps to onboarding.",
            ),
        ),
        language="en",
        requested_format="reply",
    )
    request = GenerationRequest(
        idea_summary="reply",
        content_type=ContentType.REPLY,
        language="en",
        generation_mode=GenerationMode.REPLY,
        evidence_packet=packet,
        recent_posts=("fundraising is just sales with extra steps",),
    )
    fresh = GenerationVariant(
        label="fresh",
        text="that is a lot of setup for something meant to make onboarding easier",
        similarity_risk=0.0,
        angle_id="fresh",
        angle_type="plain_update",
        evidence_ids=("source",),
        confidence="high",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
        contribution_id="fresh",
        contribution_type="plain_observation",
    )
    reused = GenerationVariant(
        label="reused",
        text="onboarding is just paperwork with extra steps",
        similarity_risk=0.0,
        angle_id="reused",
        angle_type="plain_update",
        evidence_ids=("source",),
        confidence="high",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
        contribution_id="reused",
        contribution_type="plain_observation",
    )
    fresh_report = assess_variant(
        fresh,
        evidence_packet=packet,
        recent_posts=request.recent_posts,
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )
    reused_report = assess_variant(
        reused,
        evidence_packet=packet,
        recent_posts=request.recent_posts,
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert "RECENT_SYNTAX_REUSE" in {issue.code for issue in reused_report.issues}
    assert DraftService._variant_selection_score(fresh, fresh_report, request) > (
        DraftService._variant_selection_score(reused, reused_report, request)
    )
