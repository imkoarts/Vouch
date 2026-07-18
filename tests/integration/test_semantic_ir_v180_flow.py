from __future__ import annotations

from app.domain.enums import ContentType
from app.schemas.content import GenerationVariant, UntrustedSourceData
from app.services.claim_extraction import DeterministicClaimExtractor
from app.services.editorial_quality import assess_variant, inspect_reply_naturalness
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


def _variant(context, family: str, text: str) -> GenerationVariant:
    candidate = next(item for item in context.angles if item.contribution_type == family)
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


def _inspection_service() -> FinalTextInspectionService:
    claims = DeterministicClaimExtractor()
    semantics = DeterministicSemanticInspector()
    return FinalTextInspectionService(
        deterministic_claim_extractor=claims,
        claim_extractor=claims,
        deterministic_semantic_inspector=semantics,
        semantic_inspector=semantics,
    )


async def test_mock_semantic_inspector_emits_typed_ir() -> None:
    context = _context("The server died. Two patients died during the outage.")
    contribution = next(item for item in context.angles if item.contribution_type == "dry_humor")
    inspector = DeterministicSemanticInspector()
    source_map = await inspector.inspect_source(context.evidence)

    result = await inspector.inspect_draft(
        context.evidence,
        contribution,
        "five stars for the outage",
        source_map,
        commentary_mode=False,
    )

    assert result.semantic_inspection is not None
    assert result.semantic_inspection.source_events
    assert result.semantic_inspection.evaluations
    assert "HUMOR_ON_TRAGEDY" in result.issue_codes


async def test_manual_and_live_reinspection_share_canonical_ir() -> None:
    context = _context("A hospital reported ten deaths after the outage.")
    contribution = next(
        item for item in context.angles if item.contribution_type == "plain_observation"
    )
    variant = _variant(context, "plain_observation", "stellar execution from the outage")
    service = _inspection_service()

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

    assert local.structured_semantic_inspection.semantic_inspection is not None
    assert live.structured_semantic_inspection.semantic_inspection is not None
    assert (
        local.structured_semantic_inspection.semantic_inspection
        == live.structured_semantic_inspection.semantic_inspection
    )
    assert {issue.code for issue in local.editorial_report.issues} == {
        issue.code for issue in live.editorial_report.issues
    }


def test_humor_rejection_keeps_plain_candidate_eligible() -> None:
    context = _context("A hospital reported ten deaths after the outage.")
    humor = _variant(context, "dry_humor", "five stars for the outage")
    plain = _variant(
        context,
        "plain_observation",
        "the deaths require a serious review of the outage response",
    )

    humor_report = assess_variant(
        humor,
        evidence_packet=context.evidence,
        editorial_intent="reply_reaction",
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )
    plain_report = assess_variant(
        plain,
        evidence_packet=context.evidence,
        editorial_intent="reply_reaction",
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert "HUMOR_ON_TRAGEDY" in {issue.code for issue in humor_report.issues}
    assert "HUMOR_ON_TRAGEDY" not in {issue.code for issue in plain_report.issues}
    assert plain_report.passed


def test_recent_shell_history_does_not_change_factual_grounding() -> None:
    context = _context("The company launched onboarding after a six-week pilot.")
    variant = _variant(context, "plain_observation", "onboarding is glorified paperwork")

    without_history = assess_variant(
        variant,
        evidence_packet=context.evidence,
        editorial_intent="reply_reaction",
        recent_posts=(),
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )
    with_history = assess_variant(
        variant,
        evidence_packet=context.evidence,
        editorial_intent="reply_reaction",
        recent_posts=("fundraising is just sales with extra steps",),
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert without_history.evidence_score == with_history.evidence_score
    assert without_history.specificity_score == with_history.specificity_score
    naturalness = inspect_reply_naturalness(
        variant,
        evidence_packet=context.evidence,
        recent_posts=("fundraising is just sales with extra steps",),
    )
    assert naturalness.recent_syntax_reuse
