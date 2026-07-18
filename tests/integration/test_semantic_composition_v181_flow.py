from __future__ import annotations

from app.domain.enums import ContentType
from app.schemas.content import GenerationVariant, SemanticInspection, UntrustedSourceData
from app.services.claim_extraction import DeterministicClaimExtractor
from app.services.editorial_quality import assess_variant, inspect_reply_naturalness
from app.services.final_text_inspection import FinalTextInspectionService
from app.services.generation_pipeline import build_generation_context
from app.services.semantic_adjudication import adjudicate_humor_safety
from app.services.semantic_classifier import DeterministicSemanticInspector
from app.services.semantic_extraction import extract_semantic_inspection
from app.services.semantic_reconciliation import reconcile_semantic_inspections


def _context(source: str):
    return build_generation_context(
        idea_summary="Reply to source",
        idea_explanation={"origin": "reply_to_source", "editorial_intent": "reply_reaction"},
        language="en",
        sources=(UntrustedSourceData(source_id="source", source_type="x_post", content=source),),
        content_type=ContentType.REPLY,
    )


def _variant(context, family: str, text: str) -> GenerationVariant:
    contribution = next(item for item in context.angles if item.contribution_type == family)
    return GenerationVariant(
        label=contribution.angle_id,
        text=text,
        similarity_risk=0.0,
        angle_id=contribution.angle_id,
        angle_type=contribution.angle_type,
        evidence_ids=contribution.evidence_ids,
        confidence=contribution.confidence,
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
        contribution_id=contribution.angle_id,
        contribution_type=contribution.contribution_type,
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


async def test_mock_provider_flow_returns_compositional_ir_and_coverage() -> None:
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
    assert result.semantic_inspection.source_coverage
    assert result.semantic_inspection.reply_coverage
    assert all(item.complete for item in result.semantic_inspection.reply_coverage)
    assert "HUMOR_ON_TRAGEDY" in result.issue_codes


async def test_manual_and_live_routes_share_coverage_aware_ir() -> None:
    context = _context("Two workers were incapacitated after the blast.")
    contribution = next(
        item for item in context.angles if item.contribution_type == "plain_observation"
    )
    variant = _variant(context, "plain_observation", "achievement unlocked")
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

    local_ir = local.structured_semantic_inspection.semantic_inspection
    live_ir = live.structured_semantic_inspection.semantic_inspection
    assert local_ir is not None and live_ir is not None
    assert local_ir == live_ir
    assert local_ir.source_coverage[0].unresolved_spans == ("incapacitated",)
    assert {issue.code for issue in local.editorial_report.issues} == {
        issue.code for issue in live.editorial_report.issues
    }


def test_unresolved_material_blocks_only_humorous_candidate() -> None:
    context = _context("Two workers were incapacitated after the blast.")
    humorous = _variant(context, "dry_humor", "achievement unlocked")
    plain = _variant(
        context,
        "plain_observation",
        "the reported condition needs confirmation before drawing conclusions",
    )

    humor_naturalness = inspect_reply_naturalness(humorous, evidence_packet=context.evidence)
    plain_naturalness = inspect_reply_naturalness(plain, evidence_packet=context.evidence)

    assert humor_naturalness.humor_safety is not None
    assert not humor_naturalness.humor_safety.suitable_for_humor
    assert plain_naturalness.humor_safety is None


def test_provider_none_cannot_approve_application_blocked_candidate() -> None:
    source = "Two patients died during the outage."
    reply = "five stars for the outage"
    deterministic = extract_semantic_inspection(source, reply)
    canonical = reconcile_semantic_inspections(
        deterministic,
        SemanticInspection(),
        source_text=source,
        reply_text=reply,
    )

    assert not adjudicate_humor_safety(canonical).suitable_for_humor

    context = _context(source)
    report = assess_variant(
        _variant(context, "plain_observation", reply),
        evidence_packet=context.evidence,
        editorial_intent="reply_reaction",
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )
    assert not report.passed
    assert "HUMOR_ON_TRAGEDY" in {issue.code for issue in report.issues}


async def test_result_bound_medical_harm_and_recognition_use_same_canonical_routes() -> None:
    context = _context("The deployment failed. The outage sent four technicians to intensive care.")
    contribution = next(
        item for item in context.angles if item.contribution_type == "plain_observation"
    )
    variant = _variant(context, "plain_observation", "take a bow, outage")
    service = _inspection_service()

    local = service.inspect_local(
        variant.text,
        variant=variant,
        packet=context.evidence,
        content_mode=context.content_mode,
        contribution=contribution,
        coverage=context.source_coverage,
    )
    live = await service.inspect(
        variant.text,
        variant=variant,
        packet=context.evidence,
        content_mode=context.content_mode,
        contribution=contribution,
        coverage=context.source_coverage,
        allow_live=True,
    )

    for result in (local, live):
        semantic = result.structured_semantic_inspection
        assert semantic.humor_intent_proposal is not None
        assert semantic.humor_intent_proposal.humorous_treatment
        assert semantic.humor_safety is not None
        assert not semantic.humor_safety.suitable_for_humor
        assert "HUMOR_ON_TRAGEDY" in semantic.issue_codes
        assert semantic.semantic_inspection is not None
        assert {event.event_type for event in semantic.semantic_inspection.source_events} >= {
            "technical_failure",
            "intensive_care",
        }


def test_unknown_evaluation_on_harmful_source_fails_closed_without_removing_plain_family() -> None:
    context = _context("Two patients died during the outage.")
    uncertain = _variant(
        context,
        "plain_observation",
        "what a masterclass from the outage",
    )
    respectful = _variant(
        context,
        "plain_observation",
        "the deaths make a celebratory response inappropriate",
    )

    uncertain_report = assess_variant(
        uncertain,
        evidence_packet=context.evidence,
        editorial_intent="reply_reaction",
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )
    respectful_report = assess_variant(
        respectful,
        evidence_packet=context.evidence,
        editorial_intent="reply_reaction",
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert not uncertain_report.passed
    assert "HUMOR_INTENT_UNCERTAIN" in {issue.code for issue in uncertain_report.issues}
    assert "HUMOR_ON_TRAGEDY" in {issue.code for issue in uncertain_report.issues}
    assert "HUMOR_INTENT_UNCERTAIN" not in {issue.code for issue in respectful_report.issues}
    assert "HUMOR_ON_TRAGEDY" not in {issue.code for issue in respectful_report.issues}
