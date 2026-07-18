from __future__ import annotations

from app.domain.enums import ContentType
from app.schemas.content import GenerationVariant, UntrustedSourceData
from app.services.claim_extraction import DeterministicClaimExtractor
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


def _variant(context, text: str, contribution_type: str = "plain_observation") -> GenerationVariant:
    contribution = next(
        item for item in context.angles if item.contribution_type == contribution_type
    )
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


def _service() -> FinalTextInspectionService:
    claims = DeterministicClaimExtractor()
    semantic = DeterministicSemanticInspector()
    return FinalTextInspectionService(
        deterministic_claim_extractor=claims,
        claim_extractor=claims,
        deterministic_semantic_inspector=semantic,
        semantic_inspector=semantic,
    )


async def _inspect(source: str, reply: str, contribution_type: str = "plain_observation"):
    context = _context(source)
    contribution = next(
        item for item in context.angles if item.contribution_type == contribution_type
    )
    return await _service().inspect(
        reply,
        variant=_variant(context, reply, contribution_type),
        packet=context.evidence,
        content_mode=context.content_mode,
        contribution=contribution,
        coverage=context.source_coverage,
        allow_live=False,
    )


async def test_bare_fragment_on_fatal_source_is_candidate_local_uncertainty() -> None:
    result = await _inspect(
        "A clinic reported five deaths after the cooling system failed.",
        "a ceremonial badge for the control unit",
    )

    assert "HUMOR_INTENT_UNCERTAIN" in result.structured_semantic_inspection.issue_codes
    assert "HUMOR_ON_TRAGEDY" in result.structured_semantic_inspection.issue_codes


async def test_possible_harm_source_blocks_evaluative_candidate() -> None:
    result = await _inspect(
        "The impact left two riders unable to keep their balance.",
        "what a triumph for the response team",
    )

    assert "HUMOR_ON_TRAGEDY" in result.structured_semantic_inspection.issue_codes


async def test_direct_remediation_remains_available_on_sensitive_source() -> None:
    result = await _inspect(
        "A clinic reported five deaths after the cooling system failed.",
        "the control unit needs an external audit",
        "direct_response",
    )

    extraction = result.semantic_extraction_result
    assert extraction is not None
    assert extraction.canonical.reply_communicative_function == "direct_response"
    assert "HUMOR_INTENT_UNCERTAIN" not in result.structured_semantic_inspection.issue_codes
    assert "HUMOR_ON_TRAGEDY" not in result.structured_semantic_inspection.issue_codes


async def test_evidential_qualification_remains_available_on_sensitive_source() -> None:
    result = await _inspect(
        "A clinic reported five deaths after the cooling system failed.",
        "the failure sequence remains under examination",
    )

    extraction = result.semantic_extraction_result
    assert extraction is not None
    assert extraction.canonical.reply_communicative_function == "qualification"
    assert extraction.canonical.reply_shells == ()
    assert "HUMOR_INTENT_UNCERTAIN" not in result.structured_semantic_inspection.issue_codes
