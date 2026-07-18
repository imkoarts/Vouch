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
        sources=(
            UntrustedSourceData(
                source_id="source",
                source_type="x_post",
                content=source,
            ),
        ),
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
    variant = _variant(context, reply, contribution_type)
    return await _service().inspect(
        reply,
        variant=variant,
        packet=context.evidence,
        content_mode=context.content_mode,
        contribution=contribution,
        coverage=context.source_coverage,
        allow_live=False,
    )


async def test_possible_harm_and_incomplete_reply_use_canonical_candidate_fallback() -> None:
    result = await _inspect(
        "The impact left one cyclist seeing halos.",
        "the incident merits a velvet banner",
    )

    extraction = result.semantic_extraction_result
    assert extraction is not None
    assert "possible_harm" in extraction.escalation.material_categories
    assert "HUMOR_INTENT_UNCERTAIN" in result.structured_semantic_inspection.issue_codes
    assert "HUMOR_ON_TRAGEDY" in result.structured_semantic_inspection.issue_codes


async def test_negated_factual_observation_remains_eligible_on_fatal_source() -> None:
    result = await _inspect(
        "A clinic reported six deaths after the backup generator failed.",
        "the clinic has not released the maintenance record",
    )

    extraction = result.semantic_extraction_result
    assert extraction is not None
    assert extraction.canonical.reply_communicative_function == "plain_observation"
    assert extraction.canonical.reply_communicative_function_confidence == "high"
    assert "HUMOR_INTENT_UNCERTAIN" not in result.structured_semantic_inspection.issue_codes
    assert "HUMOR_ON_TRAGEDY" not in result.structured_semantic_inspection.issue_codes


async def test_passive_evidential_qualification_suppresses_false_shell() -> None:
    result = await _inspect(
        "A clinic reported six deaths after the backup generator failed.",
        "the cause has not been verified",
        "plain_observation",
    )

    extraction = result.semantic_extraction_result
    assert extraction is not None
    assert extraction.canonical.reply_communicative_function == "qualification"
    assert extraction.canonical.reply_communicative_function_confidence == "high"
    assert extraction.canonical.reply_shells == ()
    assert "HUMOR_INTENT_UNCERTAIN" not in result.structured_semantic_inspection.issue_codes


async def test_genuine_question_remains_available_on_possible_harm_source() -> None:
    result = await _inspect(
        "The solvent left two mechanics with reduced sensation in one arm.",
        "what medical support is being provided?",
        "direct_response",
    )

    extraction = result.semantic_extraction_result
    assert extraction is not None
    assert "possible_harm" in extraction.escalation.material_categories
    assert extraction.canonical.reply_communicative_function == "question"
    assert "HUMOR_INTENT_UNCERTAIN" not in result.structured_semantic_inspection.issue_codes
    assert "HUMOR_ON_TRAGEDY" not in result.structured_semantic_inspection.issue_codes
