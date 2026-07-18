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


def _variant(context, text: str, contribution_type: str) -> GenerationVariant:
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


async def test_result_state_uncertainty_uses_canonical_route_and_blocks_humor_candidate() -> None:
    source = "The collision left two passengers glassy-eyed and barely rousable."
    context = _context(source)
    contribution = next(
        item for item in context.angles if item.contribution_type == "plain_observation"
    )
    variant = _variant(context, "a flawless showing from the collision", "plain_observation")

    result = await _service().inspect(
        variant.text,
        variant=variant,
        packet=context.evidence,
        content_mode=context.content_mode,
        contribution=contribution,
        coverage=context.source_coverage,
        allow_live=False,
    )

    extraction = result.semantic_extraction_result
    assert extraction is not None
    assert extraction.escalation.required
    assert "possible_harm" in extraction.escalation.material_categories
    assert "HUMOR_ON_TRAGEDY" in result.structured_semantic_inspection.issue_codes


async def test_respectful_question_remains_available_on_unresolved_result_state() -> None:
    source = "The collision left two passengers glassy-eyed and barely rousable."
    context = _context(source)
    contribution = next(
        item for item in context.angles if item.contribution_type == "direct_response"
    )
    variant = _variant(context, "what support is being provided now?", "direct_response")

    result = await _service().inspect(
        variant.text,
        variant=variant,
        packet=context.evidence,
        content_mode=context.content_mode,
        contribution=contribution,
        coverage=context.source_coverage,
        allow_live=False,
    )

    extraction = result.semantic_extraction_result
    assert extraction is not None
    assert extraction.escalation.required
    assert "possible_harm" in extraction.escalation.material_categories
    assert "HUMOR_INTENT_UNCERTAIN" not in result.structured_semantic_inspection.issue_codes
