from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.enums import ContentType
from app.schemas.content import (
    EvaluationRelation,
    GenerationVariant,
    SemanticCoverageResolution,
    SemanticInspection,
    SemanticProviderProposal,
    SemanticProviderRequest,
    UntrustedSourceData,
)
from app.services.claim_extraction import DeterministicClaimExtractor
from app.services.final_text_inspection import FinalTextInspectionService
from app.services.generation_pipeline import build_generation_context
from app.services.semantic_classifier import DeterministicSemanticInspector
from app.services.semantic_extraction_coordinator import SemanticExtractionCoordinator


def _context(source: str):
    return build_generation_context(
        idea_summary="Reply to source",
        idea_explanation={"origin": "reply_to_source", "editorial_intent": "reply_reaction"},
        language="en",
        sources=(UntrustedSourceData(source_id="source", source_type="x_post", content=source),),
        content_type=ContentType.REPLY,
    )


def _variant(context, text: str) -> GenerationVariant:
    contribution = next(
        item for item in context.angles if item.contribution_type == "plain_observation"
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


def _evaluation_proposal(source: str, reply: str) -> SemanticProviderProposal:
    local = SemanticExtractionCoordinator().inspect_local(source, reply)
    event_id = next(
        event.event_id for event in local.deterministic.source_events if event.event_type == "death"
    )
    resolutions = tuple(
        SemanticCoverageResolution(
            item_id=item.item_id,
            side="reply",
            clause_id=item.clause_id,
            clause_span=item.clause_span,
            unresolved_span=item.unresolved_span,
            category="possible_evaluation",
            disposition="typed_relation",
            evidence_spans=(reply,),
        )
        for coverage in local.deterministic.reply_coverage
        for item in coverage.unresolved_items
        if item.category == "possible_evaluation"
    )
    return SemanticProviderProposal(
        inspection=SemanticInspection(
            evaluations=(
                EvaluationRelation(
                    target_event_id=event_id,
                    evaluation_type="reward",
                    relation_subtype="reward",
                    polarity="positive",
                    literal_reading_plausible=False,
                    confidence="high",
                    evidence_spans=(reply,),
                ),
            ),
        ),
        coverage_resolutions=resolutions,
    )


@dataclass
class FakeHybridSemanticInspector(DeterministicSemanticInspector):
    proposal: SemanticProviderProposal
    extraction_requests: list[SemanticProviderRequest] = field(default_factory=list)

    async def extract_semantics(self, request: SemanticProviderRequest) -> SemanticProviderProposal:
        self.extraction_requests.append(request)
        return self.proposal


def _service(provider: FakeHybridSemanticInspector) -> FinalTextInspectionService:
    claims = DeterministicClaimExtractor()
    deterministic = DeterministicSemanticInspector()
    return FinalTextInspectionService(
        deterministic_claim_extractor=claims,
        claim_extractor=claims,
        deterministic_semantic_inspector=deterministic,
        semantic_inspector=provider,
    )


async def test_live_final_text_route_escalates_and_uses_canonical_provider_ir() -> None:
    source = "A clinic reported eight deaths after the blackout."
    reply = "a ceremonial sash for the grid"
    context = _context(source)
    contribution = next(
        item for item in context.angles if item.contribution_type == "plain_observation"
    )
    variant = _variant(context, reply)
    provider = FakeHybridSemanticInspector(_evaluation_proposal(source, reply))

    result = await _service(provider).inspect(
        reply,
        variant=variant,
        packet=context.evidence,
        content_mode=context.content_mode,
        contribution=contribution,
        coverage=context.source_coverage,
        allow_live=True,
    )

    extraction = result.semantic_extraction_result
    assert extraction is not None
    assert extraction.escalation.required
    assert extraction.provider_used
    assert extraction.provider_validation_errors == ()
    assert len(provider.extraction_requests) == 1
    assert "HUMOR_ON_TRAGEDY" in result.structured_semantic_inspection.issue_codes
    assert result.structured_semantic_inspection.humor_intent_proposal is not None
    assert result.structured_semantic_inspection.humor_intent_proposal.humorous_treatment


async def test_offline_and_manual_routes_preserve_uncertainty_without_provider_call() -> None:
    source = "A clinic reported eight deaths after the blackout."
    reply = "a ceremonial sash for the grid"
    context = _context(source)
    contribution = next(
        item for item in context.angles if item.contribution_type == "plain_observation"
    )
    variant = _variant(context, reply)
    provider = FakeHybridSemanticInspector(_evaluation_proposal(source, reply))
    service = _service(provider)

    manual = service.inspect_local(
        reply,
        variant=variant,
        packet=context.evidence,
        content_mode=context.content_mode,
        contribution=contribution,
        coverage=context.source_coverage,
    )
    offline = await service.inspect(
        reply,
        variant=variant,
        packet=context.evidence,
        content_mode=context.content_mode,
        contribution=contribution,
        coverage=context.source_coverage,
        allow_live=False,
    )

    assert provider.extraction_requests == []
    for result in (manual, offline):
        extraction = result.semantic_extraction_result
        assert extraction is not None
        assert extraction.escalation.required
        assert not extraction.provider_used
        assert "possible_evaluation" in extraction.escalation.material_categories
        assert "HUMOR_INTENT_UNCERTAIN" in result.structured_semantic_inspection.issue_codes
        assert "HUMOR_ON_TRAGEDY" in result.structured_semantic_inspection.issue_codes


async def test_complete_deterministic_coverage_does_not_spend_provider_call() -> None:
    source = "The clinic restored power and every patient recovered safely."
    reply = "great work from the response team"
    context = _context(source)
    contribution = next(
        item for item in context.angles if item.contribution_type == "plain_observation"
    )
    variant = _variant(context, reply)
    provider = FakeHybridSemanticInspector(SemanticProviderProposal())

    result = await _service(provider).inspect(
        reply,
        variant=variant,
        packet=context.evidence,
        content_mode=context.content_mode,
        contribution=contribution,
        coverage=context.source_coverage,
        allow_live=True,
    )

    extraction = result.semantic_extraction_result
    assert extraction is not None
    assert not extraction.escalation.required
    assert not extraction.provider_used
    assert provider.extraction_requests == []
    assert "HUMOR_ON_TRAGEDY" not in result.structured_semantic_inspection.issue_codes
