"""Canonical final-text inspection shared by every draft-content workflow."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from app.schemas.content import (
    AngleCandidate,
    CanonicalClaim,
    ClaimExtractionContext,
    ContentModeDecision,
    DraftSemanticInspection,
    EditorialReport,
    EvidencePacket,
    EvidenceSpan,
    FinalTextInspection,
    GenerationVariant,
    SourceCoverage,
    SourceSemanticMap,
)
from app.services.claim_extraction import (
    ClaimExtractorPort,
    DeterministicClaimExtractor,
    merge_extractor_results,
)
from app.services.claim_structure import extract_claims
from app.services.editorial_inspection import (
    apply_structured_semantic_inspection,
    inspect_editorial,
)
from app.services.factual_inspection import inspect_factual_integrity
from app.services.semantic_classifier import (
    DeterministicSemanticInspector,
    SemanticInspectorPort,
    derive_source_semantic_map,
    inspect_draft_deterministic,
    merge_semantic_inspections,
    merge_source_semantic_maps,
)
from app.services.semantic_extraction_coordinator import (
    SemanticExtractionCoordinator,
    SemanticExtractionProviderPort,
)


class FinalTextInspectionService:
    """Run one fail-closed inspection contract for all final rendered text paths."""

    def __init__(
        self,
        *,
        deterministic_claim_extractor: DeterministicClaimExtractor,
        claim_extractor: ClaimExtractorPort,
        deterministic_semantic_inspector: DeterministicSemanticInspector,
        semantic_inspector: SemanticInspectorPort,
    ) -> None:
        self.deterministic_claim_extractor = deterministic_claim_extractor
        self.claim_extractor = claim_extractor
        self.deterministic_semantic_inspector = deterministic_semantic_inspector
        self.semantic_inspector = semantic_inspector
        semantic_provider = (
            cast(SemanticExtractionProviderPort, semantic_inspector)
            if semantic_inspector is not deterministic_semantic_inspector
            and hasattr(semantic_inspector, "extract_semantics")
            else None
        )
        self.semantic_extraction_coordinator = SemanticExtractionCoordinator(semantic_provider)

    @property
    def live_inspection_configured(self) -> bool:
        return (
            self.claim_extractor is not self.deterministic_claim_extractor
            or self.semantic_inspector is not self.deterministic_semantic_inspector
        )

    @staticmethod
    def _editorial_report(
        text: str,
        *,
        packet: EvidencePacket,
        content_mode: ContentModeDecision,
        contribution: AngleCandidate,
        coverage: SourceCoverage | None,
        structured_semantic: DraftSemanticInspection,
    ) -> EditorialReport:
        deterministic_editorial = inspect_editorial(
            text,
            packet=packet,
            contribution=contribution,
            content_mode=content_mode,
            coverage=coverage,
            preferred_length_min=content_mode.preferred_length_min,
        )
        return apply_structured_semantic_inspection(
            deterministic_editorial,
            structured_semantic,
        )

    def inspect_local(
        self,
        text: str,
        *,
        variant: GenerationVariant,
        packet: EvidencePacket,
        content_mode: ContentModeDecision,
        contribution: AngleCandidate,
        evidence_spans: Sequence[EvidenceSpan] = (),
        previous_claims: Sequence[CanonicalClaim] = (),
        source_map: SourceSemanticMap | None = None,
        coverage: SourceCoverage | None = None,
        semantic_reinspection_required: bool = False,
    ) -> FinalTextInspection:
        """Run the canonical deterministic subset without starting an event loop.

        Manual file reconciliation is synchronous. It uses this exact service and remains
        fail-closed when live providers are configured; ``reinspect_current_version`` later calls
        :meth:`inspect` to complete the provider-backed inspection.
        """

        extracted_claims = extract_claims(text, prefix="D")
        source_text = " ".join(item.text for item in packet.items)
        semantic_result = self.semantic_extraction_coordinator.inspect_local(source_text, text)
        merged_map = source_map or derive_source_semantic_map(packet)
        structured_semantic = inspect_draft_deterministic(
            packet,
            contribution,
            text,
            merged_map,
            commentary_mode=content_mode.mode == "commentary",
            semantic_ir=semantic_result.canonical,
        )
        editorial = self._editorial_report(
            text,
            packet=packet,
            content_mode=content_mode,
            contribution=contribution,
            coverage=coverage,
            structured_semantic=structured_semantic,
        )
        factual = inspect_factual_integrity(
            variant,
            packet=packet,
            evidence_spans=evidence_spans,
            factual_inspection_required=content_mode.factual_inspection_required,
            previous_claims=previous_claims,
            extracted_claims=extracted_claims,
            provider_claims=(),
            semantic_reinspection_required=semantic_reinspection_required,
        )
        return FinalTextInspection(
            extracted_claims=extracted_claims,
            source_semantic_map=merged_map,
            structured_semantic_inspection=structured_semantic,
            semantic_extraction_result=semantic_result,
            editorial_report=editorial,
            factual_report=factual,
        )

    async def inspect(
        self,
        text: str,
        *,
        variant: GenerationVariant,
        packet: EvidencePacket,
        content_mode: ContentModeDecision,
        contribution: AngleCandidate,
        evidence_spans: Sequence[EvidenceSpan] = (),
        previous_claims: Sequence[CanonicalClaim] = (),
        source_map: SourceSemanticMap | None = None,
        coverage: SourceCoverage | None = None,
        structured_semantic: DraftSemanticInspection | None = None,
        allow_live: bool = True,
        semantic_reinspection_required: bool = False,
    ) -> FinalTextInspection:
        """Run configured claim extraction, semantic inspection, and factual adjudication."""

        context = ClaimExtractionContext(
            mode=content_mode.mode,
            strict=content_mode.factual_inspection_required,
            source_role=content_mode.source_role,
        )
        deterministic_claims = await self.deterministic_claim_extractor.extract_claims(
            text,
            context=context,
        )
        provider_claims = None
        if allow_live and self.claim_extractor is not self.deterministic_claim_extractor:
            provider_claims = await self.claim_extractor.extract_claims(text, context=context)
        extracted_claims = merge_extractor_results(text, deterministic_claims, provider_claims)

        if source_map is None:
            deterministic_map = derive_source_semantic_map(packet)
            provider_map = None
            if allow_live and self.semantic_inspector is not self.deterministic_semantic_inspector:
                provider_map = await self.semantic_inspector.inspect_source(packet)
            merged_map = merge_source_semantic_maps(packet, deterministic_map, provider_map)
        else:
            merged_map = source_map

        source_text = " ".join(item.text for item in packet.items)
        semantic_result = await self.semantic_extraction_coordinator.inspect(
            source_text,
            text,
            allow_live=allow_live,
        )

        if structured_semantic is None:
            deterministic_semantic = inspect_draft_deterministic(
                packet,
                contribution,
                text,
                merged_map,
                commentary_mode=content_mode.mode == "commentary",
                semantic_ir=semantic_result.canonical,
            )
            provider_semantic = None
            if allow_live and self.semantic_inspector is not self.deterministic_semantic_inspector:
                provider_semantic = await self.semantic_inspector.inspect_draft(
                    packet,
                    contribution,
                    text,
                    merged_map,
                    commentary_mode=content_mode.mode == "commentary",
                )
            resolved_semantic = merge_semantic_inspections(
                deterministic_semantic,
                provider_semantic,
                text=text,
                source_text=source_text,
                canonical_ir=semantic_result.canonical,
            )
        else:
            resolved_semantic = structured_semantic.model_copy(
                update={"semantic_inspection": semantic_result.canonical}
            )

        editorial = self._editorial_report(
            text,
            packet=packet,
            content_mode=content_mode,
            contribution=contribution,
            coverage=coverage,
            structured_semantic=resolved_semantic,
        )
        factual = inspect_factual_integrity(
            variant,
            packet=packet,
            evidence_spans=evidence_spans,
            factual_inspection_required=content_mode.factual_inspection_required,
            previous_claims=previous_claims,
            extracted_claims=extracted_claims,
            provider_claims=provider_claims or (),
            semantic_reinspection_required=semantic_reinspection_required,
        )
        return FinalTextInspection(
            extracted_claims=extracted_claims,
            source_semantic_map=merged_map,
            structured_semantic_inspection=resolved_semantic,
            semantic_extraction_result=semantic_result,
            editorial_report=editorial,
            factual_report=factual,
        )
