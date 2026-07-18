"""Canonical hybrid semantic extraction coordinator.

The coordinator owns deterministic extraction, coverage-driven escalation, structured-provider
validation, reconciliation, and the final typed result.  It performs no application adjudication.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Protocol

from app.schemas.content import (
    SemanticEscalationDecision,
    SemanticInspection,
    SemanticInspectionResult,
    SemanticProviderProposal,
    SemanticProviderRequest,
    SemanticUnresolvedItem,
)
from app.services.semantic_extraction import extract_semantic_inspection
from app.services.semantic_provider_validation import validate_semantic_provider_proposal
from app.services.semantic_reconciliation import reconcile_semantic_provider_proposal

MaterialCategory = Literal[
    "possible_harm",
    "possible_evaluation",
    "possible_reduction",
    "unknown_referent",
    "unknown_predicate",
    "incomplete_proposition",
]

_MATERIAL_CATEGORIES: set[MaterialCategory] = {
    "possible_harm",
    "possible_evaluation",
    "possible_reduction",
    "unknown_referent",
    "unknown_predicate",
    "incomplete_proposition",
}


class SemanticExtractionProviderPort(Protocol):
    """Bounded structured extraction only; no policy or publication authority."""

    async def extract_semantics(
        self,
        request: SemanticProviderRequest,
    ) -> SemanticProviderProposal: ...


def decide_semantic_escalation(inspection: SemanticInspection) -> SemanticEscalationDecision:
    """Escalate only clauses with unresolved material semantic coverage."""

    material_categories: list[MaterialCategory] = []
    clause_spans: list[str] = []
    item_ids: list[str] = []
    for coverage in (*inspection.source_coverage, *inspection.reply_coverage):
        material_items = tuple(
            item for item in coverage.unresolved_items if item.category in _MATERIAL_CATEGORIES
        )
        if material_items:
            for item in material_items:
                if item.category in _MATERIAL_CATEGORIES:
                    material_categories.append(item.category)
                    item_ids.append(item.item_id)
            clause_spans.append(coverage.clause_span)
            continue
        categories: tuple[MaterialCategory, ...] = tuple(
            category
            for category in coverage.unresolved_categories
            if category in _MATERIAL_CATEGORIES
        )
        if not categories:
            continue
        material_categories.extend(categories)
        clause_spans.append(coverage.clause_span)
    categories = tuple(dict.fromkeys(material_categories))
    clauses = tuple(dict.fromkeys(clause_spans))
    return SemanticEscalationDecision(
        required=bool(categories),
        material_categories=categories,
        clause_spans=clauses,
        unresolved_item_ids=tuple(dict.fromkeys(item_ids)),
        reason=(
            "material deterministic semantic coverage remains unresolved"
            if categories
            else "deterministic safety-relevant semantic coverage is complete"
        ),
    )


def _bounded_clauses(
    inspection: SemanticInspection,
    escalation: SemanticEscalationDecision,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    selected = set(escalation.clause_spans)
    safety_context_clause_ids = {
        event.clause_id
        for event in inspection.source_events
        if event.explicit_outcome
        and event.event_type
        in {
            "death",
            "serious_injury",
            "critical_condition",
            "hospitalization",
            "intensive_care",
            "permanent_injury",
            "non_survival",
            "abuse",
            "self_harm",
            "personal_tragedy",
        }
    }
    source = tuple(
        coverage.clause_span
        for coverage in inspection.source_coverage
        if coverage.clause_span in selected or coverage.clause_id in safety_context_clause_ids
    )
    reply = tuple(
        coverage.clause_span
        for coverage in inspection.reply_coverage
        if coverage.clause_span in selected
    )
    return tuple(dict.fromkeys(source)), tuple(dict.fromkeys(reply))


def _span_in_any(span: str, clauses: tuple[str, ...]) -> bool:
    return bool(span) and any(span in clause for clause in clauses)


def _bounded_deterministic_inspection(
    inspection: SemanticInspection,
    *,
    source_clauses: tuple[str, ...],
    reply_clauses: tuple[str, ...],
) -> SemanticInspection:
    """Limit the provider packet to selected unresolved clauses and their typed candidates."""

    source_entities = tuple(
        entity
        for entity in inspection.source_entities
        if _span_in_any(entity.text_span, source_clauses)
    )
    reply_entities = tuple(
        entity
        for entity in inspection.reply_entities
        if _span_in_any(entity.text_span, reply_clauses)
    )
    source_events = tuple(
        event
        for event in inspection.source_events
        if any(_span_in_any(span, source_clauses) for span in event.evidence_spans)
    )
    reply_events = tuple(
        event
        for event in inspection.reply_events
        if any(_span_in_any(span, reply_clauses) for span in event.evidence_spans)
    )
    evaluations = tuple(
        relation
        for relation in inspection.evaluations
        if any(_span_in_any(span, reply_clauses) for span in relation.evidence_spans)
    )
    reply_shells = tuple(
        shell
        for shell in inspection.reply_shells
        if any(_span_in_any(span, reply_clauses) for span in shell.evidence_spans)
    )
    source_coverage = tuple(
        item for item in inspection.source_coverage if item.clause_span in source_clauses
    )
    reply_coverage = tuple(
        item for item in inspection.reply_coverage if item.clause_span in reply_clauses
    )
    return SemanticInspection(
        source_entities=source_entities,
        predicate_scopes=tuple(
            scope
            for scope in inspection.predicate_scopes
            if any(
                scope.text_span in clause or clause in scope.text_span
                for clause in (*source_clauses, *reply_clauses)
            )
        ),
        source_events=source_events,
        reply_entities=reply_entities,
        reply_events=reply_events,
        evaluations=evaluations,
        reply_shells=reply_shells,
        result_state_materiality=inspection.result_state_materiality,
        human_function_impacts=inspection.human_function_impacts,
        reply_propositions=inspection.reply_propositions,
        literal_proposition_frames=inspection.literal_proposition_frames,
        reply_semantic_completeness=inspection.reply_semantic_completeness,
        source_coverage=source_coverage,
        reply_coverage=reply_coverage,
        reply_communicative_function=inspection.reply_communicative_function,
        reply_communicative_function_confidence=inspection.reply_communicative_function_confidence,
        reply_communicative_evidence_spans=inspection.reply_communicative_evidence_spans,
        reply_self_reference=inspection.reply_self_reference,
        extraction_conflicts=inspection.extraction_conflicts,
        unresolved_material=tuple(
            dict.fromkeys(
                span
                for item in (*source_coverage, *reply_coverage)
                for span in item.unresolved_spans
            )
        ),
    )


def _unresolved_spans(inspection: SemanticInspection) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            span
            for coverage in (*inspection.source_coverage, *inspection.reply_coverage)
            for span in coverage.unresolved_spans
        )
    )


def _selected_unresolved_items(
    inspection: SemanticInspection, escalation: SemanticEscalationDecision
) -> tuple[SemanticUnresolvedItem, ...]:
    selected = set(escalation.unresolved_item_ids)
    return tuple(
        item
        for coverage in (*inspection.source_coverage, *inspection.reply_coverage)
        for item in coverage.unresolved_items
        if not selected or item.item_id in selected
    )


def _request_id(items: tuple[SemanticUnresolvedItem, ...]) -> str:
    payload = "|".join(sorted(item.item_id for item in items)).encode()
    return f"semantic-{hashlib.sha256(payload).hexdigest()[:20]}"


class SemanticExtractionCoordinator:
    """One extraction/reconciliation route for local, live, mock, and audit workflows."""

    def __init__(self, provider: SemanticExtractionProviderPort | None = None) -> None:
        self.provider = provider

    def inspect_local(self, source_text: str, reply_text: str = "") -> SemanticInspectionResult:
        deterministic = extract_semantic_inspection(source_text, reply_text)
        escalation = decide_semantic_escalation(deterministic)
        return SemanticInspectionResult(
            deterministic=deterministic,
            canonical=deterministic,
            escalation=escalation,
            provider_used=False,
            unresolved_after_reconciliation=_unresolved_spans(deterministic),
        )

    async def inspect(
        self,
        source_text: str,
        reply_text: str = "",
        *,
        allow_live: bool = True,
    ) -> SemanticInspectionResult:
        local = self.inspect_local(source_text, reply_text)
        if not local.escalation.required or not allow_live or self.provider is None:
            return local

        source_clauses, reply_clauses = _bounded_clauses(
            local.deterministic,
            local.escalation,
        )
        bounded_deterministic = _bounded_deterministic_inspection(
            local.deterministic,
            source_clauses=source_clauses,
            reply_clauses=reply_clauses,
        )
        unresolved_items = _selected_unresolved_items(local.deterministic, local.escalation)
        request = SemanticProviderRequest(
            source_clauses=source_clauses,
            reply_clauses=reply_clauses,
            request_id=_request_id(unresolved_items),
            deterministic=bounded_deterministic,
            unresolved_items=unresolved_items,
            material_categories=local.escalation.material_categories,
            clause_spans=local.escalation.clause_spans,
        )
        try:
            proposal = await self.provider.extract_semantics(request)
        except Exception as exc:  # Provider failure remains explicit uncertainty, never safety.
            return SemanticInspectionResult(
                deterministic=local.deterministic,
                canonical=local.deterministic,
                escalation=local.escalation,
                provider_used=True,
                provider_validation_errors=(f"provider_error:{type(exc).__name__}",),
                unresolved_after_reconciliation=_unresolved_spans(local.deterministic),
            )

        validation_errors = validate_semantic_provider_proposal(
            local.deterministic,
            proposal,
            source_clauses=source_clauses,
            reply_clauses=reply_clauses,
        )
        if validation_errors:
            return SemanticInspectionResult(
                deterministic=local.deterministic,
                provider_proposal=proposal,
                canonical=local.deterministic,
                escalation=local.escalation,
                provider_used=True,
                provider_validation_errors=validation_errors,
                unresolved_after_reconciliation=_unresolved_spans(local.deterministic),
            )

        canonical = reconcile_semantic_provider_proposal(local.deterministic, proposal)
        return SemanticInspectionResult(
            deterministic=local.deterministic,
            provider_proposal=proposal,
            canonical=canonical,
            escalation=local.escalation,
            provider_used=True,
            unresolved_after_reconciliation=_unresolved_spans(canonical),
        )
