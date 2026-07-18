"""Local validation for structured semantic-extraction provider proposals.

Provider output is untrusted data.  This module validates exact spans, typed references, and
coverage-resolution claims before application reconciliation.  It never makes safety decisions.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable

from app.schemas.content import (
    EvaluationRelation,
    SemanticCoverage,
    SemanticCoverageResolution,
    SemanticEntity,
    SemanticEvent,
    SemanticInspection,
    SemanticProviderProposal,
    SemanticReplyShell,
    SemanticUnresolvedItem,
    ValidatedSpan,
)


def _word_continuation(value: str) -> bool:
    return value == "_" or value.isalnum() or unicodedata.category(value).startswith("M")


def validate_span(span: str, clause: str, *, clause_id: str = "unscoped") -> ValidatedSpan | None:
    """Return the first exact Unicode-token-aligned match inside one clause."""

    if not span:
        return None
    cursor = 0
    while True:
        start = clause.find(span, cursor)
        if start < 0:
            return None
        end = start + len(span)
        start_ok = start == 0 or not (
            _word_continuation(span[0]) and _word_continuation(clause[start - 1])
        )
        end_ok = end == len(clause) or not (
            _word_continuation(span[-1]) and _word_continuation(clause[end])
        )
        match = ValidatedSpan(
            text=span,
            clause_id=clause_id,
            start=start,
            end=end,
            exact=True,
            token_start_boundary=start_ok,
            token_end_boundary=end_ok,
        )
        if start_ok and end_ok:
            return match
        cursor = start + 1


def _exact(span: str, haystacks: tuple[str, ...]) -> bool:
    return any(validate_span(span, text) is not None for text in haystacks)


def _entity_key(entity: SemanticEntity) -> tuple[str, str]:
    return entity.text_span, entity.entity_type


def _coverage_items(
    deterministic: SemanticInspection,
) -> dict[str, SemanticUnresolvedItem]:
    items: dict[str, SemanticUnresolvedItem] = {}
    for coverage in (*deterministic.source_coverage, *deterministic.reply_coverage):
        for item in coverage.unresolved_items:
            items[item.item_id] = item
    return items


def _validate_entities(
    entities: Iterable[SemanticEntity],
    haystacks: tuple[str, ...],
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    for entity in entities:
        if not _exact(entity.text_span, haystacks):
            errors.append(f"{label} entity span is not exact: {entity.text_span!r}")
    return errors


def _validate_event(
    event: SemanticEvent,
    *,
    haystacks: tuple[str, ...],
    entity_keys: set[tuple[str, str]],
    label: str,
) -> list[str]:
    errors: list[str] = []
    if not event.evidence_spans:
        errors.append(f"{label} event {event.event_id!r} has no evidence span")
    if not _exact(event.predicate, haystacks):
        errors.append(f"{label} event predicate is not exact: {event.predicate!r}")
    for span in event.evidence_spans:
        if not _exact(span, haystacks):
            errors.append(f"{label} event span is not exact: {span!r}")
    relation_clauses = tuple(
        clause
        for clause in haystacks
        if event.predicate in clause and all(span in clause for span in event.evidence_spans)
    )
    if not relation_clauses:
        errors.append(f"{label} event {event.event_id!r} has no single exact clause scope")
    for role_name, entity in (
        ("actor", event.actor),
        ("affected_entity", event.affected_entity),
        ("object", event.object),
    ):
        if entity is None:
            continue
        if _entity_key(entity) not in entity_keys:
            errors.append(
                f"{label} event {event.event_id!r} references missing {role_name}: "
                f"{entity.text_span!r}/{entity.entity_type}"
            )
        if relation_clauses and not any(entity.text_span in clause for clause in relation_clauses):
            errors.append(
                f"{label} event {event.event_id!r} crosses clause scope for {role_name}: "
                f"{entity.text_span!r}"
            )
    return errors


def _validate_evaluation(
    evaluation: EvaluationRelation,
    *,
    reply_haystacks: tuple[str, ...],
    entity_keys: set[tuple[str, str]],
    event_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    if not evaluation.evidence_spans:
        errors.append("evaluation has no evidence span")
    for span in evaluation.evidence_spans:
        if not _exact(span, reply_haystacks):
            errors.append(f"evaluation span is not exact: {span!r}")
    if (
        evaluation.target_entity is not None
        and _entity_key(evaluation.target_entity) not in entity_keys
    ):
        errors.append(
            "evaluation references missing target entity: "
            f"{evaluation.target_entity.text_span!r}/{evaluation.target_entity.entity_type}"
        )
    if evaluation.target_event_id is not None and evaluation.target_event_id not in event_ids:
        errors.append(f"evaluation references missing event: {evaluation.target_event_id!r}")
    return errors


def _validate_shell(shell: SemanticReplyShell, reply_haystacks: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    for span in shell.evidence_spans:
        if not _exact(span, reply_haystacks):
            errors.append(f"reply-shell span is not exact: {span!r}")
    for role_name, role_span in (
        ("subject", shell.subject_span),
        ("complement", shell.complement_span),
        ("decoration", shell.decoration_span),
        ("repackaging_marker", shell.repackaging_marker),
    ):
        if role_span is not None and not _exact(role_span, reply_haystacks):
            errors.append(f"reply-shell {role_name} span is not exact: {role_span!r}")
    return errors


def _validate_coverage(
    coverage_items: Iterable[SemanticCoverage],
    haystacks: tuple[str, ...],
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    for coverage in coverage_items:
        if not _exact(coverage.clause_span, haystacks):
            errors.append(f"{label} coverage clause is not exact: {coverage.clause_span!r}")
        for span in (*coverage.consumed_spans, *coverage.unresolved_spans):
            if span not in coverage.clause_span:
                errors.append(
                    f"{label} coverage span {span!r} is outside clause {coverage.clause_span!r}"
                )
    return errors


def _evidence_overlaps(
    resolution: SemanticCoverageResolution,
    relation_spans: tuple[str, ...],
) -> bool:
    return any(
        evidence in relation_span or relation_span in evidence
        for evidence in resolution.evidence_spans
        for relation_span in relation_spans
    )


def _evidence_covers_unresolved_role(
    resolution: SemanticCoverageResolution, proposal: SemanticProviderProposal
) -> bool:
    """Require evidence for the complete request-local semantic role.

    A resolution may quote the complete unresolved role directly.  It may also quote the exact
    relation head when the proposed typed relation itself carries an evidence span covering the
    complete role.  An unrelated token inside the role cannot resolve it.
    """

    unresolved = resolution.unresolved_span
    if any(unresolved in evidence for evidence in resolution.evidence_spans):
        return True

    inspection = proposal.inspection
    events = inspection.source_events if resolution.side == "source" else inspection.reply_events
    for event in events:
        if (
            event.predicate in resolution.evidence_spans
            and validate_span(event.predicate, unresolved) is not None
        ):
            return True
    for evaluation in inspection.evaluations:
        relation_heads = {span for span in evaluation.evidence_spans if span}
        if any(evidence in relation_heads for evidence in resolution.evidence_spans) and any(
            unresolved in span for span in evaluation.evidence_spans
        ):
            return True
    for shell in inspection.reply_shells:
        role_heads = {
            span
            for span in (
                shell.subject_span,
                shell.complement_span,
                shell.decoration_span,
                shell.repackaging_marker,
            )
            if span
        }
        if any(evidence in role_heads for evidence in resolution.evidence_spans) and any(
            unresolved in span for span in shell.evidence_spans
        ):
            return True
    return False


def _typed_relation_matches_category(
    resolution: SemanticCoverageResolution,
    proposal: SemanticProviderProposal,
) -> bool:
    inspection = proposal.inspection
    events = inspection.source_events if resolution.side == "source" else inspection.reply_events
    if resolution.category == "possible_harm":
        human_outcomes = {
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
        return any(
            event.event_type in human_outcomes
            and event.affected_entity is not None
            and event.affected_entity.entity_type in {"person", "group_of_people"}
            and event.explicit_outcome
            and _evidence_overlaps(resolution, event.evidence_spans)
            for event in events
        )
    if resolution.category == "possible_evaluation":
        return any(
            evaluation.evaluation_type != "unknown"
            and evaluation.polarity != "unknown"
            and (evaluation.target_entity is not None or evaluation.target_event_id is not None)
            and _evidence_overlaps(resolution, evaluation.evidence_spans)
            for evaluation in inspection.evaluations
        )
    if resolution.category == "possible_reduction":
        role_bound_operators = {
            "identity",
            "reduction",
            "decorated_identity",
            "repackaging",
            "reveal",
        }
        return any(
            shell.operator not in {"none", "unknown"}
            and (
                shell.operator not in role_bound_operators
                or (shell.subject_span is not None and shell.complement_span is not None)
            )
            and _evidence_overlaps(resolution, shell.evidence_spans)
            for shell in inspection.reply_shells
        )
    if resolution.category == "unknown_referent":
        typed_entities = {
            _entity_key(entity)
            for entity in (*inspection.source_entities, *inspection.reply_entities)
            if entity.entity_type != "unknown"
            and _evidence_overlaps(resolution, (entity.text_span,))
        }
        if not typed_entities:
            return False
        return any(
            any(
                entity is not None and _entity_key(entity) in typed_entities
                for entity in (event.actor, event.affected_entity, event.object)
            )
            for event in (*inspection.source_events, *inspection.reply_events)
        ) or any(
            evaluation.target_entity is not None
            and _entity_key(evaluation.target_entity) in typed_entities
            for evaluation in inspection.evaluations
        )
    if resolution.category == "unknown_predicate":
        return (
            any(_evidence_overlaps(resolution, event.evidence_spans) for event in events)
            or any(
                _evidence_overlaps(resolution, evaluation.evidence_spans)
                for evaluation in inspection.evaluations
            )
            or any(
                _evidence_overlaps(resolution, shell.evidence_spans)
                for shell in inspection.reply_shells
            )
        )
    return False


def _validate_resolution(
    resolution: SemanticCoverageResolution,
    *,
    deterministic_items: dict[str, SemanticUnresolvedItem],
    source_haystacks: tuple[str, ...],
    reply_haystacks: tuple[str, ...],
    proposal: SemanticProviderProposal,
) -> list[str]:
    errors: list[str] = []
    item = deterministic_items.get(resolution.item_id)
    if item is None:
        errors.append(f"coverage resolution uses stale item id: {resolution.item_id!r}")
    else:
        if resolution.side != item.side:
            errors.append(
                f"coverage resolution side changed for {resolution.item_id!r}: "
                f"{resolution.side!r} != {item.side!r}"
            )
        if resolution.clause_id != item.clause_id:
            errors.append(
                f"coverage resolution clause id changed for {resolution.item_id!r}: "
                f"{resolution.clause_id!r} != {item.clause_id!r}"
            )
        if resolution.clause_span != item.clause_span:
            errors.append(f"coverage resolution clause span changed for {resolution.item_id!r}")
        if resolution.unresolved_span != item.unresolved_span:
            errors.append(f"coverage resolution unresolved span changed for {resolution.item_id!r}")
        if resolution.category != item.category:
            errors.append(
                f"coverage resolution category changed for {resolution.item_id!r}: "
                f"{resolution.category!r} != {item.category!r}"
            )
    haystacks = source_haystacks if resolution.side == "source" else reply_haystacks
    if not _exact(resolution.clause_span, haystacks):
        errors.append(f"coverage resolution clause is not exact: {resolution.clause_span!r}")
    if resolution.unresolved_span not in resolution.clause_span:
        errors.append(
            f"coverage unresolved span {resolution.unresolved_span!r} is outside its clause"
        )
    for span in resolution.evidence_spans:
        if not _exact(span, haystacks):
            errors.append(f"coverage resolution evidence span is not exact: {span!r}")
    if resolution.disposition == "harmless":
        if resolution.category == "incomplete_proposition":
            errors.append("incomplete proposition cannot be resolved as harmless")
        if not resolution.reason.strip():
            errors.append("harmless coverage resolution requires a reason")
        if not resolution.evidence_spans:
            errors.append("harmless coverage resolution requires exact evidence spans")
    elif resolution.disposition == "typed_relation":
        if not resolution.evidence_spans:
            errors.append("typed coverage resolution requires exact evidence spans")
        elif not _evidence_covers_unresolved_role(resolution, proposal):
            errors.append(
                "typed coverage resolution evidence does not cover the complete unresolved role"
            )
        if not _typed_relation_matches_category(resolution, proposal):
            errors.append(
                "typed coverage resolution is not backed by a category-compatible proposed relation"
            )
    return errors


def validate_semantic_provider_proposal(
    deterministic: SemanticInspection,
    proposal: SemanticProviderProposal,
    *,
    source_clauses: tuple[str, ...],
    reply_clauses: tuple[str, ...],
) -> tuple[str, ...]:
    """Return validation errors; an empty tuple means the proposal is locally admissible."""

    inspection = proposal.inspection
    errors: list[str] = []
    errors.extend(_validate_entities(inspection.source_entities, source_clauses, label="source"))
    errors.extend(_validate_entities(inspection.reply_entities, reply_clauses, label="reply"))

    source_entity_keys = {
        _entity_key(item) for item in (*deterministic.source_entities, *inspection.source_entities)
    }
    reply_entity_keys = {
        _entity_key(item) for item in (*deterministic.reply_entities, *inspection.reply_entities)
    }
    all_entity_keys = source_entity_keys | reply_entity_keys
    for event in inspection.source_events:
        errors.extend(
            _validate_event(
                event,
                haystacks=source_clauses,
                entity_keys=source_entity_keys,
                label="source",
            )
        )
    for event in inspection.reply_events:
        errors.extend(
            _validate_event(
                event,
                haystacks=reply_clauses,
                entity_keys=reply_entity_keys,
                label="reply",
            )
        )
    provider_event_ids = [
        event.event_id for event in (*inspection.source_events, *inspection.reply_events)
    ]
    if len(provider_event_ids) != len(set(provider_event_ids)):
        errors.append("provider event ids must be unique")
    event_ids = {
        event.event_id
        for event in (
            *deterministic.source_events,
            *deterministic.reply_events,
            *inspection.source_events,
            *inspection.reply_events,
        )
    }
    for evaluation in inspection.evaluations:
        errors.extend(
            _validate_evaluation(
                evaluation,
                reply_haystacks=reply_clauses,
                entity_keys=all_entity_keys,
                event_ids=event_ids,
            )
        )
    for shell in inspection.reply_shells:
        errors.extend(_validate_shell(shell, reply_clauses))
    errors.extend(_validate_coverage(inspection.source_coverage, source_clauses, label="source"))
    errors.extend(_validate_coverage(inspection.reply_coverage, reply_clauses, label="reply"))
    if inspection.reply_communicative_function not in {"unknown", "other"}:
        if inspection.reply_communicative_function_confidence != "high":
            errors.append("provider communicative function requires high confidence")
        if not inspection.reply_communicative_evidence_spans:
            errors.append("provider communicative function requires exact evidence spans")
        for span in inspection.reply_communicative_evidence_spans:
            if not _exact(span, reply_clauses):
                errors.append(f"provider communicative-function span is not exact: {span!r}")
    all_haystacks = (*source_clauses, *reply_clauses)
    for span in inspection.unresolved_material:
        if not _exact(span, all_haystacks):
            errors.append(f"provider unresolved material span is not exact: {span!r}")

    deterministic_items = _coverage_items(deterministic)
    for resolution in (*proposal.coverage_resolutions, *proposal.remaining_unresolved_items):
        errors.extend(
            _validate_resolution(
                resolution,
                deterministic_items=deterministic_items,
                source_haystacks=source_clauses,
                reply_haystacks=reply_clauses,
                proposal=proposal,
            )
        )
    for item in proposal.remaining_unresolved_items:
        if item.disposition != "unresolved":
            errors.append("remaining_unresolved_items must use disposition='unresolved'")

    return tuple(dict.fromkeys(errors))
