"""Deterministic/provider reconciliation for typed semantic extraction proposals."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable

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
)
from app.services.semantic_lexicon import normalize_surface


def _spans_valid(spans: tuple[str, ...], haystacks: tuple[str, ...]) -> bool:
    """Exact span verification is the only raw-text operation in reconciliation."""

    return bool(spans) and all(span and any(span in value for value in haystacks) for span in spans)


def _entity_key(item: SemanticEntity) -> tuple[str, str, str]:
    return item.clause_id or "unscoped", normalize_surface(item.text_span), item.entity_type


def _event_key(item: SemanticEvent) -> tuple[str, str, str, str]:
    affected = item.affected_entity
    return (
        item.clause_id,
        item.event_type,
        normalize_surface(item.predicate),
        normalize_surface(affected.text_span) if affected is not None else "",
    )


def _evaluation_key(item: EvaluationRelation) -> tuple[str, str, str, str]:
    target = item.target_entity
    return (
        item.clause_id,
        item.evaluation_type,
        item.relation_subtype,
        normalize_surface(target.text_span) if target is not None else "",
    )


def _shell_key(item: SemanticReplyShell) -> tuple[str, str, bool, bool, bool]:
    return (
        item.clause_id,
        item.operator,
        item.subject_span is not None,
        item.complement_span is not None,
        item.decoration_span is not None,
    )


def _coverage_key(item: SemanticCoverage) -> tuple[str, str]:
    return item.coverage_id, item.clause_id


def _resolution_key(item: SemanticCoverageResolution) -> str:
    return item.item_id


def _merge_unique[T](
    deterministic: tuple[T, ...],
    provider: Iterable[T],
    *,
    key: Callable[[T], Hashable],
) -> tuple[T, ...]:
    result = list(deterministic)
    seen = {key(item) for item in result}
    for item in provider:
        item_key = key(item)
        if item_key in seen:
            continue
        seen.add(item_key)
        result.append(item)
    return tuple(result)


def _merge_shells_with_refinement(
    deterministic: tuple[SemanticReplyShell, ...],
    provider: tuple[SemanticReplyShell, ...],
    resolutions: tuple[SemanticCoverageResolution, ...],
) -> tuple[SemanticReplyShell, ...]:
    resolved_reply_spans = {
        span
        for item in resolutions
        if item.side == "reply"
        and item.category == "possible_reduction"
        and item.disposition == "typed_relation"
        for span in item.evidence_spans
    }
    refined: list[SemanticReplyShell] = []
    suppressed_local: set[int] = set()
    for proposed in provider:
        for index, local in enumerate(deterministic):
            overlap = _span_sets_overlap(local.evidence_spans, proposed.evidence_spans)
            resolution_overlap = any(
                _span_sets_overlap((span,), local.evidence_spans)
                or _span_sets_overlap((span,), proposed.evidence_spans)
                for span in resolved_reply_spans
            )
            if (
                overlap
                and resolution_overlap
                and local.operator in {"identity", "reduction", "unknown"}
                and proposed.operator
                in {"decorated_identity", "repackaging", "reveal", "reduction"}
            ):
                suppressed_local.add(index)
        refined.append(proposed)
    remaining = tuple(
        item for index, item in enumerate(deterministic) if index not in suppressed_local
    )
    return _merge_unique(tuple(refined), remaining, key=_shell_key)


def reconcile_semantic_inspections(
    deterministic: SemanticInspection,
    provider: SemanticInspection | None,
    *,
    source_text: str,
    reply_text: str,
) -> SemanticInspection:
    """Merge valid provider extraction additions without allowing erasure of local relations."""

    if provider is None:
        return deterministic
    conflicts = list((*deterministic.extraction_conflicts, *provider.extraction_conflicts))
    source_haystack = (source_text,)
    reply_haystack = (reply_text,)

    provider_source_entities = tuple(
        item
        for item in provider.source_entities
        if item.text_span and item.text_span in source_text
    )
    provider_reply_entities = tuple(
        item for item in provider.reply_entities if item.text_span and item.text_span in reply_text
    )
    provider_events: list[SemanticEvent] = []
    for event in provider.source_events:
        if not _spans_valid(event.evidence_spans, source_haystack):
            conflicts.append("provider_source_event_evidence_mismatch")
            continue
        provider_events.append(event)
    provider_reply_events: list[SemanticEvent] = []
    for event in provider.reply_events:
        if not _spans_valid(event.evidence_spans, reply_haystack):
            conflicts.append("provider_reply_event_evidence_mismatch")
            continue
        provider_reply_events.append(event)
    provider_evaluations: list[EvaluationRelation] = []
    for evaluation in provider.evaluations:
        if not _spans_valid(evaluation.evidence_spans, reply_haystack):
            conflicts.append("provider_evaluation_evidence_mismatch")
            continue
        provider_evaluations.append(evaluation)
    provider_shells: list[SemanticReplyShell] = []
    for shell in provider.reply_shells:
        if not _spans_valid(shell.evidence_spans, reply_haystack):
            conflicts.append("provider_reply_shell_evidence_mismatch")
            continue
        provider_shells.append(shell)
    provider_source_coverage = tuple(
        coverage
        for coverage in provider.source_coverage
        if coverage.clause_span in source_text
        and all(
            span in coverage.clause_span
            for span in (*coverage.consumed_spans, *coverage.unresolved_spans)
        )
    )
    provider_reply_coverage = tuple(
        coverage
        for coverage in provider.reply_coverage
        if coverage.clause_span in reply_text
        and all(
            span in coverage.clause_span
            for span in (*coverage.consumed_spans, *coverage.unresolved_spans)
        )
    )

    # A provider's absence or `none` proposal cannot erase deterministic extraction. Material
    # disagreement is retained as uncertainty for candidate-level fail-closed adjudication.
    if deterministic.evaluations and provider.evaluations and not provider_evaluations:
        conflicts.append("provider_evaluation_conflict")
    if deterministic.reply_shells and provider.reply_shells and not provider_shells:
        conflicts.append("provider_reply_shell_conflict")

    return SemanticInspection(
        source_entities=_merge_unique(
            deterministic.source_entities,
            provider_source_entities,
            key=_entity_key,
        ),
        predicate_scopes=deterministic.predicate_scopes,
        source_events=_merge_unique(
            deterministic.source_events,
            provider_events,
            key=_event_key,
        ),
        reply_entities=_merge_unique(
            deterministic.reply_entities,
            provider_reply_entities,
            key=_entity_key,
        ),
        reply_events=_merge_unique(
            deterministic.reply_events,
            provider_reply_events,
            key=_event_key,
        ),
        evaluations=_merge_unique(
            deterministic.evaluations,
            provider_evaluations,
            key=_evaluation_key,
        ),
        reply_shells=_merge_unique(
            deterministic.reply_shells,
            provider_shells,
            key=_shell_key,
        ),
        result_state_materiality=deterministic.result_state_materiality,
        human_function_impacts=deterministic.human_function_impacts,
        # Proposition completeness is application-owned deterministic extraction.  A provider may
        # validate a communicative function, but it cannot manufacture a proposition frame.
        reply_propositions=deterministic.reply_propositions,
        literal_proposition_frames=deterministic.literal_proposition_frames,
        reply_semantic_completeness=deterministic.reply_semantic_completeness,
        source_coverage=_merge_unique(
            deterministic.source_coverage,
            provider_source_coverage,
            key=_coverage_key,
        ),
        reply_coverage=_merge_unique(
            deterministic.reply_coverage,
            provider_reply_coverage,
            key=_coverage_key,
        ),
        reply_communicative_function=(
            provider.reply_communicative_function
            if deterministic.reply_communicative_function in {"unknown", "other"}
            and provider.reply_communicative_function_confidence == "high"
            else deterministic.reply_communicative_function
        ),
        reply_communicative_function_confidence=(
            provider.reply_communicative_function_confidence
            if deterministic.reply_communicative_function in {"unknown", "other"}
            and provider.reply_communicative_function_confidence == "high"
            else deterministic.reply_communicative_function_confidence
        ),
        reply_communicative_evidence_spans=(
            provider.reply_communicative_evidence_spans
            if deterministic.reply_communicative_function in {"unknown", "other"}
            and provider.reply_communicative_function_confidence == "high"
            else deterministic.reply_communicative_evidence_spans
        ),
        reply_self_reference=(deterministic.reply_self_reference or provider.reply_self_reference),
        extraction_conflicts=tuple(dict.fromkeys(conflicts)),
        unresolved_material=tuple(
            dict.fromkeys((*deterministic.unresolved_material, *provider.unresolved_material))
        ),
    )


def _span_sets_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return any(
        left_span in right_span or right_span in left_span
        for left_span in left
        for right_span in right
    )


def _events_conflict(left: SemanticEvent, right: SemanticEvent) -> bool:
    if not _span_sets_overlap(left.evidence_spans, right.evidence_spans):
        return False
    if left.event_type == right.event_type or "unknown" in {left.event_type, right.event_type}:
        return False
    material = {
        "death",
        "serious_injury",
        "critical_condition",
        "hospitalization",
        "intensive_care",
        "permanent_injury",
        "non_survival",
        "technical_failure",
        "business_failure",
    }
    return left.event_type in material or right.event_type in material


def _evaluations_conflict(left: EvaluationRelation, right: EvaluationRelation) -> bool:
    if not _span_sets_overlap(left.evidence_spans, right.evidence_spans):
        return False
    return left.polarity != right.polarity or left.evaluation_type != right.evaluation_type


def _shells_conflict(left: SemanticReplyShell, right: SemanticReplyShell) -> bool:
    if not _span_sets_overlap(left.evidence_spans, right.evidence_spans):
        return False
    compatible = {
        ("decorated_identity", "repackaging"),
        ("repackaging", "decorated_identity"),
        ("identity", "decorated_identity"),
        ("decorated_identity", "identity"),
    }
    return left.operator != right.operator and (left.operator, right.operator) not in compatible


def _apply_resolutions(
    coverage_items: tuple[SemanticCoverage, ...],
    resolutions: tuple[SemanticCoverageResolution, ...],
    *,
    side: str,
    blocked_keys: set[str],
) -> tuple[SemanticCoverage, ...]:
    by_id = {
        item.item_id: item
        for item in resolutions
        if item.side == side and item.disposition in {"typed_relation", "harmless"}
    }
    output: list[SemanticCoverage] = []
    for coverage in coverage_items:
        remaining: list[SemanticUnresolvedItem] = []
        consumed = list(coverage.consumed_spans)
        for item in coverage.unresolved_items:
            resolution = by_id.get(item.item_id)
            if resolution is None or item.item_id in blocked_keys:
                remaining.append(item)
                continue
            consumed.extend(resolution.evidence_spans)
        # Coverage produced before request-local IDs is intentionally not silently resolved.
        if not coverage.unresolved_items and coverage.unresolved_spans:
            output.append(coverage)
            continue
        output.append(
            SemanticCoverage(
                coverage_id=coverage.coverage_id,
                clause_id=coverage.clause_id,
                clause_span=coverage.clause_span,
                consumed_spans=tuple(dict.fromkeys(consumed)),
                unresolved_items=tuple(remaining),
                unresolved_spans=tuple(dict.fromkeys(item.unresolved_span for item in remaining)),
                unresolved_categories=tuple(dict.fromkeys(item.category for item in remaining)),
                complete=not remaining,
            )
        )
    return tuple(output)


def reconcile_semantic_provider_proposal(
    deterministic: SemanticInspection,
    proposal: SemanticProviderProposal,
) -> SemanticInspection:
    """Resolve deterministic uncertainty with a validated provider proposal.

    Validation occurs before this function.  Reconciliation preserves every deterministic
    high-confidence relation, adds compatible provider relations, and removes unresolved coverage
    only when an explicit validated resolution names the exact deterministic item.
    """

    provider = proposal.inspection
    conflicts = list((*deterministic.extraction_conflicts, *provider.extraction_conflicts))
    blocked_resolution_keys: set[str] = set()

    for local_event in deterministic.source_events:
        for provider_event in provider.source_events:
            if _events_conflict(local_event, provider_event):
                conflicts.append(
                    f"provider_event_conflict:{local_event.event_id}:{provider_event.event_id}"
                )
                for resolution in proposal.coverage_resolutions:
                    if resolution.side == "source" and (
                        _span_sets_overlap(resolution.evidence_spans, local_event.evidence_spans)
                        or _span_sets_overlap(
                            resolution.evidence_spans, provider_event.evidence_spans
                        )
                    ):
                        blocked_resolution_keys.add(_resolution_key(resolution))
    for local_evaluation in deterministic.evaluations:
        for provider_evaluation in provider.evaluations:
            if _evaluations_conflict(local_evaluation, provider_evaluation):
                conflicts.append("provider_evaluation_conflict")
                for resolution in proposal.coverage_resolutions:
                    if resolution.side == "reply" and (
                        _span_sets_overlap(
                            resolution.evidence_spans, local_evaluation.evidence_spans
                        )
                        or _span_sets_overlap(
                            resolution.evidence_spans, provider_evaluation.evidence_spans
                        )
                    ):
                        blocked_resolution_keys.add(_resolution_key(resolution))
    for local_shell in deterministic.reply_shells:
        for provider_shell in provider.reply_shells:
            if _shells_conflict(local_shell, provider_shell):
                conflicts.append("provider_reply_shell_conflict")
                for resolution in proposal.coverage_resolutions:
                    if resolution.side == "reply" and (
                        _span_sets_overlap(resolution.evidence_spans, local_shell.evidence_spans)
                        or _span_sets_overlap(
                            resolution.evidence_spans, provider_shell.evidence_spans
                        )
                    ):
                        blocked_resolution_keys.add(_resolution_key(resolution))

    source_coverage = _apply_resolutions(
        deterministic.source_coverage,
        proposal.coverage_resolutions,
        side="source",
        blocked_keys=blocked_resolution_keys,
    )
    reply_coverage = _apply_resolutions(
        deterministic.reply_coverage,
        proposal.coverage_resolutions,
        side="reply",
        blocked_keys=blocked_resolution_keys,
    )

    unresolved = tuple(
        dict.fromkeys(
            (
                *(
                    span
                    for item in (*source_coverage, *reply_coverage)
                    for span in item.unresolved_spans
                ),
                *(
                    item.unresolved_span
                    for item in proposal.remaining_unresolved_items
                    if item.disposition == "unresolved"
                ),
                *provider.unresolved_material,
            )
        )
    )
    return SemanticInspection(
        source_entities=_merge_unique(
            deterministic.source_entities,
            provider.source_entities,
            key=_entity_key,
        ),
        predicate_scopes=deterministic.predicate_scopes,
        source_events=_merge_unique(
            deterministic.source_events,
            provider.source_events,
            key=_event_key,
        ),
        reply_entities=_merge_unique(
            deterministic.reply_entities,
            provider.reply_entities,
            key=_entity_key,
        ),
        reply_events=_merge_unique(
            deterministic.reply_events,
            provider.reply_events,
            key=_event_key,
        ),
        evaluations=_merge_unique(
            deterministic.evaluations,
            provider.evaluations,
            key=_evaluation_key,
        ),
        reply_shells=_merge_shells_with_refinement(
            deterministic.reply_shells,
            provider.reply_shells,
            proposal.coverage_resolutions,
        ),
        result_state_materiality=deterministic.result_state_materiality,
        human_function_impacts=deterministic.human_function_impacts,
        # Proposition completeness is application-owned deterministic extraction.  A provider may
        # validate a communicative function, but it cannot manufacture a proposition frame.
        reply_propositions=deterministic.reply_propositions,
        literal_proposition_frames=deterministic.literal_proposition_frames,
        reply_semantic_completeness=deterministic.reply_semantic_completeness,
        source_coverage=source_coverage,
        reply_coverage=reply_coverage,
        reply_communicative_function=(
            provider.reply_communicative_function
            if deterministic.reply_communicative_function in {"unknown", "other"}
            and provider.reply_communicative_function_confidence == "high"
            else deterministic.reply_communicative_function
        ),
        reply_communicative_function_confidence=(
            provider.reply_communicative_function_confidence
            if deterministic.reply_communicative_function in {"unknown", "other"}
            and provider.reply_communicative_function_confidence == "high"
            else deterministic.reply_communicative_function_confidence
        ),
        reply_communicative_evidence_spans=(
            provider.reply_communicative_evidence_spans
            if deterministic.reply_communicative_function in {"unknown", "other"}
            and provider.reply_communicative_function_confidence == "high"
            else deterministic.reply_communicative_evidence_spans
        ),
        reply_self_reference=deterministic.reply_self_reference or provider.reply_self_reference,
        extraction_conflicts=tuple(dict.fromkeys(conflicts)),
        unresolved_material=unresolved,
    )
