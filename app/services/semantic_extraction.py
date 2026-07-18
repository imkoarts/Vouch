"""Public compositional semantic-extraction API.

The implementation is split into lexical normalization, clause framing, relation binding, and
coverage accounting in :mod:`app.services.semantic_composition`.  This compatibility module keeps
stable call sites while preventing application policy from reading surface text.
"""

from __future__ import annotations

from app.schemas.content import (
    EvaluationRelation,
    SemanticEntity,
    SemanticEvent,
    SemanticInspection,
    SemanticReplyShell,
)
from app.services.semantic_composition import (
    bind_evaluation_relations,
    bind_event_relations,
    bind_reply_shell_relations,
    build_clause_frame,
    build_coverage,
    extract_compositional_semantics,
    extract_entity_candidates,
    extract_lexical_atoms,
    segment_clauses,
)


def extract_source_semantics(
    source_text: str,
) -> tuple[tuple[SemanticEntity, ...], tuple[SemanticEvent, ...], tuple[str, ...]]:
    """Extract source entities/events and unresolved material through compositional stages."""

    inspection = extract_compositional_semantics(source_text)
    return inspection.source_entities, inspection.source_events, inspection.unresolved_material


def extract_evaluations(
    reply_text: str,
    source_entities: tuple[SemanticEntity, ...],
    source_events: tuple[SemanticEvent, ...],
) -> tuple[EvaluationRelation, ...]:
    """Bind reply evaluations to already-extracted source entities and events."""

    relations, _ = bind_evaluation_relations(
        segment_clauses(reply_text),
        source_entities,
        source_events,
    )
    return relations


def extract_reply_shells(reply_text: str) -> tuple[SemanticReplyShell, ...]:
    """Extract canonical reply-shell relations from compositional clause frames."""

    shells, _ = bind_reply_shell_relations(segment_clauses(reply_text))
    return shells


def extract_semantic_inspection(source_text: str, reply_text: str = "") -> SemanticInspection:
    """Extract the canonical typed semantic IR and per-clause coverage accounting."""

    return extract_compositional_semantics(source_text, reply_text)


__all__ = [
    "bind_event_relations",
    "build_clause_frame",
    "build_coverage",
    "extract_entity_candidates",
    "extract_evaluations",
    "extract_lexical_atoms",
    "extract_reply_shells",
    "extract_semantic_inspection",
    "extract_source_semantics",
]
