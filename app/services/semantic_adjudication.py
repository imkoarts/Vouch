"""Application-owned decisions over the typed semantic intermediate representation.

Functions in this module never receive source or reply strings. They consume extracted semantic
objects plus non-semantic workflow flags, so lexical coverage cannot directly become policy.
"""

from __future__ import annotations

from typing import Literal

from app.schemas.content import (
    EvaluativeIronyInspection,
    HumanHarmInspection,
    HumorIntentInspection,
    HumorSafetyInspection,
    ReplyShellInspection,
    SemanticEvent,
    SemanticHumorProposal,
    SemanticInspection,
)

_HUMAN_TYPES = {"person", "group_of_people"}
_NON_HUMAN_TYPES = {
    "technical_system",
    "software_process",
    "product",
    "company_or_organization",
    "crypto_or_protocol",
    "business_or_financial_object",
    "legal_or_political_object",
    "abstract_concept",
}
_HUMAN_HARM_STATES = {
    "death",
    "serious_injury",
    "critical_condition",
    "hospitalization",
    "intensive_care",
    "permanent_injury",
    "non_survival",
}
_NON_HUMAN_FAILURE_STATES = {"technical_failure", "business_failure"}


def _human_harm_events(inspection: SemanticInspection) -> tuple[SemanticEvent, ...]:
    return tuple(
        event
        for event in inspection.source_events
        if event.explicit_outcome
        and event.assertion_state in {"asserted", "reported"}
        and event.event_type in _HUMAN_HARM_STATES
        and event.affected_entity is not None
        and event.affected_entity.entity_type in _HUMAN_TYPES
    )


def _non_human_failure_events(inspection: SemanticInspection) -> tuple[SemanticEvent, ...]:
    return tuple(
        event
        for event in inspection.source_events
        if event.event_type in _NON_HUMAN_FAILURE_STATES
        and event.affected_entity is not None
        and event.affected_entity.entity_type in _NON_HUMAN_TYPES
    )


def _context_event_type(
    event: SemanticEvent,
) -> Literal[
    "physical_crash",
    "blast",
    "fire",
    "medical",
    "incident",
    "outage",
    "technical",
    "business_or_political",
    "unknown",
]:
    object_name = event.object.canonical_name if event.object is not None else None
    if object_name in {"crash", "collision", "accident"}:
        return "physical_crash"
    if object_name in {"blast", "explosion"}:
        return "blast"
    if object_name == "fire":
        return "fire"
    if object_name == "incident":
        return "incident"
    if object_name == "outage":
        return "outage"
    if event.event_type in {
        "critical_condition",
        "hospitalization",
        "intensive_care",
        "permanent_injury",
    }:
        return "medical"
    if event.event_type == "technical_failure":
        return "technical"
    if event.event_type == "business_failure":
        return "business_or_political"
    return "unknown"


def adjudicate_human_harm(inspection: SemanticInspection) -> HumanHarmInspection:
    """Derive harm only from an explicit outcome relation and typed affected referent."""

    human_events = _human_harm_events(inspection)
    non_human_events = _non_human_failure_events(inspection)
    if human_events:
        priority = {
            "death": 0,
            "non_survival": 1,
            "critical_condition": 2,
            "intensive_care": 3,
            "hospitalization": 4,
            "permanent_injury": 5,
            "serious_injury": 6,
        }
        selected = sorted(human_events, key=lambda item: priority[item.event_type])[0]
        state: Literal[
            "death",
            "serious_injury",
            "critical_condition",
            "hospitalization",
            "intensive_care",
            "permanent_injury",
            "non_survival",
            "non_human_failure",
            "unknown",
        ] = "death" if selected.event_type == "non_survival" else selected.event_type  # type: ignore[assignment]
        evidence = tuple(
            dict.fromkeys(span for event in human_events for span in event.evidence_spans)
        )
        return HumanHarmInspection(
            harm_referent="human",
            harm_state=state,
            event_type=_context_event_type(selected),
            literal_or_metaphorical_usage="literal",
            literal_harm=True,
            metaphorical_or_technical_usage=bool(non_human_events),
            confidence="high",
            evidence_spans=evidence,
            reason="explicit outcome relation attached to a human referent",
        )
    if non_human_events:
        selected = non_human_events[0]
        evidence = tuple(
            dict.fromkeys(span for event in non_human_events for span in event.evidence_spans)
        )
        return HumanHarmInspection(
            harm_referent="non_human",
            harm_state="non_human_failure",
            event_type=_context_event_type(selected),
            literal_or_metaphorical_usage="metaphorical_or_technical",
            literal_harm=False,
            metaphorical_or_technical_usage=True,
            confidence="high",
            evidence_spans=evidence,
            reason="failure predicate attached to a typed non-human referent",
        )
    unknown_events = tuple(
        event
        for event in inspection.source_events
        if event.event_type == "unknown"
        and event.assertion_state in {"asserted", "reported"}
        and event.evidence_spans
    )
    possible_harm = tuple(
        span
        for coverage in inspection.source_coverage
        if "possible_harm" in coverage.unresolved_categories
        for span in coverage.unresolved_spans
    )
    human_context = any(entity.entity_type in _HUMAN_TYPES for entity in inspection.source_entities)
    unresolved = tuple(
        dict.fromkeys(
            (
                *(span for event in unknown_events for span in event.evidence_spans),
                *possible_harm,
            )
        )
    )
    if unresolved:
        return HumanHarmInspection(
            harm_referent="unknown",
            harm_state="unknown",
            event_type="medical" if human_context else "unknown",
            literal_or_metaphorical_usage="unknown",
            literal_harm=False,
            metaphorical_or_technical_usage=False,
            confidence="medium",
            evidence_spans=unresolved,
            reason="material outcome wording has an unresolved affected referent",
        )
    return HumanHarmInspection(confidence="high")


def adjudicate_evaluative_irony(inspection: SemanticInspection) -> EvaluativeIronyInspection:
    """Decide irony from evaluation relation, source polarity, and target alignment."""

    positive = tuple(item for item in inspection.evaluations if item.polarity == "positive")
    harmful_events = _human_harm_events(inspection)
    negative_events = tuple(
        event for event in inspection.source_events if event.polarity == "negative"
    )
    positive_events = tuple(
        event for event in inspection.source_events if event.polarity == "positive"
    )
    unresolved_possible_harm = any(
        "possible_harm" in coverage.unresolved_categories for coverage in inspection.source_coverage
    )
    if harmful_events:
        source_polarity: Literal["harmful", "negative", "positive", "mixed", "unknown"] = "harmful"
    elif negative_events and positive_events:
        source_polarity = "mixed"
    elif negative_events:
        source_polarity = "negative"
    elif unresolved_possible_harm:
        source_polarity = "harmful"
    elif positive_events:
        source_polarity = "positive"
    else:
        source_polarity = "unknown"
    relation = positive[0] if positive else None
    unresolved_evaluation = tuple(
        span
        for coverage in inspection.reply_coverage
        if "possible_evaluation" in coverage.unresolved_categories
        for span in coverage.unresolved_spans
    )
    aligned_ids = {event.event_id for event in negative_events}
    target_aligned = bool(relation is not None and relation.target_event_id in aligned_ids)
    literal = bool(relation is not None and relation.literal_reading_plausible)
    negative_context = source_polarity in {"harmful", "negative", "mixed"}
    implicit = bool(relation is not None and negative_context and target_aligned and not literal)
    if implicit:
        confidence: Literal["high", "medium", "low"] = "high"
    elif negative_context and ((relation is not None and not literal) or unresolved_evaluation):
        confidence = "medium"
    else:
        confidence = "low"
    subtype = (
        relation.relation_subtype
        if relation is not None
        else "uncertain"
        if unresolved_evaluation and negative_context
        else "none"
    )
    evaluation_polarity: Literal["positive", "rewarding", "neutral", "unknown"]
    if relation is None:
        evaluation_polarity = "unknown"
    elif relation.evaluation_type == "reward":
        evaluation_polarity = "rewarding"
    else:
        evaluation_polarity = "positive"
    return EvaluativeIronyInspection(
        evaluation_polarity=evaluation_polarity,
        evaluation_relation=subtype,
        source_event_polarity=source_polarity,
        evaluation_target=(
            relation.target_entity.text_span
            if relation is not None and relation.target_entity is not None
            else None
        ),
        target_source_alignment=target_aligned,
        positive_evaluation=bool(relation or unresolved_evaluation),
        negative_or_harmful_source_context=negative_context,
        evaluation_target_grounded=target_aligned,
        literal_positive_reading_plausible=literal,
        implicit_sarcasm=implicit,
        confidence=confidence,
        evidence_spans=(relation.evidence_spans if relation is not None else unresolved_evaluation),
    )


def adjudicate_humor_safety(inspection: SemanticInspection) -> HumorSafetyInspection:
    """Apply human-harm suitability policy from typed source and reply relations."""

    harm = adjudicate_human_harm(inspection)
    human_events = _human_harm_events(inspection)
    non_human_events = _non_human_failure_events(inspection)
    event_by_id = {event.event_id: event for event in inspection.source_events}
    categories: list[str] = []
    if human_events:
        categories.append("death_or_serious_injury")
    if any(event.event_type == "abuse" for event in inspection.source_events):
        categories.append("abuse_or_self_harm")
    if any(event.event_type == "self_harm" for event in inspection.source_events):
        categories.append("abuse_or_self_harm")
    if any(event.event_type == "personal_tragedy" for event in inspection.source_events):
        categories.append("personal_tragedy")
    if any(
        event in human_events
        and (
            (
                event.object is not None
                and event.object.canonical_name in {"earthquake", "flood", "disaster"}
            )
            or (
                event.affected_entity is not None
                and any(
                    term in (event.affected_entity.canonical_name or "")
                    for term in ("child", "family")
                )
            )
        )
        for event in inspection.source_events
    ):
        categories.append("personal_tragedy")
    ambiguous_event = any(
        event.event_type in {"failure", "damage"}
        and event.object is not None
        and event.object.entity_type == "physical_event"
        and not event.explicit_outcome
        for event in inspection.source_events
    )
    unresolved_possible_harm = any(
        "possible_harm" in coverage.unresolved_categories for coverage in inspection.source_coverage
    )
    ambiguous_harm = (
        (harm.harm_referent == "unknown" and bool(harm.evidence_spans))
        or unresolved_possible_harm
        or (ambiguous_event and not human_events and not non_human_events)
    )
    if ambiguous_harm:
        categories.append("ambiguous_human_harm")
    sensitive = bool(categories)
    targets_harm = any(
        evaluation.target_event_id is not None
        and evaluation.target_event_id in event_by_id
        and event_by_id[evaluation.target_event_id].event_type in _HUMAN_HARM_STATES
        for evaluation in inspection.evaluations
    ) or any(entity.entity_type in _HUMAN_TYPES for entity in inspection.reply_entities)
    public_target = any(
        entity.canonical_name
        in {"wording", "statement", "headline", "label", "name", "announcement"}
        for entity in inspection.reply_entities
    )
    self_deprecating = bool(
        inspection.reply_self_reference
        and not any(entity.entity_type in _HUMAN_TYPES for entity in inspection.reply_entities)
    )
    suitable = bool(
        not sensitive
        or self_deprecating
        or (public_target and not targets_harm and not ambiguous_harm)
    )
    return HumorSafetyInspection(
        sensitive_context=sensitive,
        sensitive_categories=tuple(dict.fromkeys(categories)),
        targets_victim_or_harm=targets_harm,
        targets_public_claim_or_system=public_target,
        self_deprecating=self_deprecating,
        suitable_for_humor=suitable,
        issue_codes=("HUMOR_ON_TRAGEDY",) if sensitive and not suitable else (),
        context_confidence="medium" if ambiguous_harm else "high",
        ambiguous_sensitive_term=ambiguous_harm,
        disambiguation_reason=harm.reason,
        harm_referent=harm.harm_referent,
        literal_harm=harm.literal_harm,
        metaphorical_or_technical_usage=harm.metaphorical_or_technical_usage,
    )


def _sensitive_source_with_incomplete_reply(inspection: SemanticInspection) -> bool:
    """Fail closed for one incomplete candidate on typed or unresolved human-harm sources."""

    typed_human_harm = any(
        event.event_type in _HUMAN_HARM_STATES
        and event.explicit_outcome
        and event.assertion_state in {"asserted", "reported"}
        and event.affected_entity is not None
        and event.affected_entity.entity_type in _HUMAN_TYPES
        for event in inspection.source_events
    )
    unresolved_possible_harm = any(
        "possible_harm" in coverage.unresolved_categories for coverage in inspection.source_coverage
    )
    if not (typed_human_harm or unresolved_possible_harm):
        return False
    material_categories = {
        "possible_harm",
        "possible_evaluation",
        "possible_reduction",
        "unknown_predicate",
        "unknown_referent",
        "incomplete_proposition",
    }
    reply_incomplete = any(
        any(category in material_categories for category in coverage.unresolved_categories)
        for coverage in inspection.reply_coverage
    )
    validated_literal = bool(
        inspection.reply_communicative_function
        in {"plain_observation", "direct_response", "qualification", "question"}
        and inspection.reply_communicative_function_confidence == "high"
        and inspection.reply_communicative_evidence_spans
    )
    return reply_incomplete and not validated_literal


def adjudicate_humor_intent(
    inspection: SemanticInspection,
    *,
    metadata_says_humor: bool,
    reaction_type: Literal[
        "literalization",
        "incongruity",
        "callback",
        "wordplay",
        "dry_reframe",
        "none",
        "uncertain",
    ],
    template_humor: bool,
) -> HumorIntentInspection:
    """Combine typed semantic irony with already-extracted non-evaluative humor functions."""

    irony = adjudicate_evaluative_irony(inspection)
    reaction_humor = reaction_type in {
        "literalization",
        "incongruity",
        "callback",
        "wordplay",
        "dry_reframe",
    }
    humorous = bool(irony.implicit_sarcasm or reaction_humor or template_humor)
    issue_codes: list[str] = []
    material_conflict = bool(inspection.extraction_conflicts)
    sensitive_incomplete_reply = _sensitive_source_with_incomplete_reply(inspection)
    if irony.implicit_sarcasm:
        detected: Literal[
            "literalization",
            "incongruity",
            "callback",
            "wordplay",
            "dry_reframe",
            "template_humor",
            "sarcasm",
            "irony",
            "none",
            "uncertain",
        ] = "sarcasm"
    elif reaction_humor:
        detected = reaction_type
    elif template_humor:
        detected = "template_humor"
    elif (
        (irony.confidence == "medium" and irony.negative_or_harmful_source_context)
        or material_conflict
        or sensitive_incomplete_reply
    ):
        detected = "uncertain"
        issue_codes.append("HUMOR_INTENT_UNCERTAIN")
    elif reaction_type == "uncertain":
        detected = "uncertain"
    else:
        detected = "none"
    safety = adjudicate_humor_safety(inspection)
    uncertain_sensitive = bool(
        detected == "uncertain"
        and safety.sensitive_context
        and (irony.positive_evaluation or sensitive_incomplete_reply)
    )
    conflict_sensitive = bool(material_conflict and safety.sensitive_context)
    if uncertain_sensitive and "HUMOR_INTENT_UNCERTAIN" not in issue_codes:
        issue_codes.append("HUMOR_INTENT_UNCERTAIN")
    if conflict_sensitive:
        issue_codes.append("HUMOR_INTENT_CONFLICT")
    proposal_intent = (
        "sarcasm"
        if detected == "sarcasm"
        else detected
        if detected in {"incongruity", "callback", "wordplay", "dry_reframe", "uncertain"}
        else "dry_reframe"
        if detected in {"literalization", "template_humor"}
        else "none"
    )
    proposal = SemanticHumorProposal(
        intent=proposal_intent,
        humorous_treatment=humorous,
        confidence="high" if humorous else "medium" if uncertain_sensitive else "low",
        evidence_spans=irony.evidence_spans,
    )
    return HumorIntentInspection(
        metadata_says_humor=metadata_says_humor,
        text_appears_humorous=humorous,
        detected_reaction_type=detected,
        humor_safety_required=bool(
            metadata_says_humor or humorous or uncertain_sensitive or conflict_sensitive
        ),
        extraction_conflict=bool(metadata_says_humor != humorous or material_conflict),
        evaluative_irony=irony,
        issue_codes=tuple(issue_codes),
        proposal=proposal,
    )


def adjudicate_reply_shell(inspection: SemanticInspection) -> ReplyShellInspection:
    """Project the canonical semantic shell into the established public inspection contract."""

    if any("reply_shell" in item for item in inspection.extraction_conflicts):
        evidence = tuple(
            dict.fromkeys(
                span for coverage in inspection.reply_coverage for span in coverage.unresolved_spans
            )
        )
        return ReplyShellInspection(
            operator="uncertain",
            generic_reusability=True,
            evidence_spans=evidence,
        )
    if not inspection.reply_shells:
        unresolved = tuple(
            span
            for coverage in inspection.reply_coverage
            if "possible_reduction" in coverage.unresolved_categories
            for span in coverage.unresolved_spans
        )
        if unresolved:
            return ReplyShellInspection(
                operator="uncertain",
                generic_reusability=True,
                evidence_spans=tuple(dict.fromkeys(unresolved)),
            )
        return ReplyShellInspection()
    shell = inspection.reply_shells[0]
    operator = "decorated_identity" if shell.operator == "repackaging" else shell.operator
    return ReplyShellInspection(
        operator=operator,
        subject_span=shell.subject_span,
        complement_span=shell.complement_span,
        decoration_span=shell.decoration_span,
        repackaging_marker=shell.repackaging_marker,
        opener_family=shell.opener_family,
        source_specific=False,
        generic_reusability=shell.generic_reusability,
        evidence_spans=shell.evidence_spans,
    )
