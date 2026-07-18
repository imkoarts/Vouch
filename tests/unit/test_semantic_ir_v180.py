from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.content import SemanticInspection
from app.services.semantic_adjudication import (
    adjudicate_evaluative_irony,
    adjudicate_human_harm,
    adjudicate_humor_safety,
    adjudicate_reply_shell,
)
from app.services.semantic_extraction import extract_semantic_inspection
from app.services.semantic_reconciliation import reconcile_semantic_inspections

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "semantic_ir_visible_v180.json"


def _cases() -> tuple[dict[str, str], ...]:
    return tuple(json.loads(_FIXTURE.read_text(encoding="utf-8"))["cases"])


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["id"])
def test_visible_semantic_ir_cases(case: dict[str, str]) -> None:
    inspection = extract_semantic_inspection(case["source"], case["reply"])
    harm = adjudicate_human_harm(inspection)
    irony = adjudicate_evaluative_irony(inspection)
    shell = adjudicate_reply_shell(inspection)

    match case["expected"]:
        case "sarcasm":
            assert irony.implicit_sarcasm
            assert irony.target_source_alignment
            assert not irony.literal_positive_reading_plausible
            assert not adjudicate_humor_safety(inspection).suitable_for_humor
        case "literal_praise":
            assert irony.positive_evaluation
            assert irony.literal_positive_reading_plausible
            assert not irony.implicit_sarcasm
        case "human_death":
            assert harm.harm_referent == "human"
            assert harm.harm_state == "death"
        case "critical_condition":
            assert harm.harm_referent == "human"
            assert harm.harm_state == "critical_condition"
        case "non_human_failure":
            assert harm.harm_referent == "non_human"
            assert harm.harm_state == "non_human_failure"
        case "mixed_human_harm":
            assert harm.harm_referent == "human"
            assert harm.literal_harm
            assert harm.metaphorical_or_technical_usage
        case "decorated_identity":
            assert shell.operator == "decorated_identity"
            assert shell.subject_span and shell.complement_span
        case "no_shell":
            assert shell.operator == "none"
        case _:
            raise AssertionError(f"unknown fixture expectation: {case['expected']}")


def test_adjudication_accepts_typed_ir_without_raw_text() -> None:
    inspection = extract_semantic_inspection(
        "A hospital reported ten deaths after the outage.",
        "stellar execution from the outage",
    )

    assert adjudicate_human_harm(inspection).literal_harm
    assert adjudicate_evaluative_irony(inspection).implicit_sarcasm
    assert not adjudicate_humor_safety(inspection).suitable_for_humor


def test_provider_none_cannot_erase_deterministic_relations() -> None:
    source = "A hospital reported ten deaths after the outage."
    reply = "stellar execution from the outage"
    deterministic = extract_semantic_inspection(source, reply)
    provider = SemanticInspection()

    canonical = reconcile_semantic_inspections(
        deterministic,
        provider,
        source_text=source,
        reply_text=reply,
    )

    assert canonical.source_events == deterministic.source_events
    assert canonical.evaluations == deterministic.evaluations
    assert adjudicate_evaluative_irony(canonical).implicit_sarcasm


def test_exact_evidence_spans_survive_reconciliation() -> None:
    source = "Two patients died during the outage."
    reply = "excellent work from the outage"
    deterministic = extract_semantic_inspection(source, reply)

    canonical = reconcile_semantic_inspections(
        deterministic,
        deterministic,
        source_text=source,
        reply_text=reply,
    )

    assert all(span in source for event in canonical.source_events for span in event.evidence_spans)
    assert all(span in reply for item in canonical.evaluations for span in item.evidence_spans)


def test_provider_ir_can_add_unseen_evaluation_relation_without_owning_policy() -> None:
    from app.schemas.content import EvaluationRelation, SemanticEntity

    source = "Two patients died after the blackout."
    reply = "an immaculate contribution from the blackout"
    deterministic = extract_semantic_inspection(source, reply)
    event_id = deterministic.source_events[0].event_id
    provider = SemanticInspection(
        evaluations=(
            EvaluationRelation(
                target_entity=SemanticEntity(
                    text_span="the blackout",
                    canonical_name="blackout",
                    entity_type="physical_event",
                    confidence="high",
                ),
                target_event_id=event_id,
                evaluation_type="praise",
                relation_subtype="praise",
                polarity="positive",
                literal_reading_plausible=False,
                confidence="high",
                evidence_spans=(reply,),
            ),
        )
    )

    canonical = reconcile_semantic_inspections(
        deterministic,
        provider,
        source_text=source,
        reply_text=reply,
    )

    assert adjudicate_evaluative_irony(canonical).implicit_sarcasm
    assert not adjudicate_humor_safety(canonical).suitable_for_humor
