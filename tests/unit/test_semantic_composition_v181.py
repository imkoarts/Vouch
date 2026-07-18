from __future__ import annotations

import pytest

from app.services.semantic_adjudication import (
    adjudicate_evaluative_irony,
    adjudicate_human_harm,
    adjudicate_reply_shell,
)
from app.services.semantic_extraction import (
    bind_event_relations,
    build_clause_frame,
    extract_lexical_atoms,
    extract_semantic_inspection,
    segment_clauses,
)


def test_extraction_stages_are_independently_testable() -> None:
    clause = "Three workers were killed in the blast."
    atoms = extract_lexical_atoms(clause)
    frame = build_clause_frame(clause)
    events, candidates = bind_event_relations((frame,))

    assert {atom.category for atom in atoms} >= {
        "human_head",
        "harm_predicate",
        "physical_event_head",
    }
    assert frame.subject_span == "Three workers"
    assert frame.predicate_span == "killed"
    assert events[0].event_type == "death"
    assert events[0].affected_entity is not None
    assert events[0].affected_entity.entity_type == "group_of_people"
    assert candidates[0].predicate_span == "killed"


@pytest.mark.parametrize(
    ("source", "state"),
    (
        ("Three workers died in the blast.", "death"),
        ("Three workers were killed in the blast.", "death"),
        ("A worker was hospitalized after the blast.", "hospitalization"),
        ("A worker remains in intensive care.", "intensive_care"),
        ("A worker was permanently injured in the crash.", "permanent_injury"),
        ("A worker was left in critical condition.", "critical_condition"),
        ("Three workers did not survive the crash.", "death"),
        ("Ten lives were lost after the crash.", "death"),
    ),
)
def test_human_outcome_relation_matrix(source: str, state: str) -> None:
    inspection = extract_semantic_inspection(source)
    harm = adjudicate_human_harm(inspection)

    assert harm.harm_referent == "human"
    assert harm.harm_state == state
    assert harm.literal_harm
    assert harm.evidence_spans
    assert all(span in source for span in harm.evidence_spans)


@pytest.mark.parametrize(
    "source",
    (
        "No workers died in the blast.",
        "Workers may have died in the blast.",
        "The company plans for workers to be hospitalized.",
        "Officials feared workers had died in the blast.",
        "If workers died, the inquiry would open.",
        "Compared with the prior incident, workers died less often.",
    ),
)
def test_non_asserted_harm_matrix_is_not_established(source: str) -> None:
    inspection = extract_semantic_inspection(source)
    harm = adjudicate_human_harm(inspection)

    assert not harm.literal_harm
    assert not any(event.explicit_outcome for event in inspection.source_events)


@pytest.mark.parametrize(
    ("source", "entity_type"),
    (
        ("The pipeline died overnight.", "software_process"),
        ("The database died during migration.", "technical_system"),
        ("The product died after launch.", "product"),
        ("The token died after liquidity vanished.", "crypto_or_protocol"),
        ("The deal was killed in committee.", "business_or_financial_object"),
        ("The bill was killed in committee.", "legal_or_political_object"),
    ),
)
def test_non_human_failure_relation_matrix(source: str, entity_type: str) -> None:
    inspection = extract_semantic_inspection(source)
    harm = adjudicate_human_harm(inspection)

    assert harm.harm_referent == "non_human"
    assert harm.harm_state == "non_human_failure"
    assert not harm.literal_harm
    assert any(
        event.affected_entity is not None and event.affected_entity.entity_type == entity_type
        for event in inspection.source_events
    )


@pytest.mark.parametrize(
    ("source", "event_type"),
    (
        ("The blast displaced three workers.", "displacement"),
        ("Officials evacuated three workers after the fire.", "evacuation"),
        ("The fire damaged three homes.", "damage"),
    ),
)
def test_non_fatal_outcomes_remain_distinct(source: str, event_type: str) -> None:
    inspection = extract_semantic_inspection(source)
    harm = adjudicate_human_harm(inspection)

    assert any(event.event_type == event_type for event in inspection.source_events)
    assert not harm.literal_harm


@pytest.mark.parametrize(
    "source",
    (
        "Three workers watched the explosion from a safe room.",
        "A patient described the outage.",
        "Passengers stood near the crash site.",
        "A nurse heard the blast.",
    ),
)
def test_event_proximity_does_not_construct_harm(source: str) -> None:
    harm = adjudicate_human_harm(extract_semantic_inspection(source))

    assert not harm.literal_harm
    assert harm.harm_referent != "human"


@pytest.mark.parametrize(
    ("reply", "relation"),
    (
        ("excellent work from the outage", "praise"),
        ("applause for the outage", "celebration"),
        ("five stars for the outage", "rating"),
        ("the outage deserves a medal", "reward"),
        ("so proud of the grid", "admiration"),
        ("the outage nailed it", "success_attribution"),
        ("exactly what the hospital needed", "beneficiary"),
    ),
)
def test_evaluation_relation_matrix(reply: str, relation: str) -> None:
    inspection = extract_semantic_inspection(
        "A hospital reported ten deaths after the outage.",
        reply,
    )
    irony = adjudicate_evaluative_irony(inspection)

    assert irony.evaluation_relation == relation
    assert irony.implicit_sarcasm
    assert irony.target_source_alignment


def test_same_evaluation_on_positive_event_is_literal() -> None:
    inspection = extract_semantic_inspection(
        "The hospital restored power and completed recovery safely.",
        "excellent work from the response team",
    )
    irony = adjudicate_evaluative_irony(inspection)

    assert irony.positive_evaluation
    assert irony.literal_positive_reading_plausible
    assert not irony.implicit_sarcasm


@pytest.mark.parametrize(
    ("reply", "operator"),
    (
        ("onboarding is just paperwork", "reduction"),
        ("onboarding is glorified paperwork", "decorated_identity"),
        ("onboarding is paperwork with better branding", "decorated_identity"),
        ("turns out onboarding was paperwork all along", "reveal"),
        ("nothing says simplicity like six approval screens", "inverse_praise"),
    ),
)
def test_reply_shell_relation_matrix(reply: str, operator: str) -> None:
    shell = adjudicate_reply_shell(extract_semantic_inspection("", reply))

    assert shell.operator == operator
    assert shell.subject_span
    assert shell.complement_span


@pytest.mark.parametrize(
    "reply",
    (
        "a triangle is a polygon with three sides",
        "revenue equals price plus tax",
        'the report defines "outage" as a service interruption',
        "the bill is a proposed law awaiting a vote",
        "the evals are probably harder to move than the API integration",
        "the social manager is good",
        "the service is down",
        "the team is ready",
    ),
)
def test_literal_controls_are_not_reusable_shells(reply: str) -> None:
    shell = adjudicate_reply_shell(extract_semantic_inspection("", reply))

    assert shell.operator == "none"


def test_clause_segmentation_preserves_material_clause_count() -> None:
    frames = segment_clauses(
        "The server died, but two patients survived. The hospital restored power."
    )

    assert len(frames) == 3
    assert tuple(frame.clause_span for frame in frames) == (
        "The server died",
        "but two patients survived.",
        "The hospital restored power.",
    )


@pytest.mark.parametrize(
    "reply",
    (
        "compliance, but make it onboarding",
        "the setup looked routine, but the result changed everything",
        "plot twist: the fallback became the launch plan",
    ),
)
def test_setup_twist_is_extracted_as_typed_shell(reply: str) -> None:
    shell = adjudicate_reply_shell(extract_semantic_inspection("", reply))

    assert shell.operator == "setup_twist"
    assert shell.subject_span
    assert shell.complement_span
    assert shell.evidence_spans


def test_mixed_technical_failure_and_medical_profession_outcome_remains_human_sensitive() -> None:
    inspection = extract_semantic_inspection(
        "The server died. Two clinicians remain in intensive care after the outage."
    )
    harm = adjudicate_human_harm(inspection)

    assert harm.harm_referent == "human"
    assert harm.harm_state == "intensive_care"
    assert harm.literal_harm
    assert harm.metaphorical_or_technical_usage
    assert {event.event_type for event in inspection.source_events} >= {
        "technical_failure",
        "intensive_care",
    }
