from __future__ import annotations

import pytest

from app.services.semantic_adjudication import (
    adjudicate_evaluative_irony,
    adjudicate_human_harm,
    adjudicate_humor_intent,
    adjudicate_reply_shell,
)
from app.services.semantic_extraction import extract_semantic_inspection
from scripts.run_external_semantic_holdout import HoldoutDocument, run_holdout


@pytest.mark.parametrize(
    ("source", "expected_state", "evidence"),
    (
        (
            "The blast sent four technicians to intensive care.",
            "intensive_care",
            "intensive care",
        ),
        (
            "The collision put two paramedics on life support.",
            "critical_condition",
            "on life support",
        ),
        (
            "The crash left two contractors with life threatening injuries.",
            "serious_injury",
            "injuries",
        ),
        (
            "The incident left three residents with permanent physical damage.",
            "permanent_injury",
            "physical damage",
        ),
        (
            "The fire sent five staffers to the hospital.",
            "hospitalization",
            "the hospital",
        ),
    ),
)
def test_result_complement_medical_binding_is_compositional(
    source: str,
    expected_state: str,
    evidence: str,
) -> None:
    inspection = extract_semantic_inspection(source)
    harm = adjudicate_human_harm(inspection)

    assert harm.harm_referent == "human"
    assert harm.harm_state == expected_state
    assert harm.literal_harm
    assert evidence in harm.evidence_spans
    assert all(item.complete for item in inspection.source_coverage)


@pytest.mark.parametrize(
    "source",
    (
        "Two patients deteriorated after smoke inhalation.",
        "Three residents were worsening after the incident.",
        "A technician remained unresponsive after the blast.",
    ),
)
def test_unresolved_medical_escalation_is_possible_harm(source: str) -> None:
    inspection = extract_semantic_inspection(source)
    harm = adjudicate_human_harm(inspection)

    assert harm.harm_referent == "unknown"
    assert not harm.literal_harm
    assert any(
        "possible_harm" in coverage.unresolved_categories for coverage in inspection.source_coverage
    )
    assert any(not coverage.complete for coverage in inspection.source_coverage)


def test_mixed_technical_failure_and_result_bound_hospitalization_remains_sensitive() -> None:
    inspection = extract_semantic_inspection(
        "The deployment failed. The outage sent two nurses to the hospital."
    )
    harm = adjudicate_human_harm(inspection)

    assert harm.harm_referent == "human"
    assert harm.harm_state == "hospitalization"
    assert harm.literal_harm
    assert harm.metaphorical_or_technical_usage
    assert {event.event_type for event in inspection.source_events} >= {
        "technical_failure",
        "hospitalization",
    }


@pytest.mark.parametrize(
    ("source", "entity_type"),
    (
        ("The reserve-backed stablecoin died after redemptions surged.", "crypto_or_protocol"),
        ("The municipal bond was killed before issuance.", "business_or_financial_object"),
        ("The pension fund died after the merger.", "business_or_financial_object"),
        ("The ballot initiative died before the vote.", "legal_or_political_object"),
        ("The local referendum was killed in committee.", "legal_or_political_object"),
    ),
)
def test_contextual_compound_heads_type_clear_non_human_referents(
    source: str,
    entity_type: str,
) -> None:
    inspection = extract_semantic_inspection(source)
    harm = adjudicate_human_harm(inspection)

    assert harm.harm_referent == "non_human"
    assert harm.harm_state == "non_human_failure"
    assert any(
        event.affected_entity is not None and event.affected_entity.entity_type == entity_type
        for event in inspection.source_events
    )


def test_unknown_named_referent_remains_unknown_and_fail_closed() -> None:
    harm = adjudicate_human_harm(extract_semantic_inspection("ORION died overnight."))

    assert harm.harm_referent == "unknown"
    assert not harm.literal_harm
    assert harm.evidence_spans == ("died",)


@pytest.mark.parametrize(
    "reply",
    (
        "take a bow, outage",
        "credit to the outage",
        "what a performance by the outage",
        "the grid gets the honors",
        "here's to the outage",
    ),
)
def test_recognition_and_celebration_frames_bind_to_harmful_target(reply: str) -> None:
    inspection = extract_semantic_inspection(
        "A hospital reported ten deaths after the outage.",
        reply,
    )
    irony = adjudicate_evaluative_irony(inspection)
    intent = adjudicate_humor_intent(
        inspection,
        metadata_says_humor=False,
        reaction_type="none",
        template_humor=False,
    )

    assert irony.implicit_sarcasm
    assert irony.target_source_alignment
    assert intent.text_appears_humorous
    assert intent.humor_safety_required
    assert all(item.complete for item in inspection.reply_coverage)


def test_unknown_exclamative_evaluation_on_harmful_source_is_uncertain_not_plain() -> None:
    inspection = extract_semantic_inspection(
        "Two patients died during the outage.",
        "what a masterclass from the outage",
    )
    intent = adjudicate_humor_intent(
        inspection,
        metadata_says_humor=False,
        reaction_type="none",
        template_humor=False,
    )

    assert any(
        "possible_evaluation" in coverage.unresolved_categories
        for coverage in inspection.reply_coverage
    )
    assert intent.detected_reaction_type == "uncertain"
    assert intent.humor_safety_required
    assert "HUMOR_INTENT_UNCERTAIN" in intent.issue_codes


def test_literal_recognition_on_positive_source_remains_allowed() -> None:
    inspection = extract_semantic_inspection(
        "The hospital restored power and completed the recovery safely.",
        "credit to the response team",
    )
    irony = adjudicate_evaluative_irony(inspection)

    assert irony.positive_evaluation
    assert irony.literal_positive_reading_plausible
    assert not irony.implicit_sarcasm


@pytest.mark.parametrize(
    ("reply", "operator"),
    (
        ("with glossy packaging, onboarding is paperwork", "decorated_identity"),
        ("beneath the glossy packaging, onboarding is paperwork", "reduction"),
        ("for all the branding, onboarding is paperwork", "reduction"),
        ("onboarding is paperwork, polished for product teams", "decorated_identity"),
        ("onboarding is paperwork wearing a product costume", "decorated_identity"),
    ),
)
def test_fronted_and_post_complement_decoration_bind_as_roles(
    reply: str,
    operator: str,
) -> None:
    inspection = extract_semantic_inspection("", reply)
    shell = adjudicate_reply_shell(inspection)

    assert shell.operator == operator
    assert shell.subject_span == "onboarding"
    assert shell.complement_span == "paperwork"
    assert shell.decoration_span
    assert all(item.complete for item in inspection.reply_coverage)


@pytest.mark.parametrize(
    "reply",
    (
        "a checksum is a digest used to detect changes",
        "revenue equals price multiplied by volume",
        'the report calls "outage" a service interruption',
    ),
)
def test_literal_controls_remain_non_shells(reply: str) -> None:
    shell = adjudicate_reply_shell(extract_semantic_inspection("", reply))

    assert shell.operator == "none"


def test_external_runner_separates_raw_safety_from_canonical_candidate_gating() -> None:
    document = HoldoutDocument.model_validate(
        {
            "schema_version": "1.0",
            "holdout_id": "runner-parity",
            "cases": [
                {
                    "id": "respectful-empty-reply",
                    "source": "Two patients died during the outage.",
                    "reply": "",
                    "expect": {
                        "semantic_candidate_eligible": True,
                        "required_raw_humor_safety_issue_codes": ["HUMOR_ON_TRAGEDY"],
                        "forbidden_canonical_candidate_issue_codes": ["HUMOR_ON_TRAGEDY"],
                    },
                },
                {
                    "id": "sarcastic-reply",
                    "source": "Two patients died during the outage.",
                    "reply": "credit to the outage",
                    "expect": {
                        "semantic_candidate_eligible": False,
                        "required_raw_humor_safety_issue_codes": ["HUMOR_ON_TRAGEDY"],
                        "required_canonical_candidate_issue_codes": ["HUMOR_ON_TRAGEDY"],
                    },
                },
            ],
        }
    )

    summary = run_holdout(document)

    assert summary["all_passed"] is True
    respectful = summary["results"][0]["observed"]
    assert respectful["raw_humor_safety_issue_codes"] == ["HUMOR_ON_TRAGEDY"]
    assert respectful["canonical_candidate_issue_codes"] == []
    sarcastic = summary["results"][1]["observed"]
    assert sarcastic["canonical_candidate_issue_codes"] == ["HUMOR_ON_TRAGEDY"]
