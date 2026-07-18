from __future__ import annotations

import pytest

from app.services.semantic_adjudication import (
    adjudicate_evaluative_irony,
    adjudicate_human_harm,
    adjudicate_reply_shell,
)
from app.services.semantic_extraction import extract_semantic_inspection


@pytest.mark.parametrize(
    "source",
    (
        "Three workers died in the crash.",
        "Three workers were killed in the crash.",
        "3 workers were pronounced dead after the crash.",
        "One worker died in the crash.",
        "Workers died in the crash!",
        "After the crash, three workers did not survive.",
    ),
)
def test_voice_number_and_punctuation_preserve_human_death(source: str) -> None:
    harm = adjudicate_human_harm(extract_semantic_inspection(source))

    assert harm.harm_referent == "human"
    assert harm.harm_state == "death"
    assert harm.literal_harm
    assert harm.evidence_spans


@pytest.mark.parametrize(
    "referent",
    ("server", "database", "pipeline", "build", "product", "token", "bill"),
)
def test_same_death_predicate_with_nonhuman_referent_is_nonhuman(referent: str) -> None:
    harm = adjudicate_human_harm(extract_semantic_inspection(f"The {referent} died overnight."))

    assert harm.harm_referent == "non_human"
    assert not harm.literal_harm


@pytest.mark.parametrize(
    "reply",
    (
        "stellar execution from the outage",
        "a standing ovation for the outage",
        "five stars for the outage",
        "the outage deserves a medal",
        "the outage crushed it",
    ),
)
def test_equivalent_positive_evaluation_relations_preserve_sarcasm(reply: str) -> None:
    inspection = extract_semantic_inspection(
        "A hospital reported ten deaths after the outage.", reply
    )
    irony = adjudicate_evaluative_irony(inspection)

    assert irony.positive_evaluation
    assert irony.implicit_sarcasm
    assert irony.target_source_alignment


@pytest.mark.parametrize(
    "reply",
    (
        "onboarding is just paperwork",
        "onboarding is glorified paperwork",
        "onboarding is paperwork in disguise",
        "onboarding: paperwork with better branding",
        "as it turns out, onboarding is paperwork",
    ),
)
def test_surface_changes_map_to_canonical_shell_relations(reply: str) -> None:
    shell = adjudicate_reply_shell(extract_semantic_inspection("", reply))

    assert shell.operator in {"reduction", "decorated_identity", "repackaging", "reveal"}
    assert shell.subject_span
    assert shell.complement_span
    assert shell.generic_reusability


@pytest.mark.parametrize(
    "reply",
    (
        "a triangle is a polygon with three sides",
        "x equals the sum divided by the count",
        'the documentation says "onboarding is paperwork"',
        "the protocol is the system used for consensus",
    ),
)
def test_literal_definitions_formulas_and_quotations_are_not_shells(reply: str) -> None:
    shell = adjudicate_reply_shell(extract_semantic_inspection("", reply))

    assert shell.operator == "none"
