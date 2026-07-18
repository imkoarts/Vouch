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
        "Three workers died in the blast.",
        "Three workers were killed in the blast.",
        "In the blast, three workers died.",
        "3 workers died in the blast!",
        "A worker died in the blast.",
        "Workers died during the blast.",
    ),
)
def test_voice_number_order_and_punctuation_preserve_death(source: str) -> None:
    harm = adjudicate_human_harm(extract_semantic_inspection(source))

    assert harm.harm_referent == "human"
    assert harm.harm_state == "death"


@pytest.mark.parametrize(
    "source",
    (
        "The server died overnight.",
        "Overnight, the server died.",
        "The servers died overnight.",
        "The server was dead overnight.",
        "The server died overnight!",
    ),
)
def test_surface_changes_preserve_non_human_failure(source: str) -> None:
    harm = adjudicate_human_harm(extract_semantic_inspection(source))

    assert harm.harm_referent == "non_human"
    assert not harm.literal_harm


@pytest.mark.parametrize(
    "reply",
    (
        "five stars for the outage",
        "for the outage: five stars",
        "the outage earned five stars",
        "five stars, for the outage!",
    ),
)
def test_target_position_and_punctuation_preserve_irony(reply: str) -> None:
    irony = adjudicate_evaluative_irony(
        extract_semantic_inspection(
            "Two patients died during the outage.",
            reply,
        )
    )

    assert irony.implicit_sarcasm


@pytest.mark.parametrize(
    "reply",
    (
        "onboarding is glorified paperwork",
        "onboarding: glorified paperwork",
        "onboarding is paperwork in disguise",
        "onboarding is paperwork with better branding",
        "onboarding is paperwork, but fancier",
    ),
)
def test_packaging_surface_position_preserves_shell(reply: str) -> None:
    shell = adjudicate_reply_shell(extract_semantic_inspection("", reply))

    assert shell.operator == "decorated_identity"
    assert shell.subject_span
    assert shell.complement_span
    assert shell.decoration_span or shell.repackaging_marker


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("A worker died overnight.", "human"),
        ("A server died overnight.", "non_human"),
        ("PHOENIX died overnight.", "unknown"),
    ),
)
def test_same_predicate_changes_only_with_referent_type(source: str, expected: str) -> None:
    harm = adjudicate_human_harm(extract_semantic_inspection(source))

    assert harm.harm_referent == expected


@pytest.mark.parametrize("punctuation", ("", ".", "!", "..."))
def test_coverage_and_adjudication_are_punctuation_stable(punctuation: str) -> None:
    inspection = extract_semantic_inspection(
        "Two patients died during the outage.",
        f"excellent work from the outage{punctuation}",
    )

    assert adjudicate_human_harm(inspection).literal_harm
    assert adjudicate_evaluative_irony(inspection).implicit_sarcasm
    assert all(item.complete for item in inspection.reply_coverage)
