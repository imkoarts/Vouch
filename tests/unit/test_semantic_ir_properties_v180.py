from __future__ import annotations

import pytest

from app.schemas.content import SemanticInspection
from app.services.semantic_adjudication import adjudicate_evaluative_irony, adjudicate_human_harm
from app.services.semantic_extraction import extract_semantic_inspection
from app.services.semantic_reconciliation import reconcile_semantic_inspections


@pytest.mark.parametrize(
    "source",
    (
        "Alice stood near the explosion.",
        "Two workers observed the blast from a safe room.",
        "Passengers discussed the outage.",
        "A patient reported hearing the crash.",
    ),
)
def test_event_proximity_alone_never_creates_human_harm(source: str) -> None:
    harm = adjudicate_human_harm(extract_semantic_inspection(source))

    assert not harm.literal_harm
    assert harm.harm_referent != "human"


def test_capitalization_alone_never_creates_personhood() -> None:
    lower = extract_semantic_inspection("phoenix died overnight.")
    upper = extract_semantic_inspection("PHOENIX died overnight.")

    assert lower.source_events[0].affected_entity is not None
    assert upper.source_events[0].affected_entity is not None
    assert lower.source_events[0].affected_entity.entity_type == "unknown"
    assert upper.source_events[0].affected_entity.entity_type == "unknown"


@pytest.mark.parametrize("punctuation", ("", ".", "!", "..."))
def test_adjudication_is_stable_across_harmless_punctuation(punctuation: str) -> None:
    inspection = extract_semantic_inspection(
        "A hospital reported ten deaths after the outage.",
        f"stellar execution from the outage{punctuation}",
    )

    assert adjudicate_evaluative_irony(inspection).implicit_sarcasm


def test_human_harm_requires_explicit_outcome_span() -> None:
    inspection = extract_semantic_inspection("Ten passengers were near the crash site.")
    harm = adjudicate_human_harm(inspection)

    assert not harm.literal_harm
    assert not any(
        event.explicit_outcome and event.event_type in {"death", "serious_injury"}
        for event in inspection.source_events
    )


def test_empty_provider_ir_cannot_erase_deterministic_extraction() -> None:
    source = "Two patients died during the outage."
    deterministic = extract_semantic_inspection(source, "five stars for the outage")
    canonical = reconcile_semantic_inspections(
        deterministic,
        SemanticInspection(),
        source_text=source,
        reply_text="five stars for the outage",
    )

    assert canonical.source_events
    assert canonical.evaluations
    assert adjudicate_human_harm(canonical).literal_harm


def test_adjudication_layer_has_no_raw_text_or_regex_dependency() -> None:
    import inspect

    import app.services.semantic_adjudication as module

    source = inspect.getsource(module)
    assert "import re" not in source
    assert "re.compile" not in source
    for function in (
        module.adjudicate_human_harm,
        module.adjudicate_evaluative_irony,
        module.adjudicate_humor_safety,
        module.adjudicate_reply_shell,
    ):
        parameters = inspect.signature(function).parameters
        assert tuple(parameters) == ("inspection",)
