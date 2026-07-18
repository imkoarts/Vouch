from __future__ import annotations

import inspect

import pytest

from app.schemas.content import SemanticInspection
from app.services.semantic_adjudication import (
    adjudicate_evaluative_irony,
    adjudicate_human_harm,
    adjudicate_reply_shell,
)
from app.services.semantic_extraction import extract_semantic_inspection
from app.services.semantic_reconciliation import reconcile_semantic_inspections


def test_every_material_clause_has_coverage() -> None:
    source = "The server died. Two workers were hospitalized after the outage."
    reply = "five stars for the outage; onboarding is glorified paperwork"
    inspection = extract_semantic_inspection(source, reply)

    assert len(inspection.source_coverage) == 2
    assert len(inspection.reply_coverage) == 2
    assert all(
        item.clause_span for item in (*inspection.source_coverage, *inspection.reply_coverage)
    )


def test_unconsumed_predicate_like_span_is_preserved() -> None:
    source = "The release oscillated overnight."
    inspection = extract_semantic_inspection(source)

    coverage = inspection.source_coverage[0]
    assert "oscillated" in coverage.unresolved_spans
    assert "unknown_predicate" in coverage.unresolved_categories
    assert not coverage.complete


def test_unparsed_result_state_in_human_event_context_is_possible_harm() -> None:
    source = "Two workers were incapacitated after the blast."
    inspection = extract_semantic_inspection(source)
    coverage = inspection.source_coverage[0]
    harm = adjudicate_human_harm(inspection)

    assert "incapacitated" in coverage.unresolved_spans
    assert "possible_harm" in coverage.unresolved_categories
    assert harm.harm_referent == "unknown"
    assert not harm.literal_harm
    assert harm.evidence_spans == ("incapacitated",)


def test_unaligned_positive_evaluation_on_harmful_source_is_uncertain() -> None:
    inspection = extract_semantic_inspection(
        "Two patients died during the outage.",
        "beautiful work, weather",
    )
    irony = adjudicate_evaluative_irony(inspection)

    assert irony.positive_evaluation
    assert irony.negative_or_harmful_source_context
    assert not irony.target_source_alignment
    assert not irony.implicit_sarcasm
    assert irony.confidence == "medium"


def test_unbound_copular_material_is_uncertain_shell() -> None:
    inspection = extract_semantic_inspection("", "is paperwork")
    shell = adjudicate_reply_shell(inspection)

    assert shell.operator == "uncertain"
    assert inspection.reply_coverage[0].unresolved_categories == ("possible_reduction",)


@pytest.mark.parametrize(
    ("source", "reply"),
    (
        ("Two patients died during the outage.", "five stars for the outage"),
        ("The pipeline died overnight.", "achievement unlocked"),
        ("The hospital restored power safely.", "excellent work from the response team"),
        ("", "onboarding is glorified paperwork"),
    ),
)
def test_high_confidence_relations_have_exact_consumed_spans(source: str, reply: str) -> None:
    inspection = extract_semantic_inspection(source, reply)

    for coverage in inspection.source_coverage:
        assert all(span in coverage.clause_span for span in coverage.consumed_spans)
    for coverage in inspection.reply_coverage:
        assert all(span in coverage.clause_span for span in coverage.consumed_spans)
    assert all(
        span in source for event in inspection.source_events for span in event.evidence_spans
    )
    assert all(span in reply for item in inspection.evaluations for span in item.evidence_spans)
    assert all(span in reply for shell in inspection.reply_shells for span in shell.evidence_spans)


def test_provider_none_cannot_erase_coverage_or_relations() -> None:
    source = "Two patients died during the outage."
    reply = "five stars for the outage"
    deterministic = extract_semantic_inspection(source, reply)
    canonical = reconcile_semantic_inspections(
        deterministic,
        SemanticInspection(),
        source_text=source,
        reply_text=reply,
    )

    assert canonical.source_events == deterministic.source_events
    assert canonical.evaluations == deterministic.evaluations
    assert canonical.source_coverage == deterministic.source_coverage
    assert canonical.reply_coverage == deterministic.reply_coverage


def test_reconciliation_preserves_exact_unresolved_spans() -> None:
    source = "Two workers were incapacitated after the blast."
    deterministic = extract_semantic_inspection(source)
    canonical = reconcile_semantic_inspections(
        deterministic,
        SemanticInspection(),
        source_text=source,
        reply_text="",
    )

    assert canonical.source_coverage[0].unresolved_spans == ("incapacitated",)
    assert canonical.unresolved_material == ("incapacitated",)


def test_adjudication_still_has_no_surface_text_dependency() -> None:
    import app.services.semantic_adjudication as module

    source = inspect.getsource(module)
    assert "import re" not in source
    for function in (
        module.adjudicate_human_harm,
        module.adjudicate_evaluative_irony,
        module.adjudicate_humor_safety,
        module.adjudicate_reply_shell,
    ):
        assert tuple(inspect.signature(function).parameters) == ("inspection",)
