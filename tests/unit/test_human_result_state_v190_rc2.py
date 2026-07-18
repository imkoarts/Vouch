from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.schemas.content import (
    SemanticCoverageResolution,
    SemanticEntity,
    SemanticEvent,
    SemanticInspection,
    SemanticProviderProposal,
    SemanticProviderRequest,
    SemanticUnresolvedItem,
)
from app.services.semantic_adjudication import adjudicate_human_harm
from app.services.semantic_extraction import extract_semantic_inspection
from app.services.semantic_extraction_coordinator import SemanticExtractionCoordinator


@dataclass
class FakeSemanticProvider:
    proposal: SemanticProviderProposal
    calls: list[SemanticProviderRequest]

    async def extract_semantics(self, request: SemanticProviderRequest) -> SemanticProviderProposal:
        self.calls.append(request)
        return self.proposal


def _possible_harm_resolutions(
    inspection: SemanticInspection,
    *,
    disposition: str,
    evidence_spans: tuple[str, ...],
    reason: str = "",
) -> tuple[SemanticCoverageResolution, ...]:
    return tuple(
        SemanticCoverageResolution(
            item_id=item.item_id,
            side="source",
            clause_id=item.clause_id,
            clause_span=item.clause_span,
            unresolved_span=item.unresolved_span,
            category="possible_harm",
            disposition=disposition,
            reason=reason,
            evidence_spans=evidence_spans,
        )
        for coverage in inspection.source_coverage
        for item in coverage.unresolved_items
        if item.category == "possible_harm"
    )


@pytest.mark.parametrize(
    ("source", "expected_span"),
    (
        (
            "The collision left three passengers glassy-eyed and barely rousable.",
            "glassy-eyed and barely rousable",
        ),
        (
            "The blast rendered a technician dependent on a breathing machine.",
            "dependent on a breathing machine",
        ),
        (
            "The spill left two research assistants minimally responsive.",
            "minimally responsive",
        ),
        (
            "The crash left one resident dependent on a respirator.",
            "dependent on a respirator",
        ),
        (
            "After the fire, two residents remained neurologically diminished.",
            "neurologically diminished",
        ),
    ),
)
def test_asserted_human_result_state_never_completes_safe_coverage(
    source: str,
    expected_span: str,
) -> None:
    result = SemanticExtractionCoordinator().inspect_local(source)

    assert result.escalation.required
    assert "possible_harm" in result.escalation.material_categories
    assert any(
        expected_span in span or span in expected_span
        for coverage in result.canonical.source_coverage
        for span in coverage.unresolved_spans
    )
    assert all(not coverage.complete for coverage in result.canonical.source_coverage)
    assert adjudicate_human_harm(result.canonical).confidence == "medium"


@pytest.mark.parametrize(
    ("source", "expected_state"),
    (
        ("The collision left three passengers barely conscious.", "critical_condition"),
        ("The incident left a worker with lasting loss of mobility.", "permanent_injury"),
        ("The crash put one passenger on assisted ventilation.", "critical_condition"),
    ),
)
def test_known_typed_human_result_states_become_explicit_harm(
    source: str,
    expected_state: str,
) -> None:
    inspection = extract_semantic_inspection(source)
    harm = adjudicate_human_harm(inspection)

    assert harm.literal_harm
    assert harm.harm_state == expected_state
    assert all(coverage.complete for coverage in inspection.source_coverage)


@pytest.mark.parametrize(
    "source",
    (
        "The delay left two users waiting outside.",
        "The update left the team behind schedule.",
        "The design left customers dependent on one vendor.",
        "The meeting left everyone uncertain.",
        "The patch left the interface responsive.",
        "The outage left the server dependent on a backup node.",
    ),
)
def test_harmless_or_nonhuman_result_states_are_not_possible_harm(source: str) -> None:
    inspection = extract_semantic_inspection(source)

    assert all(
        "possible_harm" not in coverage.unresolved_categories
        for coverage in inspection.source_coverage
    )
    assert adjudicate_human_harm(inspection).literal_harm is False


@pytest.mark.parametrize(
    "source",
    (
        "The collision did not leave the passenger unconscious.",
        "The blast may leave workers dependent on breathing support.",
        "If the fire leaves residents unable to breathe, crews will evacuate them.",
        "Officials feared the incident would leave a worker unable to walk.",
        "The report said the crash could leave passengers in a reduced state of consciousness.",
    ),
)
def test_unasserted_result_states_do_not_become_asserted_harm(source: str) -> None:
    inspection = extract_semantic_inspection(source)

    assert not any(event.explicit_outcome for event in inspection.source_events)
    assert adjudicate_human_harm(inspection).literal_harm is False


def test_human_result_state_candidate_binds_roles_and_exact_spans() -> None:
    inspection = extract_semantic_inspection(
        "The collapse left two engineers dependent on an unfamiliar breathing apparatus."
    )
    coverage = inspection.source_coverage[0]

    assert "possible_harm" in coverage.unresolved_categories
    assert "dependent on an unfamiliar breathing apparatus" in coverage.unresolved_spans
    assert "left" in coverage.consumed_spans
    assert any(entity.text_span == "engineers" for entity in inspection.source_entities)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_type",
    (
        "serious_injury",
        "critical_condition",
        "hospitalization",
        "intensive_care",
        "permanent_injury",
    ),
)
async def test_provider_can_resolve_human_result_state_to_typed_harm(event_type: str) -> None:
    source = "The collision left two passengers glassy-eyed and barely rousable."
    local = SemanticExtractionCoordinator().inspect_local(source)
    affected = SemanticEntity(
        text_span="passengers",
        canonical_name="passengers",
        entity_type="group_of_people",
        confidence="high",
    )
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            source_entities=(affected,),
            source_events=(
                SemanticEvent(
                    event_id="P1",
                    predicate="left",
                    affected_entity=affected,
                    event_type=event_type,
                    polarity="negative",
                    explicit_outcome=True,
                    assertion_state="asserted",
                    evidence_spans=("left", "glassy-eyed and barely rousable"),
                ),
            ),
        ),
        coverage_resolutions=_possible_harm_resolutions(
            local.deterministic,
            disposition="typed_relation",
            evidence_spans=("left", "glassy-eyed and barely rousable"),
        ),
    )
    provider = FakeSemanticProvider(proposal, [])

    result = await SemanticExtractionCoordinator(provider).inspect(source)

    assert result.provider_validation_errors == ()
    assert result.provider_used
    assert len(provider.calls) == 1
    assert provider.calls[0].source_clauses == (source,)
    assert adjudicate_human_harm(result.canonical).harm_state == event_type
    assert result.unresolved_after_reconciliation == ()


@pytest.mark.asyncio
async def test_provider_can_resolve_human_result_state_as_harmless_with_exact_reason() -> None:
    source = "The collision left two passengers waiting at the assembly point."
    local = SemanticExtractionCoordinator().inspect_local(source)
    # The deterministic layer may classify this as harmless already. Force a structurally
    # equivalent material item to exercise the provider harmless-resolution contract.
    forced_item = SemanticUnresolvedItem(
        item_id="test-source-C1-waiting",
        side="source",
        clause_id=local.deterministic.source_coverage[0].clause_id,
        clause_span=source,
        unresolved_span="waiting at the assembly point",
        category="possible_harm",
    )
    material = local.deterministic.model_copy(
        update={
            "source_coverage": (
                local.deterministic.source_coverage[0].model_copy(
                    update={
                        "unresolved_items": (forced_item,),
                        "unresolved_spans": ("waiting at the assembly point",),
                        "unresolved_categories": ("possible_harm",),
                        "complete": False,
                    }
                ),
            ),
            "unresolved_material": ("waiting at the assembly point",),
        }
    )
    proposal = SemanticProviderProposal(
        coverage_resolutions=(
            SemanticCoverageResolution(
                item_id=forced_item.item_id,
                side="source",
                clause_id=forced_item.clause_id,
                clause_span=source,
                unresolved_span="waiting at the assembly point",
                category="possible_harm",
                disposition="harmless",
                reason=(
                    "The complement is an ordinary location/activity state, not a medical outcome."
                ),
                evidence_spans=("waiting at the assembly point",),
            ),
        )
    )
    # Validate and reconcile through the public helpers against the forced deterministic material.
    from app.services.semantic_provider_validation import validate_semantic_provider_proposal
    from app.services.semantic_reconciliation import reconcile_semantic_provider_proposal

    errors = validate_semantic_provider_proposal(
        material,
        proposal,
        source_clauses=(source,),
        reply_clauses=(),
    )
    canonical = reconcile_semantic_provider_proposal(material, proposal)

    assert errors == ()
    assert canonical.source_coverage[0].complete
    assert canonical.source_coverage[0].unresolved_categories == ()


@pytest.mark.asyncio
@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_harmless", (False, True))
async def test_provider_none_or_invalid_harmless_resolution_cannot_erase_possible_harm(
    invalid_harmless: bool,
) -> None:
    source = "The collision left two passengers glassy-eyed and barely rousable."
    local = SemanticExtractionCoordinator().inspect_local(source)
    item = next(
        item
        for coverage in local.deterministic.source_coverage
        for item in coverage.unresolved_items
        if item.category == "possible_harm"
    )
    proposal = (
        SemanticProviderProposal(
            coverage_resolutions=(
                SemanticCoverageResolution(
                    item_id=item.item_id,
                    side="source",
                    clause_id=item.clause_id,
                    clause_span=item.clause_span,
                    unresolved_span=item.unresolved_span,
                    category="possible_harm",
                    disposition="harmless",
                    reason="",
                    evidence_spans=(),
                ),
            )
        )
        if invalid_harmless
        else SemanticProviderProposal()
    )
    provider = FakeSemanticProvider(proposal, [])

    result = await SemanticExtractionCoordinator(provider).inspect(source)

    assert result.escalation.required
    assert "possible_harm" in {
        category
        for coverage in result.canonical.source_coverage
        for category in coverage.unresolved_categories
    }
    assert result.unresolved_after_reconciliation


def test_material_human_result_state_offline_fails_closed_only_as_uncertain_harm() -> None:
    source = "The collapse left three workers dependent on an unfamiliar breathing apparatus."
    result = SemanticExtractionCoordinator().inspect_local(source)
    harm = adjudicate_human_harm(result.canonical)

    assert harm.literal_harm is False
    assert harm.harm_referent == "unknown"
    assert harm.evidence_spans
    assert result.escalation.required


@pytest.mark.parametrize(
    "source",
    (
        "The collision left one passenger glassy-eyed and barely rousable.",
        "The collision left four passengers glassy-eyed and barely rousable.",
        "Four passengers were left glassy-eyed and barely rousable by the collision.",
        "After the collision, a passenger remained neurologically diminished.",
        "Barely rousable, the passenger remained after the collision.",
        "The collision left a passenger glassy-eyed; barely rousable.",
    ),
)
def test_result_state_metamorphisms_preserve_possible_harm(source: str) -> None:
    result = SemanticExtractionCoordinator().inspect_local(source)

    assert result.escalation.required
    assert "possible_harm" in result.escalation.material_categories
    assert any(
        not coverage.complete and "possible_harm" in coverage.unresolved_categories
        for coverage in result.canonical.source_coverage
    )


@pytest.mark.parametrize(
    "source",
    (
        "Officials reported that the collision left a passenger barely conscious.",
        "According to responders, a passenger remained barely conscious after the collision.",
    ),
)
def test_reported_typed_result_state_remains_explicit_harm(source: str) -> None:
    inspection = extract_semantic_inspection(source)
    events = tuple(
        event for event in inspection.source_events if event.event_type == "critical_condition"
    )

    assert events
    assert any(event.assertion_state == "reported" for event in events)
    assert adjudicate_human_harm(inspection).literal_harm


def test_result_state_coverage_property_matches_bound_candidate_roles() -> None:
    from app.services.semantic_composition import (
        extract_result_state_candidates,
        segment_clauses,
    )

    source = "The collapse rendered two workers glassy-eyed and slow to respond."
    frames = segment_clauses(source)
    candidates = extract_result_state_candidates(frames)
    inspection = extract_semantic_inspection(source)

    candidate = candidates[0]
    coverage = inspection.source_coverage[0]
    assert candidate.affected_entity_type == "group_of_people"
    assert candidate.linking_predicate_span == "rendered"
    assert candidate.state_complement_span == "glassy-eyed and slow to respond"
    assert candidate.causal_event_span == "collapse"
    assert not candidate.negated
    assert candidate.outcome_type == "unknown"
    assert coverage.unresolved_spans == (candidate.state_complement_span,)
    assert coverage.unresolved_categories == ("possible_harm",)
    assert not coverage.complete


def test_physical_event_with_harmless_human_location_state_is_not_possible_harm() -> None:
    inspection = extract_semantic_inspection(
        "The collision left two passengers waiting outside the terminal."
    )

    assert all(
        "possible_harm" not in coverage.unresolved_categories
        for coverage in inspection.source_coverage
    )
    assert not adjudicate_human_harm(inspection).literal_harm


@pytest.mark.asyncio
async def test_nonhuman_provider_event_cannot_resolve_human_possible_harm() -> None:
    source = "The collision left two passengers glassy-eyed and barely rousable."
    local = SemanticExtractionCoordinator().inspect_local(source)
    system = SemanticEntity(
        text_span="collision",
        canonical_name="collision",
        entity_type="technical_system",
        confidence="high",
    )
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            source_entities=(system,),
            source_events=(
                SemanticEvent(
                    event_id="P1",
                    predicate="left",
                    affected_entity=system,
                    event_type="technical_failure",
                    polarity="negative",
                    explicit_outcome=False,
                    assertion_state="asserted",
                    evidence_spans=("left", "glassy-eyed and barely rousable"),
                ),
            ),
        ),
        coverage_resolutions=_possible_harm_resolutions(
            local.deterministic,
            disposition="typed_relation",
            evidence_spans=("left", "glassy-eyed and barely rousable"),
        ),
    )

    result = await SemanticExtractionCoordinator(FakeSemanticProvider(proposal, [])).inspect(source)

    assert any(
        "category-compatible proposed relation" in error
        for error in result.provider_validation_errors
    )
    assert "possible_harm" in {
        category
        for coverage in result.canonical.source_coverage
        for category in coverage.unresolved_categories
    }


@pytest.mark.asyncio
async def test_provider_unresolved_disposition_preserves_human_result_state_uncertainty() -> None:
    source = "The collapse rendered two workers glassy-eyed and slow to respond."
    local = SemanticExtractionCoordinator().inspect_local(source)
    resolutions = _possible_harm_resolutions(
        local.deterministic,
        disposition="unresolved",
        evidence_spans=("glassy-eyed and slow to respond",),
        reason="The bounded proposal cannot type the physiological state confidently.",
    )
    proposal = SemanticProviderProposal(remaining_unresolved_items=resolutions)

    result = await SemanticExtractionCoordinator(FakeSemanticProvider(proposal, [])).inspect(source)

    assert result.provider_validation_errors == ()
    assert result.provider_used
    assert result.unresolved_after_reconciliation == ("glassy-eyed and slow to respond",)
    assert "possible_harm" in {
        category
        for coverage in result.canonical.source_coverage
        for category in coverage.unresolved_categories
    }
