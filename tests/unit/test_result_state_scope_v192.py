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
)
from app.services.semantic_adjudication import (
    adjudicate_human_harm,
    adjudicate_humor_intent,
    adjudicate_reply_shell,
)
from app.services.semantic_composition import (
    extract_result_complement_frames,
    segment_clauses,
)
from app.services.semantic_extraction_coordinator import SemanticExtractionCoordinator


def _categories(result, *, source: bool = True) -> set[str]:
    coverage = result.canonical.source_coverage if source else result.canonical.reply_coverage
    return {category for item in coverage for category in item.unresolved_categories}


def _spans(result, *, source: bool = True) -> set[str]:
    coverage = result.canonical.source_coverage if source else result.canonical.reply_coverage
    return {span for item in coverage for span in item.unresolved_spans}


def _intent(result):
    return adjudicate_humor_intent(
        result.canonical,
        metadata_says_humor=False,
        reaction_type="none",
        template_humor=False,
    )


@pytest.mark.parametrize(
    ("source", "expected_span", "expected_form"),
    (
        ("The shock left one rider seeing halos.", "seeing halos", "gerund"),
        ("The fumes left two operators speaking incoherently.", "speaking incoherently", "gerund"),
        (
            "The smoke left a resident requiring breathing assistance.",
            "requiring breathing assistance",
            "dependency",
        ),
        (
            "The solvent left two workers with reduced sensation in one arm.",
            "reduced sensation in one arm",
            "with_state",
        ),
        (
            "The release left three staff disoriented and unable to stand.",
            "disoriented and unable to stand",
            "coordinated",
        ),
        (
            "The collision left one passenger unable to maintain balance.",
            "unable to maintain balance",
            "infinitival",
        ),
    ),
)
def test_result_complement_grammar_preserves_complete_material_span(
    source: str,
    expected_span: str,
    expected_form: str,
) -> None:
    frames = extract_result_complement_frames(segment_clauses(source))
    result = SemanticExtractionCoordinator().inspect_local(source)

    assert len(frames) == 1
    assert frames[0].complement_span == expected_span
    assert frames[0].complement_form == expected_form
    assert frames[0].affected_entity_span
    assert frames[0].assertion_scope == "asserted"
    assert "possible_harm" in _categories(result)
    assert expected_span in _spans(result)
    assert not all(item.complete for item in result.canonical.source_coverage)


@pytest.mark.parametrize(
    "source",
    (
        "The redesign left users unable to locate the settings panel.",
        "The restructuring left managers unable to approve invoices.",
        "The closure left residents dependent on the evening shuttle.",
        "The announcement left the committee surprised.",
        "The delay left two customers waiting outside.",
        "The outage left the router running on its battery pack.",
    ),
)
def test_result_state_harmless_and_nonhuman_controls_remain_controls(source: str) -> None:
    result = SemanticExtractionCoordinator().inspect_local(source)

    assert "possible_harm" not in _categories(result)
    assert not adjudicate_human_harm(result.canonical).literal_harm


@pytest.mark.parametrize(
    "source",
    (
        "The fumes did not leave any worker seeing halos.",
        "The fumes might leave workers seeing halos.",
        "If exposure leaves a worker seeing halos, responders will assist.",
        "Officials feared the leak would leave a worker speaking incoherently.",
    ),
)
def test_unasserted_result_complements_do_not_become_asserted_harm(source: str) -> None:
    result = SemanticExtractionCoordinator().inspect_local(source)

    assert "possible_harm" not in _categories(result)
    assert not any(event.explicit_outcome for event in result.canonical.source_events)


def test_complement_internal_without_does_not_negate_asserted_impairment() -> None:
    source = "The fall left one operator unable to walk without assistance."
    frames = extract_result_complement_frames(segment_clauses(source))
    result = SemanticExtractionCoordinator().inspect_local(source)

    assert frames[0].assertion_scope == "asserted"
    assert frames[0].complement_span == "unable to walk without assistance"
    assert "possible_harm" in _categories(result)
    assert "unable to walk without assistance" in _spans(result)


def test_reported_result_state_remains_attributed_and_material() -> None:
    source = "Clinicians said the exposure left two workers seeing halos."
    result = SemanticExtractionCoordinator().inspect_local(source)

    assert "possible_harm" in _categories(result)
    assert result.escalation.required
    assert any(
        entity.entity_type == "group_of_people" for entity in result.canonical.source_entities
    )


@pytest.mark.parametrize(
    ("reply", "function"),
    (
        ("the clinic has not released the maintenance record", "plain_observation"),
        ("the clinic released the maintenance record", "plain_observation"),
        ("the cause has not been verified", "qualification"),
        ("the reported condition requires verification before conclusions", "qualification"),
        ("why was the backup system unavailable?", "question"),
        ("the incident requires an independent investigation", "direct_response"),
    ),
)
def test_sensitive_source_literal_communicative_controls_are_high_confidence(
    reply: str,
    function: str,
) -> None:
    source = "A clinic reported six deaths after the backup generator failed."
    result = SemanticExtractionCoordinator().inspect_local(source, reply)
    intent = _intent(result)

    assert result.canonical.reply_communicative_function == function
    assert result.canonical.reply_communicative_function_confidence == "high"
    assert "HUMOR_INTENT_UNCERTAIN" not in intent.issue_codes
    assert not intent.humor_safety_required
    assert adjudicate_reply_shell(result.canonical).operator == "none"


def test_possible_harm_plus_incomplete_reply_fails_closed_for_candidate() -> None:
    result = SemanticExtractionCoordinator().inspect_local(
        "The impact left one cyclist seeing halos.",
        "the impact merits a velvet banner",
    )
    intent = _intent(result)

    assert "possible_harm" in _categories(result)
    assert "unknown_predicate" in _categories(result, source=False)
    assert "HUMOR_INTENT_UNCERTAIN" in intent.issue_codes
    assert intent.humor_safety_required


@pytest.mark.asyncio
async def test_provider_none_does_not_erase_sensitive_source_reply_uncertainty() -> None:
    source = "The impact left one cyclist seeing halos."
    reply = "the impact merits a velvet banner"
    local = SemanticExtractionCoordinator().inspect_local(source, reply)
    assert local.escalation.required

    @dataclass
    class Provider:
        requests: list[SemanticProviderRequest]

        async def extract_semantics(
            self, request: SemanticProviderRequest
        ) -> SemanticProviderProposal:
            self.requests.append(request)
            return SemanticProviderProposal()

    provider = Provider([])
    result = await SemanticExtractionCoordinator(provider).inspect(source, reply)

    assert provider.requests
    assert "HUMOR_INTENT_UNCERTAIN" in _intent(result).issue_codes
    assert result.unresolved_after_reconciliation


@pytest.mark.asyncio
async def test_valid_provider_can_resolve_gerund_result_state() -> None:
    source = "The impact left one cyclist seeing halos."
    local = SemanticExtractionCoordinator().inspect_local(source)
    item = next(
        item
        for coverage in local.deterministic.source_coverage
        for item in coverage.unresolved_items
        if item.category == "possible_harm"
    )
    affected = SemanticEntity(
        clause_id=item.clause_id,
        text_span="one cyclist",
        entity_type="person",
        confidence="high",
    )
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            source_entities=(affected,),
            source_events=(
                SemanticEvent(
                    clause_id=item.clause_id,
                    event_id="P1",
                    predicate="seeing halos",
                    affected_entity=affected,
                    event_type="serious_injury",
                    polarity="negative",
                    explicit_outcome=True,
                    assertion_state="asserted",
                    evidence_spans=("seeing halos",),
                ),
            ),
        ),
        coverage_resolutions=(
            SemanticCoverageResolution(
                item_id=item.item_id,
                side=item.side,
                clause_id=item.clause_id,
                clause_span=item.clause_span,
                unresolved_span=item.unresolved_span,
                category=item.category,
                disposition="typed_relation",
                evidence_spans=(item.unresolved_span,),
            ),
        ),
    )

    @dataclass
    class Provider:
        proposal: SemanticProviderProposal
        requests: list[SemanticProviderRequest]

        async def extract_semantics(
            self, request: SemanticProviderRequest
        ) -> SemanticProviderProposal:
            self.requests.append(request)
            return self.proposal

    provider = Provider(proposal, [])
    result = await SemanticExtractionCoordinator(provider).inspect(source)

    assert result.provider_validation_errors == ()
    assert result.provider_used
    assert adjudicate_human_harm(result.canonical).harm_state == "serious_injury"


def test_subordinate_predicate_cannot_borrow_matrix_numeral() -> None:
    source = "A clinic reported six deaths after the backup generator failed."
    result = SemanticExtractionCoordinator().inspect_local(source)

    failed_events = [
        event for event in result.canonical.source_events if event.predicate == "failed"
    ]
    assert all(
        event.affected_entity is None or event.affected_entity.text_span.casefold() != "six"
        for event in failed_events
    )
    assert any(event.event_type == "death" for event in result.canonical.source_events)
    assert len(result.canonical.source_coverage) == 2


@pytest.mark.parametrize("connector", ("after", "before", "because"))
def test_subordinate_scope_never_borrows_matrix_death_count(connector: str) -> None:
    source = f"A clinic reported six deaths {connector} the backup generator failed."
    result = SemanticExtractionCoordinator().inspect_local(source)

    assert all(
        event.affected_entity is None or event.affected_entity.text_span.casefold() != "six"
        for event in result.canonical.source_events
        if event.predicate == "failed"
    )


@pytest.mark.asyncio
async def test_provider_can_validate_literal_function_for_incomplete_sensitive_reply() -> None:
    source = "The impact left one cyclist seeing halos."
    reply = "the maintenance record has yet to surface"
    local = SemanticExtractionCoordinator().inspect_local(source, reply)
    item = next(
        unresolved
        for coverage in local.deterministic.reply_coverage
        for unresolved in coverage.unresolved_items
        if unresolved.category in {"unknown_predicate", "possible_reduction"}
    )
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            reply_communicative_function="plain_observation",
            reply_communicative_function_confidence="high",
            reply_communicative_evidence_spans=(reply,),
        ),
        coverage_resolutions=(
            SemanticCoverageResolution(
                item_id=item.item_id,
                side=item.side,
                clause_id=item.clause_id,
                clause_span=item.clause_span,
                unresolved_span=item.unresolved_span,
                category=item.category,
                disposition="harmless",
                reason="literal source-status observation with no evaluative relation",
                evidence_spans=(item.unresolved_span,),
            ),
        ),
    )

    @dataclass
    class Provider:
        proposal: SemanticProviderProposal

        async def extract_semantics(
            self,
            request: SemanticProviderRequest,
        ) -> SemanticProviderProposal:
            assert request.reply_clauses == (reply,)
            return self.proposal

    result = await SemanticExtractionCoordinator(Provider(proposal)).inspect(source, reply)

    assert result.provider_validation_errors == ()
    assert result.canonical.reply_communicative_function == "plain_observation"
    assert result.canonical.reply_communicative_function_confidence == "high"
    assert "HUMOR_INTENT_UNCERTAIN" not in _intent(result).issue_codes


def test_with_state_coordination_preserves_the_complete_complement_role() -> None:
    source = "The impact left one cyclist with impaired coordination and blurred vision."
    frames = extract_result_complement_frames(segment_clauses(source))
    result = SemanticExtractionCoordinator().inspect_local(source)

    assert frames[0].affected_entity_span == "one cyclist"
    assert frames[0].complement_span == "impaired coordination and blurred vision"
    assert frames[0].complement_form == "with_state"
    assert "impaired coordination and blurred vision" in _spans(result)


def test_never_been_evidential_passive_is_a_qualification_not_a_shell() -> None:
    source = "The impact left one cyclist seeing halos."
    reply = "the cause has never been established"
    result = SemanticExtractionCoordinator().inspect_local(source, reply)

    assert result.canonical.reply_communicative_function == "qualification"
    assert result.canonical.reply_communicative_function_confidence == "high"
    assert adjudicate_reply_shell(result.canonical).operator == "none"
    assert "HUMOR_INTENT_UNCERTAIN" not in _intent(result).issue_codes
