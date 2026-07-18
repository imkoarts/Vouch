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
    adjudicate_humor_safety,
)
from app.services.semantic_composition import (
    extract_result_state_candidates,
    inspect_result_state_materiality,
    segment_clauses,
)
from app.services.semantic_extraction_coordinator import SemanticExtractionCoordinator
from app.services.semantic_provider_validation import validate_span


def _categories(result, *, source: bool = True) -> set[str]:
    coverage = result.canonical.source_coverage if source else result.canonical.reply_coverage
    return {category for item in coverage for category in item.unresolved_categories}


def _intent(result):
    return adjudicate_humor_intent(
        result.canonical,
        metadata_says_humor=False,
        reaction_type="none",
        template_humor=False,
    )


@pytest.mark.parametrize(
    "source",
    (
        "The delayed ferry left three riders waiting beneath a shelter.",
        "The update left two operators switching between dashboards.",
        "The schedule change left four attendees standing near the entrance.",
        "The redesign left customers writing notes in the sidebar.",
    ),
)
def test_human_ordinary_activity_is_not_possible_harm_or_material_escalation(source: str) -> None:
    result = SemanticExtractionCoordinator().inspect_local(source)

    assert "possible_harm" not in _categories(result)
    assert not result.escalation.required
    assert not adjudicate_human_harm(result.canonical).literal_harm
    assert result.canonical.result_state_materiality
    assert all(not item.safety_material for item in result.canonical.result_state_materiality)


@pytest.mark.parametrize(
    ("source", "expected_relation"),
    (
        ("The impact left two riders unable to keep their balance.", "inability"),
        ("The fumes left a technician with reduced vision.", "loss_or_reduction"),
        ("The exposure left three workers needing assistance to breathe.", "assistance_need"),
        ("The incident left one resident dependent on breathing support.", "support_dependency"),
        ("The blast left two operators speaking incoherently.", "abnormal_physiological_state"),
    ),
)
def test_bound_impairment_relation_drives_possible_harm(
    source: str, expected_relation: str
) -> None:
    frames = segment_clauses(source)
    candidates = extract_result_state_candidates(frames)
    materiality = tuple(
        inspect_result_state_materiality(
            next(frame for frame in frames if frame.clause_id == candidate.clause_id), candidate
        )
        for candidate in candidates
    )
    result = SemanticExtractionCoordinator().inspect_local(source)

    assert any(item.impairment_relation == expected_relation for item in materiality)
    assert any(item.safety_material for item in materiality)
    assert "possible_harm" in _categories(result)
    assert result.escalation.required


def test_bare_target_fragment_is_not_a_plain_proposition_on_fatal_source() -> None:
    result = SemanticExtractionCoordinator().inspect_local(
        "A clinic reported five deaths after the cooling system failed.",
        "a ceremonial badge for the control unit",
    )
    intent = _intent(result)

    assert result.canonical.reply_communicative_function == "unknown"
    assert result.canonical.reply_propositions
    assert not result.canonical.reply_propositions[0].proposition_complete
    assert "possible_evaluation" in _categories(result, source=False)
    assert "HUMOR_INTENT_UNCERTAIN" in intent.issue_codes
    assert intent.humor_safety_required


def test_possible_harm_source_is_sensitive_for_confident_evaluation() -> None:
    result = SemanticExtractionCoordinator().inspect_local(
        "The impact left two riders unable to keep their balance.",
        "what a triumph for the response team",
    )
    intent = _intent(result)
    safety = adjudicate_humor_safety(result.canonical)

    assert "possible_harm" in _categories(result)
    assert intent.detected_reaction_type in {"sarcasm", "uncertain"}
    assert intent.humor_safety_required
    assert safety.sensitive_context
    assert not safety.suitable_for_humor
    assert "HUMOR_ON_TRAGEDY" in safety.issue_codes


@pytest.mark.parametrize(
    ("reply", "function"),
    (
        ("the failure sequence remains under examination", "qualification"),
        ("the outage cause is under independent review", "qualification"),
        ("the control unit needs an external audit", "direct_response"),
        ("the pipeline requires an independent inspection", "direct_response"),
    ),
)
def test_structural_literal_controls_precede_shell_or_unknown_coverage(
    reply: str, function: str
) -> None:
    result = SemanticExtractionCoordinator().inspect_local(
        "A clinic reported five deaths after the cooling system failed.", reply
    )

    assert result.canonical.reply_communicative_function == function
    assert result.canonical.reply_communicative_function_confidence == "high"
    assert result.canonical.reply_shells == ()
    assert not _categories(result, source=False)
    assert "HUMOR_INTENT_UNCERTAIN" not in _intent(result).issue_codes


@pytest.mark.parametrize(
    ("span", "clause", "valid"),
    (
        ("ringing in both ears", "workers reported ringing in both ears", True),
        ("ringing in both ear", "workers reported ringing in both ears", False),
        ("both ears", "workers reported ringing in both ears", True),
        ("ears", "workers reported ringing in both ears", True),
        ("ear", "workers reported ringing in both ears", False),
        ("échec", "l'échec was documented", True),
        ("chec", "l'échec was documented", False),
        ("failure,", "the failure, then recovery", True),
    ),
)
def test_provider_span_validation_requires_unicode_token_boundaries(
    span: str, clause: str, valid: bool
) -> None:
    assert (validate_span(span, clause) is not None) is valid


@pytest.mark.asyncio
async def test_partial_token_provider_proposal_cannot_resolve_possible_harm() -> None:
    source = "The impact left two riders with ringing in both ears."
    local = SemanticExtractionCoordinator().inspect_local(source)
    item = next(
        unresolved
        for coverage in local.deterministic.source_coverage
        for unresolved in coverage.unresolved_items
        if unresolved.category == "possible_harm"
    )
    affected = SemanticEntity(
        clause_id=item.clause_id,
        text_span="two riders",
        entity_type="group_of_people",
        confidence="high",
    )
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            source_entities=(affected,),
            source_events=(
                SemanticEvent(
                    clause_id=item.clause_id,
                    event_id="P1",
                    predicate="ringing in both ear",
                    affected_entity=affected,
                    event_type="serious_injury",
                    polarity="negative",
                    explicit_outcome=True,
                    assertion_state="asserted",
                    evidence_spans=("ringing in both ear",),
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
                evidence_spans=("ringing in both ear",),
            ),
        ),
    )

    @dataclass
    class Provider:
        proposal: SemanticProviderProposal

        async def extract_semantics(
            self, request: SemanticProviderRequest
        ) -> SemanticProviderProposal:
            return self.proposal

    result = await SemanticExtractionCoordinator(Provider(proposal)).inspect(source)

    assert result.provider_validation_errors
    assert any("not exact" in error for error in result.provider_validation_errors)
    assert "possible_harm" in {
        category
        for coverage in result.canonical.source_coverage
        for category in coverage.unresolved_categories
    }


@pytest.mark.parametrize("connector", ("After", "Before", "Because", "While", "Following"))
def test_leading_subordinate_scope_isolated_from_matrix_human_outcome(connector: str) -> None:
    source = f"{connector} the router failed, the clinic reported five deaths."
    result = SemanticExtractionCoordinator().inspect_local(source)

    failure = next(event for event in result.canonical.source_events if event.predicate == "failed")
    death = next(event for event in result.canonical.source_events if event.event_type == "death")
    assert failure.clause_id != death.clause_id
    assert (
        failure.affected_entity is None
        or "clinic" not in failure.affected_entity.text_span.casefold()
    )
    assert (
        failure.affected_entity is None
        or "five" not in failure.affected_entity.text_span.casefold()
    )
    assert death.affected_entity is not None


def test_provider_none_cannot_make_bare_fragment_safe_on_possible_harm_source() -> None:
    result = SemanticExtractionCoordinator().inspect_local(
        "The impact left two riders unable to keep their balance.",
        "a ceremonial marker for the response unit",
    )

    assert "possible_harm" in _categories(result)
    assert "possible_evaluation" in _categories(result, source=False)
    assert "HUMOR_INTENT_UNCERTAIN" in _intent(result).issue_codes


@pytest.mark.asyncio
async def test_exact_token_subspan_cannot_resolve_a_longer_material_role() -> None:
    source = "The impact left two riders with ringing in both ears."
    local = SemanticExtractionCoordinator().inspect_local(source)
    item = next(
        unresolved
        for coverage in local.deterministic.source_coverage
        for unresolved in coverage.unresolved_items
        if unresolved.category == "possible_harm"
        and unresolved.unresolved_span == "ringing in both ears"
    )
    affected = SemanticEntity(
        clause_id=item.clause_id,
        text_span="two riders",
        entity_type="group_of_people",
        confidence="high",
    )
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            source_entities=(affected,),
            source_events=(
                SemanticEvent(
                    clause_id=item.clause_id,
                    event_id="P-role",
                    predicate="ringing",
                    affected_entity=affected,
                    event_type="serious_injury",
                    polarity="negative",
                    explicit_outcome=True,
                    assertion_state="asserted",
                    evidence_spans=("ringing in both ears",),
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
                evidence_spans=("ears",),
            ),
        ),
    )

    @dataclass
    class Provider:
        proposal: SemanticProviderProposal

        async def extract_semantics(
            self, request: SemanticProviderRequest
        ) -> SemanticProviderProposal:
            return self.proposal

    result = await SemanticExtractionCoordinator(Provider(proposal)).inspect(source)

    assert any("complete unresolved role" in error for error in result.provider_validation_errors)
    assert "possible_harm" in _categories(result)


def test_causal_normative_response_is_complete_literal_control_on_sensitive_source() -> None:
    result = SemanticExtractionCoordinator().inspect_local(
        "Six workers died after the collision.",
        "those losses render a victory lap unsuitable",
    )
    inspection = result.canonical

    assert inspection.reply_communicative_function == "direct_response"
    assert inspection.reply_communicative_function_confidence == "high"
    assert any(
        proposition.proposition_complete and proposition.communicative_function == "direct_response"
        for proposition in inspection.reply_propositions
    )
    assert not (_categories(result, source=False) & {"possible_evaluation", "unknown_predicate"})
