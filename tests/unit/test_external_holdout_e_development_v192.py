from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.schemas.content import (
    EvaluationRelation,
    SemanticCoverageResolution,
    SemanticEntity,
    SemanticInspection,
    SemanticProviderProposal,
    SemanticProviderRequest,
)
from app.services.semantic_adjudication import (
    adjudicate_human_harm,
    adjudicate_humor_intent,
    adjudicate_reply_shell,
)
from app.services.semantic_extraction_coordinator import SemanticExtractionCoordinator


def _categories(result, *, source: bool) -> set[str]:
    coverage = result.canonical.source_coverage if source else result.canonical.reply_coverage
    return {category for item in coverage for category in item.unresolved_categories}


def _spans(result, *, source: bool) -> set[str]:
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
    ("source", "expected_span"),
    (
        ("The impact left one cyclist seeing double.", "seeing double"),
        ("The vapor release left two technicians slurring their words.", "slurring their words"),
        (
            "The exposure left three workers unable to remember where they were.",
            "unable to remember where they were",
        ),
        (
            "The smoke left one resident requiring supplemental oxygen.",
            "requiring supplemental oxygen",
        ),
        (
            "The fall left one operator unable to walk without support.",
            "unable to walk without support",
        ),
        (
            "The solvent left two mechanics with numbness in both hands.",
            "numbness in both hands",
        ),
    ),
)
def test_holdout_e_material_result_states_are_development_regressions(
    source: str,
    expected_span: str,
) -> None:
    result = SemanticExtractionCoordinator().inspect_local(
        source,
        "excellent work from the incident",
    )

    assert "possible_harm" in _categories(result, source=True)
    assert expected_span in _spans(result, source=True)
    assert "HUMOR_INTENT_UNCERTAIN" in _intent(result).issue_codes


@pytest.mark.parametrize(
    "source",
    (
        "The reorganization left two managers unable to approve invoices.",
        "The redesign left users unable to locate the preferences panel.",
        "The announcement left the board stunned.",
        "The closure left residents dependent on the morning ferry.",
        "The outage left the router running on its battery pack.",
    ),
)
def test_holdout_e_harmless_result_controls_do_not_become_human_harm(source: str) -> None:
    result = SemanticExtractionCoordinator().inspect_local(
        source, "this is a plain operational note"
    )

    assert "possible_harm" not in _categories(result, source=True)
    assert not adjudicate_human_harm(result.canonical).literal_harm
    assert "HUMOR_INTENT_UNCERTAIN" not in _intent(result).issue_codes


@pytest.mark.parametrize(
    "source",
    (
        "The fumes did not leave any worker unable to focus.",
        "The fumes might leave workers unable to coordinate their movements.",
        "If exposure leaves a technician unable to see clearly, the team will call an ambulance.",
    ),
)
def test_holdout_e_unasserted_states_remain_nonliteral(source: str) -> None:
    result = SemanticExtractionCoordinator().inspect_local(source)

    assert "possible_harm" not in _categories(result, source=True)
    assert not adjudicate_human_harm(result.canonical).literal_harm


def test_holdout_e_reported_state_remains_material() -> None:
    result = SemanticExtractionCoordinator().inspect_local(
        "Doctors said the fumes left two workers unable to see clearly.",
        "excellent work from the ventilation system",
    )

    assert "possible_harm" in _categories(result, source=True)
    assert result.escalation.required
    assert "HUMOR_INTENT_UNCERTAIN" in _intent(result).issue_codes


@pytest.mark.parametrize(
    ("reply", "communicative_function"),
    (
        ("the clinic has not published the maintenance log", "plain_observation"),
        ("why was the backup generator unavailable?", "question"),
        ("the cause has not been confirmed", "qualification"),
    ),
)
def test_holdout_e_fatal_source_literal_controls_remain_eligible(
    reply: str,
    communicative_function: str,
) -> None:
    result = SemanticExtractionCoordinator().inspect_local(
        "A clinic reported six deaths after the backup generator failed.",
        reply,
    )

    assert result.canonical.reply_communicative_function == communicative_function
    assert result.canonical.reply_communicative_function_confidence == "high"
    assert "HUMOR_INTENT_UNCERTAIN" not in _intent(result).issue_codes


def test_holdout_e_unknown_ceremonial_reply_remains_candidate_local_uncertainty() -> None:
    result = SemanticExtractionCoordinator().inspect_local(
        "A clinic reported six deaths after the backup generator failed.",
        "the generator gets its own brass plaque",
    )

    assert "HUMOR_INTENT_UNCERTAIN" in _intent(result).issue_codes
    assert _intent(result).humor_safety_required


def test_holdout_e_clause_scope_and_mixed_events_remain_separate() -> None:
    result = SemanticExtractionCoordinator().inspect_local(
        "The routing worker crashed, and two residents later died from smoke inhalation.",
        "the incident requires a full investigation",
    )
    event_types = {event.event_type for event in result.canonical.source_events}

    assert "death" in event_types
    assert adjudicate_human_harm(result.canonical).harm_referent == "human"
    crashed = [event for event in result.canonical.source_events if event.predicate == "crashed"]
    assert crashed
    assert all(event.clause_id == "C1" for event in crashed)
    assert all(
        event.affected_entity is None
        or "resident" not in event.affected_entity.text_span.casefold()
        for event in crashed
    )


def test_holdout_e_subordinate_failure_does_not_borrow_death_count() -> None:
    result = SemanticExtractionCoordinator().inspect_local(
        "A clinic reported six deaths after the backup generator failed."
    )

    assert all(
        event.affected_entity is None or event.affected_entity.text_span.casefold() != "six"
        for event in result.canonical.source_events
        if event.predicate == "failed"
    )


@pytest.mark.parametrize(
    "reply",
    (
        "onboarding is paperwork behind a showroom facade",
        "onboarding is bureaucracy behind a showroom facade",
    ),
)
def test_holdout_e_packaging_expectations_are_migrated(reply: str) -> None:
    result = SemanticExtractionCoordinator().inspect_local(
        "A startup published a revised onboarding flow.",
        reply,
    )

    assert adjudicate_reply_shell(result.canonical).operator == "decorated_identity"


def test_holdout_e_literal_formula_control_remains_non_shell() -> None:
    result = SemanticExtractionCoordinator().inspect_local(
        "",
        "error rate is failed requests divided by total requests",
    )

    assert adjudicate_reply_shell(result.canonical).operator == "none"


@pytest.mark.asyncio
async def test_holdout_e_provider_evaluation_uses_current_request_item() -> None:
    source = "A clinic reported six deaths after the backup generator failed."
    reply = "the generator gets its own brass plaque"
    local = SemanticExtractionCoordinator().inspect_local(source, reply)
    item = next(
        unresolved
        for coverage in local.deterministic.reply_coverage
        for unresolved in coverage.unresolved_items
        if unresolved.category == "unknown_predicate"
    )
    target = SemanticEntity(
        clause_id=item.clause_id,
        text_span="generator",
        entity_type="technical_system",
        confidence="high",
    )
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            reply_entities=(target,),
            evaluations=(
                EvaluationRelation(
                    clause_id=item.clause_id,
                    target_entity=target,
                    evaluation_type="reward",
                    relation_subtype="reward",
                    polarity="positive",
                    literal_reading_plausible=False,
                    confidence="high",
                    evidence_spans=(reply,),
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
                evidence_spans=(reply,),
            ),
        ),
    )

    @dataclass
    class Provider:
        proposal: SemanticProviderProposal
        requests: list[SemanticProviderRequest]

        async def extract_semantics(
            self,
            request: SemanticProviderRequest,
        ) -> SemanticProviderProposal:
            self.requests.append(request)
            return self.proposal

    provider = Provider(proposal, [])
    result = await SemanticExtractionCoordinator(provider).inspect(source, reply)

    assert result.provider_used
    assert result.provider_validation_errors == ()
    assert (
        "HUMOR_ON_TRAGEDY" in _intent(result).issue_codes or _intent(result).humor_safety_required
    )


def test_holdout_e_technical_clause_does_not_capture_later_human_result_state() -> None:
    result = SemanticExtractionCoordinator().inspect_local(
        "The backup controller died, while the refrigerant leak left three engineers "
        "unable to follow a conversation.",
        "the leak handled that beautifully",
    )

    assert "possible_harm" in _categories(result, source=True)
    assert all(
        event.affected_entity is None
        or "engineer" not in event.affected_entity.text_span.casefold()
        for event in result.canonical.source_events
        if event.predicate == "died"
    )
    assert "HUMOR_INTENT_UNCERTAIN" in _intent(result).issue_codes


@pytest.mark.parametrize(
    ("source", "reply"),
    (
        (
            "The cache daemon died; two clinicians completed the evening handoff.",
            "the daemon needs a restart",
        ),
        (
            "The compiler died after the patch. The nurse filed the incident report.",
            "the compiler needs a rollback",
        ),
    ),
)
def test_holdout_e_unknown_technical_heads_do_not_become_human_deaths(
    source: str,
    reply: str,
) -> None:
    result = SemanticExtractionCoordinator().inspect_local(source, reply)
    harm = adjudicate_human_harm(result.canonical)

    assert not harm.literal_harm
    assert harm.harm_referent != "human"
    assert "HUMOR_INTENT_UNCERTAIN" not in _intent(result).issue_codes
