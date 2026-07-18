from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.schemas.content import (
    EvaluationRelation,
    SemanticCoverageResolution,
    SemanticEntity,
    SemanticEvent,
    SemanticInspection,
    SemanticProviderProposal,
    SemanticProviderRequest,
    SemanticReplyShell,
)
from app.services.semantic_adjudication import (
    adjudicate_evaluative_irony,
    adjudicate_human_harm,
    adjudicate_humor_intent,
    adjudicate_reply_shell,
)
from app.services.semantic_extraction import extract_semantic_inspection
from app.services.semantic_extraction_coordinator import SemanticExtractionCoordinator


@dataclass
class FakeSemanticProvider:
    proposal: SemanticProviderProposal
    calls: list[SemanticProviderRequest]

    async def extract_semantics(self, request: SemanticProviderRequest) -> SemanticProviderProposal:
        self.calls.append(request)
        return self.proposal


def _coverage_resolutions(
    inspection: SemanticInspection,
    *,
    side: str,
    category: str,
    disposition: str,
    evidence_spans: tuple[str, ...],
    reason: str = "",
) -> tuple[SemanticCoverageResolution, ...]:
    coverage_items = inspection.source_coverage if side == "source" else inspection.reply_coverage
    return tuple(
        SemanticCoverageResolution(
            item_id=item.item_id,
            side=side,
            clause_id=item.clause_id,
            clause_span=item.clause_span,
            unresolved_span=item.unresolved_span,
            category=category,
            disposition=disposition,
            reason=reason,
            evidence_spans=evidence_spans,
        )
        for coverage in coverage_items
        for item in coverage.unresolved_items
        if item.category == category
    )


def test_literal_recovery_is_positive_but_recovering_bodies_is_death() -> None:
    recovery = extract_semantic_inspection(
        "The clinic restored backup power and every patient recovered safely."
    )
    bodies = extract_semantic_inspection("The bodies were recovered after the crash.")

    assert "death" not in {event.event_type for event in recovery.source_events}
    assert "success" in {event.event_type for event in recovery.source_events}
    assert adjudicate_human_harm(recovery).literal_harm is False
    assert "death" in {event.event_type for event in bodies.source_events}
    assert adjudicate_human_harm(bodies).literal_harm is True


def test_unknown_celebratory_and_packaging_material_is_not_silently_complete() -> None:
    evaluation = extract_semantic_inspection(
        "A clinic reported eight deaths after the blackout.",
        "a ceremonial sash for the grid",
    )
    packaging = extract_semantic_inspection("", "onboarding is paperwork in a velvet sleeve")

    assert evaluation.reply_coverage[0].complete is False
    assert "possible_evaluation" in evaluation.reply_coverage[0].unresolved_categories
    assert packaging.reply_coverage[0].complete is False
    assert "possible_reduction" in packaging.reply_coverage[0].unresolved_categories


def test_consumed_failure_with_unknown_affected_role_requires_escalation() -> None:
    coordinator = SemanticExtractionCoordinator()
    result = coordinator.inspect_local("The routing daemon died overnight.")

    assert result.escalation.required
    assert "unknown_referent" in result.escalation.material_categories
    assert result.unresolved_after_reconciliation
    assert adjudicate_human_harm(result.canonical).harm_referent == "unknown"


@pytest.mark.asyncio
async def test_valid_exact_span_provider_resolves_possible_harm() -> None:
    source = "The blast left two residents medically unstable."
    local = SemanticExtractionCoordinator().inspect_local(source)
    residents = SemanticEntity(
        text_span="residents",
        canonical_name="residents",
        entity_type="group_of_people",
        confidence="high",
    )
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            source_entities=(residents,),
            source_events=(
                SemanticEvent(
                    event_id="P1",
                    predicate="unstable",
                    affected_entity=residents,
                    event_type="critical_condition",
                    polarity="negative",
                    explicit_outcome=True,
                    assertion_state="asserted",
                    evidence_spans=("unstable",),
                ),
            ),
        ),
        coverage_resolutions=_coverage_resolutions(
            local.deterministic,
            side="source",
            category="possible_harm",
            disposition="typed_relation",
            evidence_spans=("unstable",),
        ),
    )
    fake = FakeSemanticProvider(proposal, [])

    result = await SemanticExtractionCoordinator(fake).inspect(source)

    assert result.provider_used
    assert result.provider_validation_errors == ()
    assert adjudicate_human_harm(result.canonical).harm_state == "critical_condition"
    assert "possible_harm" not in {
        category
        for item in result.canonical.source_coverage
        for category in item.unresolved_categories
    }
    assert len(fake.calls) == 1
    assert fake.calls[0].source_clauses == (source,)


@pytest.mark.asyncio
async def test_valid_provider_resolves_unknown_nonhuman_referent() -> None:
    source = "The routing daemon died overnight."
    local = SemanticExtractionCoordinator().inspect_local(source)
    daemon = SemanticEntity(
        text_span="The routing daemon",
        canonical_name="routing daemon",
        entity_type="software_process",
        confidence="high",
    )
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            source_entities=(daemon,),
            source_events=(
                SemanticEvent(
                    event_id="P1",
                    predicate="died",
                    affected_entity=daemon,
                    event_type="technical_failure",
                    polarity="negative",
                    explicit_outcome=False,
                    evidence_spans=("died",),
                ),
            ),
        ),
        coverage_resolutions=_coverage_resolutions(
            local.deterministic,
            side="source",
            category="unknown_referent",
            disposition="typed_relation",
            evidence_spans=("The routing daemon", "died"),
        ),
    )

    result = await SemanticExtractionCoordinator(FakeSemanticProvider(proposal, [])).inspect(source)

    harm = adjudicate_human_harm(result.canonical)
    assert result.provider_validation_errors == ()
    assert harm.harm_referent == "non_human"
    assert harm.harm_state == "non_human_failure"
    assert harm.literal_harm is False


@pytest.mark.asyncio
async def test_valid_provider_resolves_evaluative_irony() -> None:
    source = "A clinic reported eight deaths after the blackout."
    reply = "a ceremonial sash for the grid"
    local = SemanticExtractionCoordinator().inspect_local(source, reply)
    harmful_event = next(
        event for event in local.deterministic.source_events if event.event_type == "death"
    )
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            evaluations=(
                EvaluationRelation(
                    target_event_id=harmful_event.event_id,
                    evaluation_type="reward",
                    relation_subtype="reward",
                    polarity="positive",
                    literal_reading_plausible=False,
                    confidence="high",
                    evidence_spans=(reply,),
                ),
            ),
        ),
        coverage_resolutions=_coverage_resolutions(
            local.deterministic,
            side="reply",
            category="possible_evaluation",
            disposition="typed_relation",
            evidence_spans=(reply,),
        ),
    )

    result = await SemanticExtractionCoordinator(FakeSemanticProvider(proposal, [])).inspect(
        source, reply
    )

    irony = adjudicate_evaluative_irony(result.canonical)
    intent = adjudicate_humor_intent(
        result.canonical,
        metadata_says_humor=False,
        reaction_type="none",
        template_humor=False,
    )
    assert result.provider_validation_errors == ()
    assert irony.implicit_sarcasm
    assert intent.text_appears_humorous
    assert intent.humor_safety_required


@pytest.mark.asyncio
async def test_valid_provider_refines_identity_into_decorated_shell() -> None:
    reply = "onboarding is paperwork in a velvet sleeve"
    local = SemanticExtractionCoordinator().inspect_local("", reply)
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            reply_shells=(
                SemanticReplyShell(
                    operator="decorated_identity",
                    subject_span="onboarding",
                    complement_span="paperwork",
                    decoration_span="in a velvet sleeve",
                    generic_reusability=True,
                    evidence_spans=(reply,),
                ),
            ),
        ),
        coverage_resolutions=_coverage_resolutions(
            local.deterministic,
            side="reply",
            category="possible_reduction",
            disposition="typed_relation",
            evidence_spans=(reply,),
        ),
    )

    result = await SemanticExtractionCoordinator(FakeSemanticProvider(proposal, [])).inspect(
        "", reply
    )

    shell = adjudicate_reply_shell(result.canonical)
    assert result.provider_validation_errors == ()
    assert shell.operator == "decorated_identity"
    assert shell.decoration_span == "in a velvet sleeve"
    assert result.canonical.reply_coverage[0].complete


@pytest.mark.asyncio
async def test_invalid_provider_span_rejects_entire_proposal() -> None:
    source = "The blast left two residents medically unstable."
    local = SemanticExtractionCoordinator().inspect_local(source)
    residents = SemanticEntity(
        text_span="residents",
        entity_type="group_of_people",
        confidence="high",
    )
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            source_entities=(residents,),
            source_events=(
                SemanticEvent(
                    event_id="P1",
                    predicate="unstable",
                    affected_entity=residents,
                    event_type="critical_condition",
                    polarity="negative",
                    explicit_outcome=True,
                    evidence_spans=("critically unstable",),
                ),
            ),
        ),
        coverage_resolutions=_coverage_resolutions(
            local.deterministic,
            side="source",
            category="possible_harm",
            disposition="typed_relation",
            evidence_spans=("unstable",),
        ),
    )

    result = await SemanticExtractionCoordinator(FakeSemanticProvider(proposal, [])).inspect(source)

    assert result.provider_validation_errors
    assert result.canonical == result.deterministic
    assert "possible_harm" in result.escalation.material_categories


@pytest.mark.asyncio
async def test_provider_relation_referencing_missing_entity_is_rejected() -> None:
    source = "The routing daemon died overnight."
    local = SemanticExtractionCoordinator().inspect_local(source)
    missing = SemanticEntity(
        text_span="The routing daemon",
        entity_type="software_process",
        confidence="high",
    )
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            source_events=(
                SemanticEvent(
                    event_id="P1",
                    predicate="died",
                    affected_entity=missing,
                    event_type="technical_failure",
                    polarity="negative",
                    evidence_spans=("died",),
                ),
            ),
        ),
        coverage_resolutions=_coverage_resolutions(
            local.deterministic,
            side="source",
            category="unknown_referent",
            disposition="typed_relation",
            evidence_spans=("died",),
        ),
    )

    result = await SemanticExtractionCoordinator(FakeSemanticProvider(proposal, [])).inspect(source)

    assert any("missing affected_entity" in item for item in result.provider_validation_errors)
    assert result.canonical == result.deterministic


@pytest.mark.asyncio
async def test_provider_none_cannot_erase_uncertainty() -> None:
    source = "The routing daemon died overnight."
    result = await SemanticExtractionCoordinator(
        FakeSemanticProvider(SemanticProviderProposal(), [])
    ).inspect(source)

    assert result.provider_used
    assert result.unresolved_after_reconciliation
    assert "unknown_referent" in result.escalation.material_categories


@pytest.mark.asyncio
async def test_harmless_resolution_requires_exact_spans_and_reason() -> None:
    source = "A clinic reported eight deaths after the blackout."
    reply = "a report for the grid"
    local = SemanticExtractionCoordinator().inspect_local(source, reply)
    invalid = SemanticProviderProposal(
        coverage_resolutions=_coverage_resolutions(
            local.deterministic,
            side="reply",
            category="possible_evaluation",
            disposition="harmless",
            evidence_spans=(reply,),
            reason="",
        )
    )
    invalid_result = await SemanticExtractionCoordinator(FakeSemanticProvider(invalid, [])).inspect(
        source, reply
    )

    valid = SemanticProviderProposal(
        coverage_resolutions=_coverage_resolutions(
            local.deterministic,
            side="reply",
            category="possible_evaluation",
            disposition="harmless",
            evidence_spans=(reply,),
            reason="literal document-delivery noun phrase, not praise or reward",
        )
    )
    valid_result = await SemanticExtractionCoordinator(FakeSemanticProvider(valid, [])).inspect(
        source, reply
    )

    assert invalid_result.provider_validation_errors
    assert valid_result.provider_validation_errors == ()
    assert valid_result.canonical.reply_coverage[0].complete


@pytest.mark.asyncio
async def test_offline_mode_never_calls_provider_and_retains_candidate_level_uncertainty() -> None:
    class ExplodingProvider:
        async def extract_semantics(
            self, request: SemanticProviderRequest
        ) -> SemanticProviderProposal:
            raise AssertionError("provider must not be called")

    source = "A clinic reported eight deaths after the blackout."
    reply = "a ceremonial sash for the grid"
    result = await SemanticExtractionCoordinator(ExplodingProvider()).inspect(
        source,
        reply,
        allow_live=False,
    )
    intent = adjudicate_humor_intent(
        result.canonical,
        metadata_says_humor=False,
        reaction_type="none",
        template_humor=False,
    )

    assert not result.provider_used
    assert "possible_evaluation" in result.escalation.material_categories
    assert "HUMOR_INTENT_UNCERTAIN" in intent.issue_codes
    assert intent.humor_safety_required


@pytest.mark.asyncio
async def test_provider_packet_contains_only_material_unresolved_clauses() -> None:
    source = "The clinic restored power. The routing daemon died overnight."
    local = SemanticExtractionCoordinator().inspect_local(source)
    proposal = SemanticProviderProposal(
        remaining_unresolved_items=tuple(
            SemanticCoverageResolution(
                item_id=item.item_id,
                side="source",
                clause_id=item.clause_id,
                clause_span=item.clause_span,
                unresolved_span=item.unresolved_span,
                category=item.category,
                disposition="unresolved",
                reason="referent remains unknown",
                evidence_spans=(item.unresolved_span,),
            )
            for coverage in local.deterministic.source_coverage
            for item in coverage.unresolved_items
            if coverage.clause_span == "The routing daemon died overnight."
        )
    )
    fake = FakeSemanticProvider(proposal, [])

    result = await SemanticExtractionCoordinator(fake).inspect(source)

    assert result.provider_used
    assert len(fake.calls) == 1
    request = fake.calls[0]
    assert request.source_clauses == ("The routing daemon died overnight.",)
    assert {item.clause_span for item in request.deterministic.source_coverage} == {
        "The routing daemon died overnight."
    }
    assert "The clinic restored power." not in request.model_dump_json()


@pytest.mark.asyncio
async def test_provider_conflict_remains_explicit_and_does_not_resolve_coverage() -> None:
    source = "Two patients died while NEXUS faltered."
    local = SemanticExtractionCoordinator().inspect_local(source)
    local_event = next(
        event for event in local.deterministic.source_events if event.predicate == "died"
    )
    patients = SemanticEntity(
        text_span="Two patients",
        entity_type="technical_system",
        confidence="high",
    )
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            source_entities=(patients,),
            source_events=(
                SemanticEvent(
                    event_id="P1",
                    predicate="died",
                    affected_entity=patients,
                    event_type="technical_failure",
                    polarity="negative",
                    explicit_outcome=False,
                    assertion_state="asserted",
                    evidence_spans=local_event.evidence_spans,
                ),
            ),
        ),
        remaining_unresolved_items=tuple(
            SemanticCoverageResolution(
                item_id=item.item_id,
                side="source",
                clause_id=item.clause_id,
                clause_span=item.clause_span,
                unresolved_span=item.unresolved_span,
                category=item.category,
                disposition="unresolved",
                reason="unresolved predicate",
                evidence_spans=(item.unresolved_span,),
            )
            for coverage in local.deterministic.source_coverage
            for item in coverage.unresolved_items
        ),
    )

    result = await SemanticExtractionCoordinator(FakeSemanticProvider(proposal, [])).inspect(source)

    assert result.provider_validation_errors == ()
    assert any(
        item.startswith("provider_event_conflict:")
        for item in result.canonical.extraction_conflicts
    )
    assert result.unresolved_after_reconciliation


@pytest.mark.asyncio
async def test_typed_resolution_must_match_its_material_category() -> None:
    source = "A clinic reported eight deaths after the blackout."
    reply = "a ceremonial sash for the grid"
    local = SemanticExtractionCoordinator().inspect_local(source, reply)
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            reply_shells=(
                SemanticReplyShell(
                    operator="identity",
                    subject_span="sash",
                    complement_span="grid",
                    evidence_spans=(reply,),
                ),
            ),
        ),
        coverage_resolutions=_coverage_resolutions(
            local.deterministic,
            side="reply",
            category="possible_evaluation",
            disposition="typed_relation",
            evidence_spans=(reply,),
        ),
    )

    result = await SemanticExtractionCoordinator(FakeSemanticProvider(proposal, [])).inspect(
        source, reply
    )

    assert any(
        "category-compatible proposed relation" in error
        for error in result.provider_validation_errors
    )
    assert result.canonical == result.deterministic


@pytest.mark.asyncio
async def test_provider_event_predicate_must_be_an_exact_span() -> None:
    source = "The blast left two residents medically unstable."
    local = SemanticExtractionCoordinator().inspect_local(source)
    residents = SemanticEntity(
        text_span="residents",
        entity_type="group_of_people",
        confidence="high",
    )
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            source_entities=(residents,),
            source_events=(
                SemanticEvent(
                    event_id="P1",
                    predicate="deteriorated critically",
                    affected_entity=residents,
                    event_type="critical_condition",
                    polarity="negative",
                    explicit_outcome=True,
                    evidence_spans=("unstable",),
                ),
            ),
        ),
        coverage_resolutions=_coverage_resolutions(
            local.deterministic,
            side="source",
            category="possible_harm",
            disposition="typed_relation",
            evidence_spans=("unstable",),
        ),
    )

    result = await SemanticExtractionCoordinator(FakeSemanticProvider(proposal, [])).inspect(source)

    assert "source event predicate is not exact: 'deteriorated critically'" in (
        result.provider_validation_errors
    )


def test_semantic_provider_proposal_schema_is_strict_for_every_object_node() -> None:
    violations: list[str] = []

    def walk(value: object, path: str = "$") -> None:
        if isinstance(value, dict):
            if value.get("type") == "object" and value.get("additionalProperties") is not False:
                violations.append(path)
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(SemanticProviderProposal.model_json_schema())
    assert violations == []


@pytest.mark.asyncio
async def test_responses_provider_uses_strict_bounded_semantic_request() -> None:
    from app.providers.responses_provider import ResponsesLLMProvider

    captured: dict[str, object] = {}
    proposal = SemanticProviderProposal()

    class FakeResponses:
        async def parse(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return type("Response", (), {"output_parsed": proposal})()

    class FakeClient:
        def __init__(self) -> None:
            self.responses = FakeResponses()

        async def close(self) -> None:
            return None

    local_request = SemanticExtractionCoordinator().inspect_local(
        "The routing daemon died overnight."
    )
    unresolved_items = tuple(
        item
        for coverage in local_request.deterministic.source_coverage
        for item in coverage.unresolved_items
    )
    request = SemanticProviderRequest(
        source_clauses=("The routing daemon died overnight.",),
        request_id="test-request",
        deterministic=local_request.deterministic,
        unresolved_items=unresolved_items,
        material_categories=("unknown_referent",),
        clause_spans=("The routing daemon died overnight.",),
    )
    provider = ResponsesLLMProvider(
        name="openai",
        model="semantic-test-model",
        api_key="synthetic-key",
        client_factory=FakeClient,
    )

    result = await provider.extract_semantics(request)

    assert result == proposal
    assert captured["text_format"] is SemanticProviderProposal
    assert captured["store"] is False
    serialized = str(captured["input"])
    assert "approval" in serialized
    assert "The routing daemon died overnight." in serialized
    assert "chain of thought" not in serialized.casefold()


@pytest.mark.asyncio
async def test_possible_evaluation_resolution_requires_a_bound_target() -> None:
    source = "A clinic reported eight deaths after the blackout."
    reply = "a ceremonial sash for the grid"
    local = SemanticExtractionCoordinator().inspect_local(source, reply)
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            evaluations=(
                EvaluationRelation(
                    evaluation_type="reward",
                    relation_subtype="reward",
                    polarity="positive",
                    literal_reading_plausible=False,
                    confidence="high",
                    evidence_spans=(reply,),
                ),
            ),
        ),
        coverage_resolutions=_coverage_resolutions(
            local.deterministic,
            side="reply",
            category="possible_evaluation",
            disposition="typed_relation",
            evidence_spans=(reply,),
        ),
    )

    result = await SemanticExtractionCoordinator(FakeSemanticProvider(proposal, [])).inspect(
        source, reply
    )

    assert any(
        "category-compatible proposed relation" in error
        for error in result.provider_validation_errors
    )
    assert result.canonical == result.deterministic


@pytest.mark.asyncio
async def test_possible_reduction_resolution_requires_bound_subject_and_complement() -> None:
    reply = "onboarding is paperwork in a velvet sleeve"
    local = SemanticExtractionCoordinator().inspect_local("", reply)
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            reply_shells=(
                SemanticReplyShell(
                    operator="decorated_identity",
                    decoration_span="in a velvet sleeve",
                    evidence_spans=(reply,),
                ),
            ),
        ),
        coverage_resolutions=_coverage_resolutions(
            local.deterministic,
            side="reply",
            category="possible_reduction",
            disposition="typed_relation",
            evidence_spans=(reply,),
        ),
    )

    result = await SemanticExtractionCoordinator(FakeSemanticProvider(proposal, [])).inspect(
        "", reply
    )

    assert any(
        "category-compatible proposed relation" in error
        for error in result.provider_validation_errors
    )
    assert result.canonical == result.deterministic


@pytest.mark.asyncio
async def test_human_harm_resolution_requires_explicit_outcome() -> None:
    source = "The blast left two residents medically unstable."
    local = SemanticExtractionCoordinator().inspect_local(source)
    residents = SemanticEntity(
        text_span="residents",
        entity_type="group_of_people",
        confidence="high",
    )
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(
            source_entities=(residents,),
            source_events=(
                SemanticEvent(
                    event_id="P1",
                    predicate="unstable",
                    affected_entity=residents,
                    event_type="critical_condition",
                    polarity="negative",
                    explicit_outcome=False,
                    evidence_spans=("unstable",),
                ),
            ),
        ),
        coverage_resolutions=_coverage_resolutions(
            local.deterministic,
            side="source",
            category="possible_harm",
            disposition="typed_relation",
            evidence_spans=("unstable",),
        ),
    )

    result = await SemanticExtractionCoordinator(FakeSemanticProvider(proposal, [])).inspect(source)

    assert any(
        "category-compatible proposed relation" in error
        for error in result.provider_validation_errors
    )
    assert result.canonical == result.deterministic


@pytest.mark.asyncio
async def test_provider_unresolved_material_must_be_an_exact_supplied_span() -> None:
    source = "The routing daemon died overnight."
    proposal = SemanticProviderProposal(
        inspection=SemanticInspection(unresolved_material=("an altered unresolved phrase",)),
    )

    result = await SemanticExtractionCoordinator(FakeSemanticProvider(proposal, [])).inspect(source)

    assert "provider unresolved material span is not exact: 'an altered unresolved phrase'" in (
        result.provider_validation_errors
    )
    assert result.canonical == result.deterministic
