from __future__ import annotations

import pytest

from app.domain.enums import ContentType, GenerationMode
from app.providers.mock_provider import MockLLMProvider
from app.schemas.content import (
    AngleCandidate,
    DraftSemanticInspection,
    EvidenceItem,
    EvidencePacket,
    GenerationRequest,
    GenerationResult,
    GenerationVariant,
    HumanizerRevisionTarget,
    MediaPlan,
    QualityReport,
    UntrustedSourceData,
)
from app.services.claim_structure import (
    approval_invalidation_reason,
    compare_claim_sets,
    extract_claims,
)
from app.services.draft_service import DraftService
from app.services.editorial_quality import assess_result
from app.services.factual_inspection import inspect_factual_integrity
from app.services.generation_pipeline import build_generation_context
from app.services.semantic_classifier import (
    DeterministicSemanticInspector,
    derive_source_semantic_map,
)


def _packet(text: str) -> EvidencePacket:
    return EvidencePacket(
        mode=GenerationMode.SOURCE_POST,
        topic="hardening",
        language="en",
        items=(EvidenceItem(evidence_id="e1", source_type="x_post", text=text),),
    )


def _variant(text: str, *, claims: tuple[str, ...] = ()) -> GenerationVariant:
    return GenerationVariant(
        label="v1",
        text=text,
        similarity_risk=0.0,
        angle_id="a1",
        angle_type="plain_update",
        evidence_ids=("e1",),
        confidence="high",
        factual_claims=claims,
        attributed_claims=(),
        uncertainty_markers=(),
    )


@pytest.mark.parametrize(
    ("source", "draft", "field"),
    [
        (
            "Alice signed the bill on Tuesday.",
            "Alice did not sign the bill on Tuesday.",
            "polarity",
        ),
        (
            "Hungary\u2019s parliament voted to remove President Tamás Sulyok.",
            "Hungary\u2019s parliament voted to keep President Tamás Sulyok.",
            "predicate",
        ),
        (
            "The company cut the fee from 2% to 1%.",
            "The company increased the fee from 1% to 2%.",
            "quantities",
        ),
        (
            "Alice denied the fraud allegation.",
            "Alice admitted the fraud allegation.",
            "predicate",
        ),
    ],
)
def test_structural_contradictions_block_even_with_high_word_overlap(
    source: str, draft: str, field: str
) -> None:
    report = inspect_factual_integrity(
        _variant(draft),
        packet=_packet(source),
        factual_inspection_required=True,
    )

    assert report.status == "block"
    assert report.approval_eligible is False
    assert report.claims[0].support == "source_conflict"
    assert field in report.claims[0].explanation


@pytest.mark.parametrize(
    ("source", "draft", "field"),
    [
        (
            "Alice signed the bill on Tuesday.",
            "Bob signed the bill on Tuesday.",
            "actor",
        ),
        (
            "Alice defeated Bob.",
            "Bob defeated Alice.",
            "source_target_reversed",
        ),
        (
            "Alice: The launch was delayed.",
            "Bob: The launch was delayed.",
            "attribution",
        ),
    ],
)
def test_changed_actor_attribution_and_relation_direction_block(
    source: str, draft: str, field: str
) -> None:
    report = inspect_factual_integrity(
        _variant(draft),
        packet=_packet(source),
        factual_inspection_required=True,
    )

    assert report.status == "block"
    assert report.claims[0].support == "source_conflict"
    assert field in report.claims[0].explanation


def test_numeric_paraphrase_preserving_direction_and_values_passes() -> None:
    report = inspect_factual_integrity(
        _variant("The company reduced the fee from 2% to 1%."),
        packet=_packet("The company cut the fee from 2% to 1%."),
        factual_inspection_required=True,
    )

    assert report.status == "pass"
    assert report.approval_eligible is True
    assert report.claims[0].support == "supported"


def test_final_text_claim_is_extracted_when_provider_metadata_is_empty() -> None:
    report = inspect_factual_integrity(
        _variant("Bob signed the bill on Tuesday.", claims=()),
        packet=_packet("Alice signed the bill on Tuesday."),
        factual_inspection_required=True,
    )

    assert report.applicable is True
    assert report.extracted_claims
    assert report.provider_hint_claims == ()
    assert report.status == "block"
    assert report.claims[0].support in {"unsupported", "source_conflict"}


@pytest.mark.parametrize(
    ("before", "after", "expected_field", "reason_fragment"),
    [
        (
            "Alice signed the bill on Tuesday.",
            "Alice did not sign the bill on Tuesday.",
            "polarity",
            "polarity",
        ),
        (
            "The company cut the fee from 2% to 1%.",
            "The company cut the fee from 2% to 3%.",
            "quantities",
            "quantity",
        ),
        (
            "Alice signed the bill on Tuesday.",
            "Alice signed the bill on Wednesday.",
            "dates",
            "date",
        ),
        (
            "Hungary\u2019s parliament voted to remove President Tamás Sulyok.",
            "Hungary\u2019s parliament voted to keep President Tamás Sulyok.",
            "predicate",
            "action",
        ),
    ],
)
def test_manual_edit_claim_diff_identifies_material_change(
    before: str, after: str, expected_field: str, reason_fragment: str
) -> None:
    changes = compare_claim_sets(extract_claims(before), extract_claims(after))
    changed = next(item for item in changes if item.change_type == "changed")

    assert expected_field in changed.fields_changed
    assert reason_fragment in approval_invalidation_reason(changes).casefold()


HUNGARY_SOURCE = (
    "BREAKING: Hungary\u2019s parliament votes to remove President Tamás Sulyok, in the latest "
    "move to dismantle Viktor Orbán\u2019s remaining influence."
)
HUNGARY_PARAPHRASES = (
    "The vote moves the fight over Orbán\u2019s influence into the presidency. Removing Sulyok "
    "puts the office inside that political realignment.",
    "The vote carries the push against Orbán from political pressure into the presidency. "
    "Sulyok\u2019s removal puts the office directly inside that process.",
    "Replacing Sulyok gives the campaign against Orbán an institutional expression at the "
    "presidential level.",
)


def _commentary_angle() -> AngleCandidate:
    return AngleCandidate(
        angle_id="a1",
        angle_type="consequence",
        thesis="The removal puts the presidency inside the campaign against Orbán.",
        evidence_ids=("e1",),
        why_interesting="Semantic metamorphic control.",
        confidence="medium",
        contribution_type="bounded_consequence",
        support_status="supported",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("draft", HUNGARY_PARAPHRASES)
async def test_semantic_classifier_blocks_paraphrase_invariant_hungary_slop(draft: str) -> None:
    packet = _packet(HUNGARY_SOURCE)
    inspector = DeterministicSemanticInspector()
    source_map = derive_source_semantic_map(packet)

    result = await inspector.inspect_draft(
        packet,
        _commentary_angle(),
        draft,
        source_map,
        commentary_mode=True,
    )

    assert result.no_grounded_contribution is True
    assert {
        "SOURCE_ECHO",
        "IMPLIED_POINT_RESTATEMENT",
        "LOW_INFORMATION_GAIN",
        "ANALYTICAL_PACKAGING",
        "SOURCE_STRONGER_THAN_DRAFT",
    } <= set(result.issue_codes)


def test_explicit_auxiliary_role_cannot_rescue_unrelated_sources() -> None:
    context = build_generation_context(
        idea_summary="unrelated sources",
        idea_explanation={
            "editorial_intent": "report_event",
            "auxiliary_evidence_roles": {"fifa": "confirms_anchor_claim"},
        },
        language="en",
        sources=(
            UntrustedSourceData(
                source_id="messi", source_type="x_post", content="Messi scored for Inter Miami."
            ),
            UntrustedSourceData(
                source_id="fifa",
                source_type="x_post",
                content="FIFA announced a tournament format change.",
            ),
        ),
        content_type=ContentType.SHORT_POST,
    )

    assert context.status == "ok"
    assert {item.evidence_id for item in context.evidence.items} == {"messi"}
    assert context.auxiliary_evidence == ()
    assert context.auxiliary_relations == ()
    assert any(
        "AUXILIARY_RELATION_REJECTED" in reason for reason in context.rejected_candidate_reasons
    )


def test_valid_auxiliary_source_reporting_same_event_is_bound_propositionally() -> None:
    context = build_generation_context(
        idea_summary="same event",
        idea_explanation={
            "editorial_intent": "report_event",
            "auxiliary_evidence_roles": {"second": "reports_same_event"},
        },
        language="en",
        sources=(
            UntrustedSourceData(
                source_id="anchor",
                source_type="x_post",
                content="Alice signed the bill on Tuesday.",
            ),
            UntrustedSourceData(
                source_id="second",
                source_type="x_post",
                content="Alice signed the bill on Tuesday.",
            ),
        ),
        content_type=ContentType.SHORT_POST,
    )

    assert context.status == "ok"
    assert context.auxiliary_evidence[0].role == "reports_same_event"
    assert context.auxiliary_relations[0].relationship == "same actor-action-object proposition"
    assert context.auxiliary_relations[0].anchor_evidence_spans
    assert context.auxiliary_relations[0].auxiliary_evidence_spans


def _revision_request(source_text: str) -> GenerationRequest:
    packet = _packet(source_text)
    angle = AngleCandidate(
        angle_id="a1",
        angle_type="plain_update",
        thesis=source_text,
        evidence_ids=("e1",),
        why_interesting="Direct update.",
        confidence="high",
        contribution_type="direct_update",
        support_status="supported",
    )
    source_variant = _variant(source_text, claims=(source_text,))
    return GenerationRequest(
        idea_summary="revision",
        content_type=ContentType.SHORT_POST,
        language="en",
        generation_mode=GenerationMode.SOURCE_POST,
        editorial_intent="report_event",
        evidence_packet=packet,
        angle_candidates=(angle,),
        revision_targets=(
            HumanizerRevisionTarget(variant=source_variant, issue_codes=("ANALYTICAL_PACKAGING",)),
        ),
        revision_issue_codes=("ANALYTICAL_PACKAGING",),
    )


def test_humanizer_cannot_change_predicate_while_preserving_ids() -> None:
    request = _revision_request("Hungary\u2019s parliament voted to remove President Tamás Sulyok.")
    changed = _variant(
        "Hungary\u2019s parliament voted to keep President Tamás Sulyok.",
        claims=("Hungary\u2019s parliament voted to remove President Tamás Sulyok.",),
    ).model_copy(update={"revision_status": "revised"})
    result = GenerationResult(
        idea_summary="revision",
        recommended_format=ContentType.SHORT_POST,
        variants=(changed,),
        recommended_variant=0,
        media_plan=MediaPlan(),
    )

    reports = assess_result(result, request)

    assert "UNSUPPORTED_CONTRIBUTION" in {issue.code for issue in reports[0].issues}
    assert reports[0].passed is False


@pytest.mark.asyncio
async def test_mock_provider_refuses_to_pad_explicit_minimum_with_generic_filler() -> None:
    request = GenerationRequest(
        idea_summary="Alice signed the bill.",
        content_type=ContentType.SHORT_POST,
        language="en",
        minimum_characters=300,
        maximum_characters=500,
    )

    result = await MockLLMProvider().generate_structured(request)

    assert result.status == "insufficient_context"
    assert result.reason_code == "MOCK_HARD_MINIMUM_UNSUPPORTED"
    assert result.variants == ()


def test_categorical_semantic_block_overrides_perfect_numeric_report() -> None:
    report = QualityReport(
        passed=True,
        thesis_score=1.0,
        specificity_score=1.0,
        evidence_score=1.0,
        naturalness_score=1.0,
        plainness_score=1.0,
        rhetorical_density_score=0.0,
        reply_fit_score=1.0,
        distinctiveness_score=1.0,
        issues=(),
    )
    semantic = DraftSemanticInspection(
        contribution_delivered=False,
        source_echo=True,
        implied_restatement=True,
        internal_repetition=False,
        analytical_packaging=True,
        source_stronger_than_draft=True,
        no_grounded_contribution=True,
        issue_codes=("SOURCE_ECHO", "LOW_INFORMATION_GAIN"),
        evidence_spans=("source paraphrase",),
    )

    updated = DraftService._quality_reports_with_structured_semantics((report,), (semantic,))

    assert updated[0].passed is False
    assert updated[0].semantic_decision == "no_post_needed"
    assert {"SOURCE_ECHO", "LOW_INFORMATION_GAIN"} <= {issue.code for issue in updated[0].issues}
