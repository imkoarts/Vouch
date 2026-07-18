from __future__ import annotations

# ruff: noqa: RUF001
import pytest

from app.domain.enums import GenerationMode
from app.schemas.content import (
    AngleCandidate,
    CanonicalClaim,
    EvidenceItem,
    EvidencePacket,
    EvidenceSpan,
    GenerationVariant,
    SourceSemanticMap,
    SourceSemanticProposition,
)
from app.services.claim_structure import (
    compare_claim_sets,
    extract_claims,
    proposition_relation,
)
from app.services.factual_inspection import inspect_factual_integrity
from app.services.semantic_classifier import (
    DeterministicSemanticInspector,
    derive_source_semantic_map,
    merge_source_semantic_maps,
)


def packet(text: str) -> EvidencePacket:
    return EvidencePacket(
        mode=GenerationMode.SOURCE_POST,
        topic="claim matching",
        language="en",
        items=(EvidenceItem(evidence_id="e1", source_type="x_post", text=text),),
    )


def variant(text: str, *, factual_claims: tuple[str, ...] = ()) -> GenerationVariant:
    return GenerationVariant(
        label="v1",
        text=text,
        similarity_risk=0.0,
        angle_id="a1",
        angle_type="plain_update",
        evidence_ids=("e1",),
        confidence="high",
        factual_claims=factual_claims,
        attributed_claims=(),
        uncertainty_markers=(),
    )


@pytest.mark.parametrize(
    ("source", "draft"),
    [
        (
            "Revenue increased to 10 million. Users increased to 20 million.",
            "Revenue increased to 10 million.",
        ),
        (
            "Alice signed the bill on Tuesday. Bob signed the treaty on Wednesday.",
            "Alice signed the bill on Tuesday.",
        ),
    ],
)
def test_unrelated_evidence_claims_do_not_poison_exact_support(source: str, draft: str) -> None:
    report = inspect_factual_integrity(
        variant(draft), packet=packet(source), factual_inspection_required=True
    )

    assert report.status == "pass"
    assert report.approval_eligible is True
    assert report.claims[0].support == "supported"


@pytest.mark.parametrize(
    "text",
    [
        "Alice bought Acme.",
        "Alice acquired Acme.",
        "Alice sold Acme.",
        "Alice resigned.",
        "Alice was appointed CEO.",
        "Alice filed the complaint.",
        "Alice was arrested.",
        "Alice died.",
        "Acme merged with Beta.",
        "The board approved the deal.",
        "The board rejected the deal.",
        "Acme closed the office.",
        "Acme opened the office.",
        "Acme delayed the launch.",
        "Acme cancelled the launch.",
    ],
)
def test_common_declarative_event_is_not_silently_opinion(text: str) -> None:
    claim = extract_claims(text)[0]
    assert claim.kind != "opinion"


@pytest.mark.parametrize(
    ("text", "forbidden"),
    [
        ("Alice was assigned the bill.", "sign"),
        ("The redesign shipped today.", "sign"),
        ("The effort to unwind Orbán’s influence has reached the head of state.", "win"),
    ],
)
def test_predicate_matching_uses_token_boundaries(text: str, forbidden: str) -> None:
    assert extract_claims(text)[0].predicate != forbidden


@pytest.mark.parametrize(
    "draft",
    [
        "Alice failed to sign the bill.",
        "Alice refused to sign the bill.",
        "Alice declined to sign the bill.",
        "Alice has yet to sign the bill.",
        "Alice signed no bill.",
    ],
)
def test_lexical_non_completion_conflicts_with_completed_source(draft: str) -> None:
    report = inspect_factual_integrity(
        variant(draft), packet=packet("Alice signed the bill."), factual_inspection_required=True
    )
    assert report.status == "block"
    assert report.claims[0].support == "source_conflict"


def test_copular_negation_is_source_conflict() -> None:
    report = inspect_factual_integrity(
        variant("Acme is not profitable."),
        packet=packet("Acme is profitable."),
        factual_inspection_required=True,
    )
    assert report.status == "block"
    assert report.claims[0].support == "source_conflict"


def test_possible_evidence_cannot_support_asserted_completion() -> None:
    report = inspect_factual_integrity(
        variant("Alice signed the bill."),
        packet=packet("Alice may sign the bill."),
        factual_inspection_required=True,
    )
    assert report.status == "block"
    assert report.claims[0].support in {"unsupported", "source_conflict"}
    assert "MODALITY_ESCALATION" in report.claims[0].issue_codes


def test_attribution_cannot_be_removed_as_objective_fact() -> None:
    report = inspect_factual_integrity(
        variant("Alice signed the bill."),
        packet=packet("According to Bob, Alice signed the bill."),
        factual_inspection_required=True,
    )
    assert report.status == "block"
    assert report.approval_eligible is False
    assert report.human_verification_required is True
    assert "ATTRIBUTION_LOST" in report.claims[0].issue_codes


def test_provider_hint_missed_locally_is_inspected_when_present_in_final_text() -> None:
    report = inspect_factual_integrity(
        variant("Alice bought Acme.", factual_claims=("Alice bought Acme.",)),
        packet=packet("Bob bought Acme."),
        factual_inspection_required=True,
    )
    assert report.status == "block"
    assert report.claims


def test_provider_hint_absent_from_final_text_is_ignored() -> None:
    report = inspect_factual_integrity(
        variant("I like this.", factual_claims=("Alice bought Acme.",)),
        packet=packet("Alice bought Acme."),
        factual_inspection_required=False,
    )
    assert all(item.claim != "Alice bought Acme." for item in report.claims)


def test_partial_support_blocks_strict_approval() -> None:
    report = inspect_factual_integrity(
        variant("Alice cut the fee from 2% to 1% on Tuesday."),
        packet=packet("Alice cut the fee."),
        factual_inspection_required=True,
    )
    assert report.status == "block"
    assert report.approval_eligible is False
    assert report.human_verification_required is True


def test_broader_auxiliary_claim_is_not_exact_confirmation() -> None:
    anchor = extract_claims("Alice signed the bill.")[0]
    auxiliary = extract_claims("Alice and Bob signed the bill and the treaty.")[0]
    valid, _ = proposition_relation(anchor, auxiliary, "confirms_anchor_claim")
    assert valid is False


def test_missing_quantity_is_not_full_confirmation() -> None:
    anchor = extract_claims("Alice cut the fee from 2% to 1%.")[0]
    auxiliary = extract_claims("Alice cut the fee.")[0]
    valid, _ = proposition_relation(anchor, auxiliary, "confirms_anchor_claim")
    assert valid is False


HUNGARY_SOURCE = (
    "BREAKING: Hungary’s parliament votes to remove President Tamás Sulyok, in the latest "
    "move to dismantle Viktor Orbán’s remaining influence."
)
HUNGARY_PARAPHRASES = (
    "Removing Sulyok shows the anti-Orbán campaign now reaches the presidency.",
    "This takes the anti-Orbán push all the way to the presidency.",
    "The presidency is no longer outside the effort to dismantle Orbán’s influence.",
    "Sulyok’s removal gives the anti-Orbán effort a presidential dimension.",
    "The effort to unwind Orbán’s influence has reached the head of state.",
    "Parliament has turned the presidency into another front in the campaign against Orbán.",
    "The campaign against Orbán now includes the presidency.",
)


def commentary_angle() -> AngleCandidate:
    return AngleCandidate(
        angle_id="a1",
        angle_type="consequence",
        thesis="The removal extends the anti-Orbán effort into the presidency.",
        evidence_ids=("e1",),
        why_interesting="Metamorphic source-echo regression.",
        confidence="medium",
        contribution_type="bounded_consequence",
        support_status="supported",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("draft", HUNGARY_PARAPHRASES)
async def test_new_hungary_metamorphic_paraphrases_abstain(draft: str) -> None:
    evidence = packet(HUNGARY_SOURCE)
    inspector = DeterministicSemanticInspector()
    result = await inspector.inspect_draft(
        evidence,
        commentary_angle(),
        draft,
        derive_source_semantic_map(evidence),
        commentary_mode=True,
    )
    assert result.no_grounded_contribution is True
    assert {"SOURCE_ECHO", "IMPLIED_POINT_RESTATEMENT", "LOW_INFORMATION_GAIN"} <= set(
        result.issue_codes
    )


def test_provider_strong_implication_requires_relation_compatibility() -> None:
    evidence = packet("Alice signed the bill.")
    deterministic = derive_source_semantic_map(evidence)
    bad_relation = extract_claims("Bob bought Acme.", prefix="P")[0]
    provider = SourceSemanticMap(
        propositions=(
            SourceSemanticProposition(
                proposition_id="provider-1",
                text="Bob bought Acme.",
                status="strongly_implied",
                relation=bad_relation,
                evidence_spans=(EvidenceSpan(evidence_id="e1", quote="Alice signed the bill."),),
            ),
        ),
        factual_premises=(),
    )
    merged = merge_source_semantic_maps(evidence, deterministic, provider)
    assert all(item.text != "Bob bought Acme." for item in merged.propositions)


@pytest.mark.parametrize(
    ("before", "after", "field"),
    [
        ("Alice bought Acme.", "Alice sold Acme.", "predicate"),
        ("Alice may sign the bill.", "Alice signed the bill.", "modality"),
        (
            "According to Bob, Alice signed the bill.",
            "Alice signed the bill.",
            "attribution_source",
        ),
        ("Acme is profitable.", "Acme is not profitable.", "polarity"),
    ],
)
def test_manual_edit_diff_detects_remaining_semantic_changes(
    before: str, after: str, field: str
) -> None:
    changes = compare_claim_sets(extract_claims(before), extract_claims(after))
    changed = next(item for item in changes if item.change_type == "changed")
    assert field in changed.fields_changed


@pytest.mark.parametrize(
    ("source", "draft", "expected"),
    [
        ("Alice signed the bill.", "Alice may sign the bill.", "supported"),
        ("Alice signed the bill.", "Alice will likely sign the bill.", "partially_supported"),
        ("Alice is likely to sign the bill.", "Alice signed the bill.", "source_conflict"),
        ("Alice may sign the bill.", "Alice will likely sign the bill.", "source_conflict"),
        (
            "According to Bob, Alice signed the bill.",
            "According to Bob, Alice signed the bill.",
            "supported",
        ),
        (
            "Bob denied that Alice signed the bill.",
            "Bob admitted Alice signed the bill.",
            "source_conflict",
        ),
    ],
)
def test_application_owned_modality_and_attribution_lattice(
    source: str, draft: str, expected: str
) -> None:
    report = inspect_factual_integrity(
        variant(draft), packet=packet(source), factual_inspection_required=True
    )
    assert report.claims[0].support == expected
    assert report.status == ("pass" if expected == "supported" else "block")


def test_provider_claim_cannot_replace_uncertain_local_structure() -> None:
    text = "Alice brokered Acme."
    provider_claim = CanonicalClaim(
        claim_id="provider-1",
        exact_text=text,
        kind="factual",
        subject="Alice",
        predicate="broker",
        object="Acme",
        polarity="positive",
        modality="asserted",
        start=0,
        end=len(text),
    )
    report = inspect_factual_integrity(
        variant(text),
        packet=packet("Bob brokered Acme."),
        factual_inspection_required=True,
        provider_claims=(provider_claim,),
    )
    assert all(item.predicate != "broker" for item in report.extracted_claims)
    assert report.claims[0].support == "extraction_uncertain"
    assert "CLAIM_EXTRACTION_CONFLICT" in report.claims[0].issue_codes
    assert report.status == "block"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "draft",
    (
        "To me, removing Sulyok shows the presidency is now part of the anti-Orbán campaign.",
        "Has the anti-Orbán campaign reached the presidency? Sulyok’s removal suggests it has.",
        "The presidency has been pulled into the anti-Orbán campaign by Sulyok’s removal.",
        "Sulyok’s removal is the presidency’s inclusion in the campaign against Orbán.",
        "The anti-Orbán effort reached a new office: the presidency.",
    ),
)
async def test_hungary_semantic_echo_survives_syntax_and_voice_changes(draft: str) -> None:
    evidence = packet(HUNGARY_SOURCE)
    inspector = DeterministicSemanticInspector()
    result = await inspector.inspect_draft(
        evidence,
        commentary_angle(),
        draft,
        derive_source_semantic_map(evidence),
        commentary_mode=True,
    )
    assert result.no_grounded_contribution is True
    assert "SOURCE_ECHO" in result.issue_codes
    assert "LOW_INFORMATION_GAIN" in result.issue_codes


def test_provider_strong_implication_with_inexact_offsets_is_rejected() -> None:
    evidence = packet("Alice signed the bill.")
    deterministic = derive_source_semantic_map(evidence)
    relation = extract_claims("The bill was signed by Alice.", prefix="P")[0]
    provider = SourceSemanticMap(
        propositions=(
            SourceSemanticProposition(
                proposition_id="provider-offset",
                text="The bill was signed by Alice.",
                status="strongly_implied",
                relation=relation,
                evidence_spans=(
                    EvidenceSpan(
                        evidence_id="e1",
                        quote="Alice signed the bill.",
                        start=1,
                        end=len("Alice signed the bill.") + 1,
                    ),
                ),
            ),
        ),
    )
    merged = merge_source_semantic_maps(evidence, deterministic, provider)
    assert all(item.proposition_id != "provider-offset" for item in merged.propositions)
