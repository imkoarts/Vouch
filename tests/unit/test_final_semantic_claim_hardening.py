from __future__ import annotations

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
from app.services.claim_extraction import merge_extractor_results
from app.services.claim_structure import extract_claims
from app.services.factual_inspection import inspect_factual_integrity
from app.services.semantic_classifier import (
    DeterministicSemanticInspector,
    derive_source_semantic_map,
    merge_source_semantic_maps,
)


def packet(*texts: str) -> EvidencePacket:
    return EvidencePacket(
        mode=GenerationMode.SOURCE_POST,
        topic="hardening",
        language="en",
        items=tuple(
            EvidenceItem(evidence_id=f"e{index}", source_type="x_post", text=text)
            for index, text in enumerate(texts, start=1)
        ),
    )


def variant(text: str, evidence_ids: tuple[str, ...] | None = None) -> GenerationVariant:
    return GenerationVariant(
        label="v1",
        text=text,
        similarity_risk=0.0,
        angle_id="a1",
        angle_type="plain_update",
        evidence_ids=evidence_ids or ("e1",),
        confidence="high",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
    )


def inspect(source: str, draft: str):
    return inspect_factual_integrity(
        variant(draft), packet=packet(source), factual_inspection_required=True
    )


@pytest.mark.parametrize(
    ("evidence", "draft"),
    [
        (("Alice signed the bill.", "Alice did not sign the bill."), "Alice signed the bill."),
        (("The fee increased to 2%.", "The fee decreased to 1%."), "The fee increased to 2%."),
        (
            (
                "According to Bob, Alice signed the bill.",
                "According to Carol, Alice did not sign the bill.",
            ),
            "According to Bob, Alice signed the bill.",
        ),
    ],
)
def test_support_cannot_hide_direct_conflict(evidence: tuple[str, ...], draft: str) -> None:
    report = inspect_factual_integrity(
        variant(draft, tuple(f"e{i}" for i in range(1, len(evidence) + 1))),
        packet=packet(*evidence),
        factual_inspection_required=True,
    )
    assert report.status == "block"
    assert report.claims[0].support == "source_conflict"
    assert report.claims[0].conflicting_evidence_ids


def test_direct_conflict_inside_one_evidence_item_blocks() -> None:
    report = inspect(
        "Alice signed the bill. Alice did not sign the bill.",
        "Alice signed the bill.",
    )
    assert report.status == "block"
    assert report.claims[0].support == "source_conflict"
    assert report.claims[0].supporting_evidence_ids == ("e1",)
    assert report.claims[0].conflicting_evidence_ids == ("e1",)


def test_evidence_order_does_not_change_conflict_result() -> None:
    first = inspect_factual_integrity(
        variant("Alice signed the bill.", ("e1", "e2")),
        packet=packet("Alice signed the bill.", "Alice did not sign the bill."),
        factual_inspection_required=True,
    )
    second = inspect_factual_integrity(
        variant("Alice signed the bill.", ("e1", "e2")),
        packet=packet("Alice did not sign the bill.", "Alice signed the bill."),
        factual_inspection_required=True,
    )
    assert first.claims[0].support == second.claims[0].support == "source_conflict"


@pytest.mark.parametrize(
    "source",
    [
        "Did Alice sign the bill?",
        "If Alice signed the bill, the policy would change.",
        "Alice would sign the bill if the amendment passed.",
        "Unless Alice signs the bill, the policy will not change.",
        "Had Alice signed the bill, the policy would have changed.",
    ],
)
def test_question_condition_and_counterfactual_do_not_establish_event(source: str) -> None:
    report = inspect(source, "Alice signed the bill.")
    assert report.status == "block"
    assert report.claims[0].support in {"unsupported", "source_conflict"}


@pytest.mark.parametrize(
    ("source", "expected_status"),
    [
        ("Alice almost signed the bill.", "near_event"),
        ("Alice attempted to sign the bill.", "attempted"),
        ("Alice was about to sign the bill.", "near_event"),
        ("Alice plans to sign the bill.", "planned"),
        ("Alice is expected to sign the bill.", "future"),
        ("Alice will sign the bill.", "future"),
    ],
)
def test_non_completed_event_cannot_support_completion(source: str, expected_status: str) -> None:
    assert extract_claims(source)[0].event_status == expected_status
    report = inspect(source, "Alice signed the bill.")
    assert report.status == "block"
    assert "EVENT_STATUS_ESCALATION" in report.claims[0].issue_codes


@pytest.mark.parametrize(
    "source", ["Alice reportedly signed the bill.", "Alice allegedly signed the bill."]
)
def test_hearsay_cannot_become_objective_assertion(source: str) -> None:
    report = inspect(source, "Alice signed the bill.")
    assert report.status == "block"
    assert report.human_verification_required is True


@pytest.mark.parametrize(
    ("source", "draft"),
    [
        (
            "According to Bob, Alice may sign the bill.",
            "According to Bob, Alice signed the bill.",
        ),
        ("Bob said Alice may sign the bill.", "Bob said Alice signed the bill."),
    ],
)
def test_nested_attributed_modality_is_preserved(source: str, draft: str) -> None:
    report = inspect(source, draft)
    assert report.status == "block"
    assert "MODALITY_ESCALATION" in report.claims[0].issue_codes


def test_provider_structure_cannot_replace_final_text_meaning() -> None:
    text = "Alice brokered Acme."
    local = extract_claims(text)
    provider = (
        CanonicalClaim(
            claim_id="p1",
            exact_text=text,
            kind="factual",
            subject="Alice",
            predicate="buy",
            object="Acme",
            polarity="positive",
            modality="asserted",
            start=0,
            end=len(text),
        ),
    )
    merged = merge_extractor_results(text, local, provider)
    assert merged[0].extraction_conflict is True
    report = inspect_factual_integrity(
        variant(text),
        packet=packet("Alice bought Acme."),
        factual_inspection_required=True,
        extracted_claims=merged,
    )
    assert report.claims[0].support == "extraction_uncertain"
    assert "CLAIM_EXTRACTION_CONFLICT" in report.claims[0].issue_codes


def test_bad_provider_structure_does_not_become_a_second_draft_claim() -> None:
    text = "Alice bought Acme."
    local = extract_claims(text)
    provider = (
        local[0].model_copy(
            update={
                "claim_id": "p1",
                "subject": "Bob",
                "extraction_provenance": "provider",
            }
        ),
    )
    merged = merge_extractor_results(text, local, provider)
    assert len(merged) == 1
    assert merged[0].extraction_conflict is True


@pytest.mark.parametrize(
    "text",
    [
        "The fee increased to 0.02.",
        "The fee increased to 2.5%.",
        "The fee is $1.25.",
        "Version 2.1 shipped.",
    ],
)
def test_decimal_and_version_tokens_remain_one_claim(text: str) -> None:
    claims = extract_claims(text)
    assert len(claims) == 1
    assert claims[0].exact_text == text


@pytest.mark.parametrize(
    ("source", "draft"),
    [
        ("Revenue increased to 10 million.", "Revenue increased to 10m."),
        ("The fee increased to 2 percent.", "The fee increased to 2%."),
        ("Revenue increased to 1 billion.", "Revenue increased to 1bn."),
    ],
)
def test_conservative_equivalent_numeric_forms_match(source: str, draft: str) -> None:
    report = inspect(source, draft)
    assert report.status == "pass"
    assert report.claims[0].support == "supported"


@pytest.mark.parametrize(
    ("source", "draft"),
    [
        ("Alice signed the bill.", "Alice may have signed the bill."),
        ("Alice will sign the bill.", "Alice will sign the bill."),
        ("Alice plans to sign the bill.", "Alice plans to sign the bill."),
        ("According to Bob, Alice signed the bill.", "According to Bob, Alice signed the bill."),
    ],
)
def test_semantic_dimension_positive_controls(source: str, draft: str) -> None:
    report = inspect(source, draft)
    assert report.status == "pass"
    assert report.claims[0].support == "supported"


def commentary_angle(text: str) -> AngleCandidate:
    return AngleCandidate(
        angle_id="a1",
        angle_type="consequence",
        thesis=text,
        evidence_ids=("e1",),
        why_interesting="test",
        confidence="medium",
        contribution_type="bounded_consequence",
    )


HUNGARY = (
    "BREAKING: Hungary's parliament votes to remove President Tamás Sulyok, in the latest "
    "move to dismantle Viktor Orbán's remaining influence."
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "thesis",
    [
        "A vote on Sulyok turns a broad anti-Orbán agenda into an institutional action.",
        "Sulyok is the latest officeholder caught in the effort to reduce Orbán's influence.",
        "The vote gives the anti-Orbán project control over another institution.",
        "Sulyok's ouster broadens the campaign against Orbán to the country's highest office.",
        "Parliament is operationalizing the rollback of Orbán's influence through the presidency.",
        "I see Sulyok's removal as the anti-Orbán campaign reaching the presidency.",
        "Has the campaign against Orbán reached the presidency? Sulyok's removal says yes.",
        "The presidency was brought into the campaign against Orbán by Sulyok's removal.",
        "Sulyok's removal is a presidential extension of the campaign against Orbán.",
        "The anti-Orbán effort has widened; it now includes the presidency.",
        "By removing Sulyok, parliament broadened the rollback of Orbán's influence.",
    ],
)
async def test_planner_rejects_hungary_repackaging_before_drafting(thesis: str) -> None:
    evidence = packet(HUNGARY)
    inspector = DeterministicSemanticInspector()
    result = await inspector.inspect_contributions(
        evidence, (commentary_angle(thesis),), derive_source_semantic_map(evidence)
    )
    assert result[0].grounded is False
    assert result[0].already_covered or result[0].requires_new_assumption


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("planned", "revision", "issue"),
    [
        ("Alice cut the fee from 2% to 1%.", "Alice cut the fee.", "CONTRIBUTION_DETAIL_DROPPED"),
        ("Alice may sign the bill.", "Alice signed the bill.", "CERTAINTY_CHANGED_DURING_REVISION"),
        (
            "According to Bob, Alice signed the bill.",
            "Alice signed the bill.",
            "ATTRIBUTION_CHANGED_DURING_REVISION",
        ),
        ("Alice signed the bill Tuesday.", "Alice signed the bill.", "CONTRIBUTION_DETAIL_DROPPED"),
    ],
)
async def test_humanizer_revision_cannot_change_contribution(
    planned: str, revision: str, issue: str
) -> None:
    evidence = packet(planned)
    inspector = DeterministicSemanticInspector()
    result = await inspector.inspect_draft(
        evidence,
        AngleCandidate(
            angle_id="a1",
            angle_type="plain_update",
            thesis=planned,
            evidence_ids=("e1",),
            why_interesting="direct",
            confidence="high",
            contribution_type="direct_update",
        ),
        revision,
        derive_source_semantic_map(evidence),
        commentary_mode=False,
    )
    assert result.contribution_preserved is False
    assert issue in result.issue_codes


def test_provider_semantic_text_relation_mismatch_is_rejected() -> None:
    evidence = packet("Alice bought Acme.")
    deterministic = derive_source_semantic_map(evidence)
    fake_relation = extract_claims("Alice bought Acme.", prefix="P")[0].model_copy(
        update={"exact_text": "Alice controls Acme."}
    )
    provider = SourceSemanticMap(
        propositions=(
            SourceSemanticProposition(
                proposition_id="p1",
                text="Alice controls Acme.",
                status="strongly_implied",
                relation=fake_relation,
                evidence_spans=(EvidenceSpan(evidence_id="e1", quote="Alice bought Acme."),),
            ),
        )
    )
    merged = merge_source_semantic_maps(evidence, deterministic, provider)
    assert all(item.text != "Alice controls Acme." for item in merged.propositions)
