from __future__ import annotations

import pytest

from app.domain.enums import GenerationMode
from app.schemas.content import AngleCandidate, EvidenceItem, EvidencePacket, GenerationVariant
from app.services.claim_structure import extract_claims
from app.services.factual_inspection import inspect_factual_integrity
from app.services.semantic_classifier import (
    DeterministicSemanticInspector,
    derive_source_semantic_map,
)


def packet(*texts: str) -> EvidencePacket:
    return EvidencePacket(
        mode=GenerationMode.SOURCE_POST,
        topic="scope hardening",
        language="en",
        items=tuple(
            EvidenceItem(evidence_id=f"e{i}", source_type="x_post", text=text)
            for i, text in enumerate(texts, start=1)
        ),
    )


def variant(text: str, evidence_ids: tuple[str, ...] = ("e1",)) -> GenerationVariant:
    return GenerationVariant(
        label="v1",
        text=text,
        similarity_risk=0.0,
        angle_id="a1",
        angle_type="plain_update",
        evidence_ids=evidence_ids,
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
    ("draft", "subtype"),
    [
        ("Alice bought Acme from Bob.", "seller_or_counterparty"),
        ("Alice bought Acme in cash.", "instrument"),
        ("Alice secretly bought Acme.", "secrecy"),
        ("Alice bought Acme to avoid bankruptcy.", "purpose"),
        ("Alice bought Acme with government approval.", "approval"),
    ],
)
def test_unsupported_acquisition_detail_blocks(draft: str, subtype: str) -> None:
    report = inspect("Alice bought Acme.", draft)
    assert report.status == "block"
    assert report.claims[0].support in {"partially_supported", "unsupported"}
    assert "UNSUPPORTED_EVENT_DETAIL" in report.claims[0].issue_codes
    assert f"UNSUPPORTED_EVENT_DETAIL:{subtype}" in report.claims[0].issue_codes


@pytest.mark.parametrize(
    ("source", "draft", "subtype"),
    [
        (
            "Hungary's parliament removed President Sulyok.",
            "Hungary's parliament illegally removed President Sulyok.",
            "legal_status",
        ),
        ("Alice signed the bill.", "Alice reluctantly signed the bill.", "voluntariness"),
        ("Alice signed the bill.", "Alice signed the bill under pressure.", "pressure"),
        ("Alice signed the bill.", "Alice narrowly signed the bill.", "margin"),
        ("Alice signed the bill.", "Alice voluntarily signed the bill.", "voluntariness"),
    ],
)
def test_unsupported_manner_and_status_detail_blocks(source: str, draft: str, subtype: str) -> None:
    report = inspect(source, draft)
    assert report.status == "block"
    assert "UNSUPPORTED_EVENT_DETAIL" in report.claims[0].issue_codes
    assert f"UNSUPPORTED_EVENT_DETAIL:{subtype}" in report.claims[0].issue_codes


@pytest.mark.parametrize(
    "source",
    [
        "Alice is scheduled to sign the bill.",
        "Alice agreed to sign the bill.",
        "Alice promised to sign the bill.",
        "Alice hoped to sign the bill.",
        "Alice wanted to sign the bill.",
        "Alice sought to sign the bill.",
        "Alice was authorized to sign the bill.",
        "Alice was prevented from signing the bill.",
    ],
)
def test_matrix_event_does_not_establish_embedded_completion(source: str) -> None:
    report = inspect(source, "Alice signed the bill.")
    assert report.status == "block"
    assert "EMBEDDED_EVENT_NOT_ASSERTED" in report.claims[0].issue_codes


@pytest.mark.parametrize(
    ("source", "expected_issue"),
    [
        ("Whether Alice signed the bill remains unclear.", "UNCERTAINTY_AS_ASSERTION"),
        ("It is unclear whether Alice signed the bill.", "UNCERTAINTY_AS_ASSERTION"),
        ("There is no evidence Alice signed the bill.", "EVIDENTIAL_SCOPE_LOST"),
        ("The claim that Alice signed the bill is false.", "DENIED_PROPOSITION_AS_FACT"),
        ("Reports dispute whether Alice signed the bill.", "UNCERTAINTY_AS_ASSERTION"),
        ("It has not been established that Alice signed the bill.", "EVIDENTIAL_SCOPE_LOST"),
    ],
)
def test_uncertainty_and_denial_scope_does_not_establish_fact(
    source: str, expected_issue: str
) -> None:
    report = inspect(source, "Alice signed the bill.")
    assert report.status == "block"
    assert expected_issue in report.claims[0].issue_codes


@pytest.mark.parametrize(
    "source",
    [
        "Alice signed the bill only if Bob approved it.",
        "Alice signs the bill when Bob approves it.",
        "Alice signed the bill provided that Bob approved it.",
        "Alice can sign the bill once Bob approves it.",
        "Alice may sign the bill subject to committee approval.",
    ],
)
def test_condition_bearing_clause_does_not_establish_unconditional_fact(source: str) -> None:
    report = inspect(source, "Alice signed the bill.")
    assert report.status == "block"
    assert "EMBEDDED_EVENT_NOT_ASSERTED" in report.claims[0].issue_codes


@pytest.mark.parametrize(
    ("source", "draft"),
    [
        (
            "Alice signed the tax bill. Alice did not sign the spending bill.",
            "Alice signed the tax bill.",
        ),
        (
            "Alice signed the merger agreement. Alice did not sign the employment agreement.",
            "Alice signed the merger agreement.",
        ),
        (
            "Alice approved the US policy. Alice rejected the EU policy.",
            "Alice approved the US policy.",
        ),
        (
            "Alice approved the first proposal. Alice rejected the second proposal.",
            "Alice approved the first proposal.",
        ),
        (
            "Alice opened the iOS app. Alice closed the Android app.",
            "Alice opened the iOS app.",
        ),
    ],
)
def test_distinct_objects_do_not_create_false_conflict(source: str, draft: str) -> None:
    report = inspect(source, draft)
    assert report.status == "pass"
    assert report.claims[0].support == "supported"


@pytest.mark.parametrize(
    ("text", "claim_count"),
    [
        ("U.S. President Alice signed the bill.", 1),
        ("U.K. regulators approved the deal.", 1),
        ("J.D. Vance signed the bill.", 1),
        ("A.I. systems changed.", 1),
        ("Alice signed the bill. See https://example.com/report.", 1),
        ("Email alice@example.com for the source.", 0),
        ("The fee increased to 0.02.", 1),
        ("The fee increased to 2.5%.", 1),
        ("The fee is $1.25.", 1),
        ("Version 2.1 shipped.", 1),
    ],
)
def test_segmentation_preserves_atomic_tokens(text: str, claim_count: int) -> None:
    claims = extract_claims(text)
    assert len(claims) == claim_count
    assert all(claim.exact_text not in {"U.", "S.", "J.", "D."} for claim in claims)


@pytest.mark.asyncio
async def test_all_planned_contribution_propositions_must_survive_revision() -> None:
    planned = "Alice cut the fee from 2% to 1%. Bob approved the change."
    evidence = packet(planned)
    inspector = DeterministicSemanticInspector()
    contribution = AngleCandidate(
        angle_id="a1",
        angle_type="plain_update",
        thesis=planned,
        evidence_ids=("e1",),
        why_interesting="direct",
        confidence="high",
        contribution_type="direct_update",
    )
    dropped = await inspector.inspect_draft(
        evidence,
        contribution,
        "Alice cut the fee from 2% to 1%.",
        derive_source_semantic_map(evidence),
        commentary_mode=False,
    )
    assert dropped.contribution_preserved is False
    assert "CONTRIBUTION_PROPOSITION_DROPPED" in dropped.issue_codes

    changed = await inspector.inspect_draft(
        evidence,
        contribution,
        "Alice cut the fee from 2% to 1%. Carol approved the change.",
        derive_source_semantic_map(evidence),
        commentary_mode=False,
    )
    assert changed.contribution_preserved is False
    assert "CONTRIBUTION_PROPOSITION_CHANGED" in changed.issue_codes


@pytest.mark.asyncio
async def test_material_qualifier_must_survive_revision() -> None:
    planned = "According to Bob, Alice bought Acme from Carol."
    evidence = packet(planned)
    inspector = DeterministicSemanticInspector()
    contribution = AngleCandidate(
        angle_id="a1",
        angle_type="plain_update",
        thesis=planned,
        evidence_ids=("e1",),
        why_interesting="direct",
        confidence="high",
        contribution_type="direct_update",
    )
    result = await inspector.inspect_draft(
        evidence,
        contribution,
        "According to Bob, Alice bought Acme.",
        derive_source_semantic_map(evidence),
        commentary_mode=False,
    )
    assert result.contribution_preserved is False
    assert "CONTRIBUTION_QUALIFIER_CHANGED" in result.issue_codes


HUNGARY = (
    "BREAKING: Hungary's parliament votes to remove President Tamás Sulyok, in the latest "
    "move to dismantle Viktor Orbán's remaining influence."
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("contribution_type", "thesis"),
    [
        ("joke", "Hungary removed Sulyok, now with anti-Orbán branding."),
        ("dry_reframe", "The campaign against Orbán now includes the presidency."),
    ],
)
async def test_reaction_label_cannot_ground_plain_paraphrase(
    contribution_type: str, thesis: str
) -> None:
    evidence = packet(HUNGARY)
    inspector = DeterministicSemanticInspector()
    result = await inspector.inspect_contributions(
        evidence,
        (
            AngleCandidate(
                angle_id="a1",
                angle_type="concise_joke",
                thesis=thesis,
                evidence_ids=("e1",),
                why_interesting="reaction",
                confidence="medium",
                contribution_type=contribution_type,  # type: ignore[arg-type]
            ),
        ),
        derive_source_semantic_map(evidence),
    )
    assert result[0].grounded is False
    assert result[0].plain_paraphrase_only is True


@pytest.mark.asyncio
async def test_supported_qualifier_and_real_source_specific_joke_can_pass() -> None:
    report = inspect("Alice bought Acme from Bob.", "Alice bought Acme from Bob.")
    assert report.status == "pass"

    evidence = packet("The app renamed its error page to the success screen.")
    inspector = DeterministicSemanticInspector()
    result = await inspector.inspect_contributions(
        evidence,
        (
            AngleCandidate(
                angle_id="a1",
                angle_type="concise_joke",
                thesis="nothing says success like renaming the error page",
                evidence_ids=("e1",),
                why_interesting="literal callback",
                confidence="medium",
                contribution_type="joke",
            ),
        ),
        derive_source_semantic_map(evidence),
    )
    assert result[0].grounded is True
    assert result[0].reaction_function_present is True


def test_supported_alias_legality_and_promise_controls() -> None:
    alias = inspect("Alice bought Acme.", "Alice purchased Acme.")
    assert alias.status == "pass"

    legality = inspect(
        "Hungary's parliament legally removed President Sulyok.",
        "Hungary's parliament legally removed President Sulyok.",
    )
    assert legality.status == "pass"

    promise = inspect("Alice promised to sign the bill.", "Alice promised to sign the bill.")
    assert promise.status == "pass"


@pytest.mark.asyncio
async def test_two_planned_propositions_survive_equivalent_cleanup() -> None:
    planned = "Alice cut the fee from 2% to 1%. Bob approved the change."
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
        "Alice reduced the fee from 2% to 1%. Bob approved the change.",
        derive_source_semantic_map(evidence),
        commentary_mode=False,
    )
    assert result.contribution_preserved is True


@pytest.mark.asyncio
async def test_grounded_dry_reframe_control() -> None:
    evidence = packet("The app renamed its error page to the success screen.")
    inspector = DeterministicSemanticInspector()
    result = await inspector.inspect_contributions(
        evidence,
        (
            AngleCandidate(
                angle_id="a1",
                angle_type="dry_observation",
                thesis="plot twist: the error page is now the success screen",
                evidence_ids=("e1",),
                why_interesting="source-specific inversion",
                confidence="medium",
                contribution_type="dry_reframe",
            ),
        ),
        derive_source_semantic_map(evidence),
    )
    assert result[0].grounded is True
    assert result[0].plain_paraphrase_only is False
