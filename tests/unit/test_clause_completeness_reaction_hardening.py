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
        topic="clause completeness",
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
    "draft",
    [
        "Alice bought Acme, which Bob founded.",
        "Alice bought Acme, which was bankrupt.",
        "Alice bought Acme, while Bob sold Beta.",
        "Alice bought Acme, but Bob sold Beta.",
        "Alice bought Acme, then Bob sold Beta.",
        "Alice bought Acme, as Bob approved the deal.",
        "Alice bought Acme, in a deal approved by regulators.",
    ],
)
def test_unsupported_secondary_proposition_blocks(draft: str) -> None:
    report = inspect("Alice bought Acme.", draft)
    assert report.status == "block"
    assert report.approval_eligible is False
    assert len(report.claims) >= 2 or any(
        "UNINSPECTED_SECONDARY_PROPOSITION" in claim.issue_codes for claim in report.claims
    )


def test_changed_relative_clause_actor_blocks() -> None:
    report = inspect(
        "Alice bought Acme, which Bob founded.",
        "Alice bought Acme, which Carol founded.",
    )
    assert report.status == "block"
    assert any(claim.support in {"unsupported", "source_conflict"} for claim in report.claims)


@pytest.mark.parametrize(
    ("source", "draft", "expected"),
    [
        (
            "Revenue increased from 5 million to 10 million.",
            "Revenue increased to 10 million.",
            "pass",
        ),
        (
            "The fee decreased from 2% to 1%.",
            "The fee decreased to 1%.",
            "pass",
        ),
        (
            "Revenue increased by 10% to 5 million.",
            "Revenue increased to 5 million.",
            "pass",
        ),
        (
            "Revenue increased to 10 million.",
            "Revenue increased from 5 million to 10 million.",
            "block",
        ),
        (
            "The fee decreased from 2% to 1%.",
            "The fee decreased from 3% to 1%.",
            "block",
        ),
    ],
)
def test_quantity_role_entailment(source: str, draft: str, expected: str) -> None:
    report = inspect(source, draft)
    assert report.status == expected


@pytest.mark.parametrize(
    "text",
    [
        "Gov. Alice signed the bill.",
        "Sen. Alice signed the bill.",
        "Rep. Alice signed the bill.",
        "Gen. Alice approved the plan.",
        "Lt. Alice signed the order.",
        "No. 5 was approved.",
        "Jan. 5 was the filing date.",
        "Sept. 12 was the hearing date.",
    ],
)
def test_common_news_abbreviations_remain_atomic(text: str) -> None:
    claims = extract_claims(text)
    assert len(claims) == 1
    assert claims[0].exact_text == text


@pytest.mark.asyncio
async def test_relative_and_linked_contribution_clauses_must_survive() -> None:
    inspector = DeterministicSemanticInspector()
    cases = [
        (
            "Alice bought Acme, which Bob founded.",
            "Alice bought Acme.",
            "CONTRIBUTION_PROPOSITION_DROPPED",
        ),
        (
            "Alice bought Acme, which Bob founded.",
            "Alice bought Acme, which Carol founded.",
            "CONTRIBUTION_PROPOSITION_CHANGED",
        ),
        (
            "Alice bought Acme, while Bob sold Beta.",
            "Alice bought Acme.",
            "CONTRIBUTION_PROPOSITION_DROPPED",
        ),
    ]
    for planned, revised, expected in cases:
        evidence = packet(planned)
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
            revised,
            derive_source_semantic_map(evidence),
            commentary_mode=False,
        )
        assert result.contribution_preserved is False
        assert expected in result.issue_codes


HUNGARY = (
    "BREAKING: Hungary's parliament votes to remove President Tamás Sulyok, in the latest move "
    "to dismantle Viktor Orbán's remaining influence."
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "thesis",
    [
        "nothing says dismantling influence like removing the president",
        "finally, a presidential anti-Orbán update",
        "plot twist: the anti-Orbán campaign reached the presidency",
        "THE ANTI-ORBÁN CAMPAIGN REACHED THE PRESIDENCY — PLOT TWIST",
        '"finally": the anti-Orbán campaign reached the presidency',
    ],
)
async def test_template_only_reaction_abstains(thesis: str) -> None:
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
                contribution_type="joke",
            ),
        ),
        derive_source_semantic_map(evidence),
    )
    assert result[0].grounded is False
    assert result[0].plain_paraphrase_only or (
        result[0].reaction_inspection is not None
        and result[0].reaction_inspection.generic_template_only
    )


@pytest.mark.parametrize(
    "text",
    [
        "Alice appointed Bob, who founded Acme.",
        "Alice bought Acme, whose board Bob chaired.",
        "Alice bought Acme, where Bob worked.",
        "Alice bought Acme, whereas Bob sold Beta.",
        "Alice bought Acme, although Bob objected.",
        "Alice bought Acme, though Bob objected.",
        "Alice bought Acme, because Bob approved it.",
        "Alice bought Acme, after Bob sold Beta.",
        "Alice bought Acme, before Bob sold Beta.",
    ],
)
def test_material_relative_and_linked_tails_are_never_silently_discarded(text: str) -> None:
    claims = extract_claims(text)
    assert len(claims) >= 2
    assert claims[1].clause_relation != "main"
    assert claims[1].predicate is not None or claims[1].unconsumed_material_text


@pytest.mark.parametrize(
    "text",
    [
        "Alice bought Acme, which Bob founded, while Carol sold Beta.",
        "Alice bought Acme, which Bob founded and Carol sold Beta.",
        "Alice bought Acme, while Bob sold Beta, and Carol approved the deal.",
    ],
)
def test_nested_material_tail_fails_closed_in_bounded_fallback(text: str) -> None:
    claims = extract_claims(text)
    assert len(claims) == 2
    assert claims[1].kind == "material_claim_unknown"
    assert claims[1].unconsumed_material_text


def test_supported_relative_and_coordinated_propositions_pass() -> None:
    relative = inspect(
        "Alice bought Acme, which Bob founded.",
        "Alice bought Acme, which Bob founded.",
    )
    coordinated = inspect(
        "Alice bought Acme, while Bob sold Beta.",
        "Alice bought Acme, while Bob sold Beta.",
    )
    assert relative.status == "pass"
    assert coordinated.status == "pass"
    assert len(relative.claims) == 2
    assert len(coordinated.claims) == 2


def test_ambiguous_relative_antecedent_fails_closed() -> None:
    report = inspect(
        "Alice bought Acme.",
        "Alice bought Acme, whose board Bob chaired.",
    )
    assert report.status == "block"
    assert any(
        "UNINSPECTED_SECONDARY_PROPOSITION" in claim.issue_codes
        or "CLAIM_EXTRACTION_UNCERTAIN" in claim.issue_codes
        for claim in report.claims
    )


def test_quantity_mentions_preserve_semantic_roles() -> None:
    claims = extract_claims("Revenue increased by 10% from 5 million to 10 million.")
    assert len(claims) == 1
    roles = {item.role: item.normalized_value for item in claims[0].quantity_mentions}
    assert roles["delta"] == "10:%"
    assert roles["start_value"] == "5:m"
    assert roles["end_value"] == "10:m"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "thesis",
    [
        "nothing says dismantling influence like removing the president?",
        "“nothing says dismantling influence like removing the president”",
        "the anti-Orbán campaign reached the presidency — plot twist",
        "PLOT TWIST? THE ANTI-ORBÁN CAMPAIGN REACHED THE PRESIDENCY.",
        "at last: the anti-Orbán campaign reached the presidency",
        "finally — the presidency joins the anti-Orbán campaign",
    ],
)
async def test_template_reaction_transformations_still_abstain(thesis: str) -> None:
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
                contribution_type="joke",
            ),
        ),
        derive_source_semantic_map(evidence),
    )
    assert result[0].grounded is False
    assert result[0].reaction_inspection is not None
    assert result[0].reaction_inspection.source_echo is True


@pytest.mark.parametrize(
    ("source", "reaction", "expected_type"),
    [
        (
            "The team called the outage a small configuration issue.",
            '"configuration issue" is doing heroic amounts of work here.',
            "callback",
        ),
        (
            "The app now requires a second confirmation before deleting files.",
            "Delete now comes with a second opinion.",
            "dry_reframe",
        ),
    ],
)
def test_genuine_source_specific_reactions_remain_grounded(
    source: str,
    reaction: str,
    expected_type: str,
) -> None:
    from app.services.semantic_classifier import inspect_reaction_function

    result = inspect_reaction_function(packet(source), reaction)
    assert result.grounded is True
    assert result.source_specific is True
    assert result.source_echo is False
    assert result.reaction_type == expected_type
