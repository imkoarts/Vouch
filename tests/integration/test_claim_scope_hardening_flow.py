from __future__ import annotations

import pytest

from app.domain.enums import GenerationMode
from app.schemas.content import AngleCandidate, EvidenceItem, EvidencePacket, GenerationVariant
from app.services.factual_inspection import inspect_factual_integrity
from app.services.semantic_classifier import (
    DeterministicSemanticInspector,
    derive_source_semantic_map,
)


def packet(text: str) -> EvidencePacket:
    return EvidencePacket(
        mode=GenerationMode.SOURCE_POST,
        topic="claim scope integration",
        language="en",
        items=(EvidenceItem(evidence_id="e1", source_type="x_post", text=text),),
    )


def variant(text: str) -> GenerationVariant:
    return GenerationVariant(
        label="v1",
        text=text,
        similarity_risk=0.0,
        angle_id="a1",
        angle_type="plain_update",
        evidence_ids=("e1",),
        confidence="high",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
    )


def test_unsupported_event_detail_is_not_approval_eligible() -> None:
    report = inspect_factual_integrity(
        variant("Alice bought Acme with government approval."),
        packet=packet("Alice bought Acme."),
        factual_inspection_required=True,
    )
    assert report.status == "block"
    assert report.approval_eligible is False
    assert "UNSUPPORTED_EVENT_DETAIL" in report.claims[0].issue_codes


def test_matrix_scope_is_not_flattened_in_factual_workflow() -> None:
    report = inspect_factual_integrity(
        variant("Alice signed the bill."),
        packet=packet("Alice promised to sign the bill."),
        factual_inspection_required=True,
    )
    assert report.status == "block"
    assert "EMBEDDED_EVENT_NOT_ASSERTED" in report.claims[0].issue_codes


def test_distinct_object_conflict_does_not_poison_selected_claim() -> None:
    report = inspect_factual_integrity(
        variant("Alice signed the tax bill."),
        packet=packet("Alice signed the tax bill. Alice did not sign the spending bill."),
        factual_inspection_required=True,
    )
    assert report.status == "pass"
    assert report.approval_eligible is True


@pytest.mark.asyncio
async def test_plain_paraphrase_mislabeled_as_reaction_abstains_before_drafting() -> None:
    evidence = packet(
        "Hungary's parliament removed President Sulyok in a move against Orbán's influence."
    )
    contribution = AngleCandidate(
        angle_id="a1",
        angle_type="concise_joke",
        thesis="The campaign against Orbán now includes the presidency.",
        evidence_ids=("e1",),
        why_interesting="provider-labeled reaction",
        confidence="medium",
        contribution_type="dry_reframe",
    )
    inspector = DeterministicSemanticInspector()
    result = await inspector.inspect_contributions(
        evidence,
        (contribution,),
        derive_source_semantic_map(evidence),
    )
    assert result[0].grounded is False
    assert result[0].plain_paraphrase_only is True
