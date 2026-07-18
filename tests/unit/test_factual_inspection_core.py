from __future__ import annotations

from app.domain.enums import GenerationMode
from app.schemas.content import EvidenceItem, EvidencePacket, EvidenceSpan, GenerationVariant
from app.services.factual_inspection import inspect_factual_integrity, validate_evidence_spans


def _packet() -> EvidencePacket:
    return EvidencePacket(
        mode=GenerationMode.SOURCE_POST,
        topic="fact",
        language="en",
        items=(
            EvidenceItem(
                evidence_id="canonical",
                source_type="official_statement",
                text="Alice signed the bill on Tuesday.",
            ),
        ),
    )


def _variant(**updates: object) -> GenerationVariant:
    base = GenerationVariant(
        label="v",
        text="Alice signed the bill on Tuesday.",
        similarity_risk=0.0,
        angle_id="a",
        angle_type="plain_update",
        evidence_ids=("canonical",),
        confidence="high",
        factual_claims=("Alice signed the bill on Tuesday.",),
        attributed_claims=(),
        uncertainty_markers=(),
    )
    return base.model_copy(update=updates)


def test_invalid_evidence_id_is_rejected() -> None:
    report = inspect_factual_integrity(
        _variant(evidence_ids=("invented",)),
        packet=_packet(),
        factual_inspection_required=True,
    )

    assert report.status == "block"
    assert report.invalid_evidence_ids == ("invented",)
    assert report.approval_eligible is False


def test_invented_excerpt_is_rejected_without_replacement() -> None:
    valid, invalid_ids, invalid_excerpts = validate_evidence_spans(
        _packet(),
        (EvidenceSpan(evidence_id="canonical", quote="Alice rejected the bill."),),
    )

    assert valid == ()
    assert invalid_ids == ()
    assert invalid_excerpts == ("Alice rejected the bill.",)


def test_exact_excerpt_and_offsets_are_accepted() -> None:
    text = "Alice signed the bill on Tuesday."
    valid, invalid_ids, invalid_excerpts = validate_evidence_spans(
        _packet(),
        (EvidenceSpan(evidence_id="canonical", quote=text, start=0, end=len(text)),),
    )

    assert len(valid) == 1
    assert invalid_ids == ()
    assert invalid_excerpts == ()


def test_opinion_without_factual_claims_is_not_applicable() -> None:
    report = inspect_factual_integrity(
        _variant(
            text="I like the shorter version.",
            factual_claims=(),
            attributed_claims=(),
            evidence_ids=("canonical",),
        ),
        packet=_packet(),
        factual_inspection_required=False,
    )

    assert report.applicable is False
    assert report.status == "not_applicable"
    assert report.claims[0].kind == "opinion"
    assert report.approval_eligible is True
