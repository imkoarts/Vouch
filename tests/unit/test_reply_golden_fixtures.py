from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.enums import GenerationMode
from app.schemas.content import EvidenceItem, EvidencePacket, GenerationVariant
from app.services.editorial_quality import assess_variant

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "reply_quality_golden.json"
FIXTURES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_reply_golden_catalog_has_thirty_public_safe_cases() -> None:
    assert len(FIXTURES) >= 30
    assert len({fixture["id"] for fixture in FIXTURES}) == len(FIXTURES)
    assert all(fixture["mode"] == "reply" for fixture in FIXTURES)
    assert all("expected_issue_codes" in fixture for fixture in FIXTURES)
    assert all("known_bad_output" in fixture for fixture in FIXTURES)


def test_reply_golden_catalog_covers_all_new_plainness_categories() -> None:
    expected = {
        "UNSUPPORTED_SUPERLATIVE",
        "IMAGINARY_AWARENESS_GAP",
        "METAPHOR_STACKING",
        "INSIGHT_PACKAGING",
        "HISTORICAL_CONTRAST_FRAME",
        "REPLY_OVERPRODUCTION",
        "REDUNDANT_PARAGRAPH",
        "UNSUPPORTED_STAKES",
        "OVERCOMPRESSED_THEORY",
        "ABSTRACTION_LADDER",
        "POLISHED_CAUSAL_ARC",
        "UNNECESSARY_METAPHOR",
        "UNSUPPORTED_GROUP_GENERALIZATION",
        "FALSE_CASUALNESS",
        "CONCRETE_TO_ABSTRACT_DRIFT",
    }
    covered = {code for fixture in FIXTURES for code in fixture["expected_issue_codes"]}
    assert expected <= covered


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture["id"])
def test_reply_golden_fixture_matches_expected_issue_codes(fixture: dict[str, object]) -> None:
    source = str(fixture["source"])
    evidence = str(fixture["evidence"])
    text = str(fixture["known_bad_output"])
    packet = EvidencePacket(
        mode=GenerationMode.REPLY,
        topic=source,
        items=(
            EvidenceItem(
                evidence_id="source",
                source_type="x_post",
                text=f"{source} {evidence}",
                verification_status="direct_source",
            ),
        ),
        language="en",
        requested_format="reply",
    )
    variant = GenerationVariant(
        label="golden",
        text=text,
        similarity_risk=0.0,
        angle_id="angle",
        angle_type="plain_update",
        evidence_ids=("source",),
        confidence="medium",
        factual_claims=(text,),
        attributed_claims=(),
        uncertainty_markers=(),
    )

    report = assess_variant(
        variant,
        evidence_packet=packet,
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )
    actual = {issue.code for issue in report.issues}
    expected = set(fixture["expected_issue_codes"])

    assert expected <= actual
    if not expected:
        blocking_plainness_codes = {
            "UNSUPPORTED_SUPERLATIVE",
            "IMAGINARY_AWARENESS_GAP",
            "METAPHOR_STACKING",
            "INSIGHT_PACKAGING",
            "HISTORICAL_CONTRAST_FRAME",
            "REPLY_OVERPRODUCTION",
            "REDUNDANT_PARAGRAPH",
            "UNSUPPORTED_STAKES",
            "OVERCOMPRESSED_THEORY",
            "ABSTRACTION_LADDER",
            "POLISHED_CAUSAL_ARC",
            "UNNECESSARY_METAPHOR",
            "UNSUPPORTED_GROUP_GENERALIZATION",
            "FALSE_CASUALNESS",
            "CONCRETE_TO_ABSTRACT_DRIFT",
        }
        assert not (actual & blocking_plainness_codes)
        assert report.plainness_score >= 0.8
        assert report.reply_fit_score >= 0.8
