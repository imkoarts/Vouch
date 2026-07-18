from __future__ import annotations

from app.domain.enums import GenerationMode
from app.schemas.content import EvidenceItem, EvidencePacket, GenerationVariant
from app.services.editorial_quality import assess_variant


def _reply(text: str):
    packet = EvidencePacket(
        mode=GenerationMode.REPLY,
        topic="model portability",
        items=(
            EvidenceItem(
                evidence_id="source",
                source_type="x_post",
                text="Switching providers may require moving evals and corrections.",
                verification_status="single_source",
            ),
        ),
        language="en",
        requested_format="reply",
    )
    variant = GenerationVariant(
        label="reply",
        text=text,
        similarity_risk=0.0,
        angle_id="angle",
        angle_type="mechanism",
        evidence_ids=("source",),
        confidence="medium",
        factual_claims=("Moving evals and corrections can be part of switching providers.",),
        attributed_claims=(),
        uncertainty_markers=("may",),
    )
    return assess_variant(variant, evidence_packet=packet)


def test_reply_overproduction_is_blocked() -> None:
    report = _reply(
        "Companies spent years treating data as an asset. Now the correction history is the "
        "company learning out loud.\n\nThat creates a larger risk than most teams realize. "
        "Every company should own the loop."
    )
    codes = {issue.code for issue in report.issues}
    assert not report.passed
    assert "REPLY_OVERPRODUCTION" in codes
    assert "METAPHOR_STACKING" in codes
    assert "IMAGINARY_AWARENESS_GAP" in codes


def test_two_sentence_literal_reply_passes_plainness_checks() -> None:
    report = _reply("Evals may move harder than APIs. That complicates switching.")
    forbidden = {
        "REPLY_OVERPRODUCTION",
        "INSIGHT_PACKAGING",
        "METAPHOR_STACKING",
        "IMAGINARY_AWARENESS_GAP",
        "UNSUPPORTED_SUPERLATIVE",
    }
    assert not ({issue.code for issue in report.issues} & forbidden)
    assert report.plainness_score >= 0.8
    assert report.reply_fit_score >= 0.8


def test_reply_score_penalizes_rhetoric_more_than_literal_control() -> None:
    literal = _reply("The evals may be harder to move than the API integration.")
    rhetorical = _reply(
        "The real moat is not the API. It is the institutional memory inside the feedback loop."
    )
    assert literal.plainness_score > rhetorical.plainness_score
    assert literal.rhetorical_density_score < rhetorical.rhetorical_density_score
    assert literal.reply_fit_score > rhetorical.reply_fit_score
