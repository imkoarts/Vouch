from __future__ import annotations

import pytest

from app.domain.enums import GenerationMode
from app.schemas.content import EvidenceItem, EvidencePacket, GenerationVariant
from app.services.editorial_quality import assess_variant


def _report(
    *,
    source: str,
    text: str,
    angle_type: str = "concise_joke",
    recent_posts: tuple[str, ...] = (),
):
    packet = EvidencePacket(
        mode=GenerationMode.REPLY,
        topic="micro reply",
        items=(
            EvidenceItem(
                evidence_id="source",
                source_type="x_post",
                text=source,
                verification_status="direct_source",
            ),
        ),
        language="en",
        requested_format="reply",
    )
    variant = GenerationVariant(
        label="reply",
        text=text,
        similarity_risk=0.0,
        angle_id="reply-angle",
        angle_type=angle_type,
        evidence_ids=("source",),
        confidence="medium",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
    )
    return assess_variant(variant, evidence_packet=packet, recent_posts=recent_posts)


@pytest.mark.parametrize(
    ("source", "text", "angle_type"),
    [
        (
            "Polymarket users paid extra fees while farming a possible airdrop.",
            "turns out airdrop farming was just paying extra fees with a hopeful name",
            "dry_observation",
        ),
        (
            "The AI model now has a low verbosity setting.",
            "finally an official low-verbosity anti-yap switch",
            "concise_joke",
        ),
        (
            "A joke post claims early access to GPT-6 and GTA 6.",
            "bro got GPT-6 and GTA 6 access in one tweet",
            "concise_joke",
        ),
        (
            "The blueprint is available for free, but consistency remains difficult.",
            "the blueprint is free. unfortunately consistency still has no download button",
            "concise_joke",
        ),
        (
            "Two accounts both posted solana summer.",
            "two accounts have now confirmed solana summer",
            "concise_joke",
        ),
        (
            "A third-party growth analytics app could compromise an X account.",
            "the growth app is always called something trustworthy like 'analytics pro'",
            "dry_observation",
        ),
    ],
)
def test_approved_micro_reply_shapes_are_not_flattened(
    source: str, text: str, angle_type: str
) -> None:
    report = _report(source=source, text=text, angle_type=angle_type)

    assert report.passed
    assert "UNNECESSARY_METAPHOR" not in {issue.code for issue in report.issues}
    assert "UNSUPPORTED_FACT" not in {issue.code for issue in report.issues}


def test_source_specific_joke_does_not_need_a_factual_claim_boundary() -> None:
    report = _report(
        source="The model added a low verbosity setting.",
        text="finally a low-verbosity anti-yap switch",
    )

    assert report.passed
    assert report.evidence_score == 1.0


def test_humor_label_does_not_bypass_a_believable_false_accusation() -> None:
    report = _report(
        source="A growth analytics app requests access to an X account.",
        text="the growth app stole everyone's passwords",
    )

    assert not report.passed
    assert "UNSUPPORTED_FACT" in {issue.code for issue in report.issues}


def test_punchline_exception_does_not_allow_analytical_insight_packaging() -> None:
    report = _report(
        source="The workflow stores evals and corrections.",
        text="the evals are the operating system of organizational judgment",
    )

    assert not report.passed
    assert "INSIGHT_PACKAGING" in {issue.code for issue in report.issues}


def test_one_clearly_humorous_metaphor_is_allowed_in_micro_reply() -> None:
    report = _report(
        source="The product added a button for managing the feedback loop.",
        text="finally a button to own the loop",
    )

    codes = {issue.code for issue in report.issues}
    assert report.passed
    assert "UNNECESSARY_METAPHOR" not in codes
    assert "METAPHOR_STACKING" not in codes


def test_turns_out_and_finally_rotate_against_last_five_posts() -> None:
    report = _report(
        source="Airdrop farmers paid more fees.",
        text="turns out airdrop farming was the fee product",
        angle_type="dry_observation",
        recent_posts=(
            "old post one",
            "turns out the strategy was leaving Robinhood",
            "old post three",
        ),
    )

    assert "RECENT_OPENING_SIMILARITY" in {issue.code for issue in report.issues}
