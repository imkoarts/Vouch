from __future__ import annotations

import pytest

from app.schemas.content import GenerationVariant
from app.services.editorial_quality import assess_variant


def _report(text: str, *, recent_posts: tuple[str, ...] = ()):
    variant = GenerationVariant(
        label="test",
        text=text,
        similarity_risk=0.0,
        angle_id="a",
        angle_type="plain_update",
        evidence_ids=("e",),
        confidence="medium",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
    )
    return assess_variant(variant, recent_posts=recent_posts)


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("The part companies will realize too late: the model is not the moat.", "REVERSAL_HOOK"),
        ("This is not about the API. It is about ownership.", "MANUFACTURED_CONTRAST"),
        ("This changes everything for the industry.", "ARTIFICIAL_STAKES"),
        ("Read that again. The future is already here.", "FAKE_DEPTH"),
        ("Execution will be key.", "PLACEHOLDER_COMMENTARY"),
        ("In a major development, the move signals a new era.", "MEDIA_BRAND_VOICE"),
        ("This could reshape the industry and drive adoption.", "PREDICTIVE_VAGUENESS"),
        ("Here is what founders need to understand.", "CREATOR_COACH_VOICE"),
        ("The implementation of the optimization improved latency.", "NOMINALIZATION_OVERLOAD"),
        ("No hype. No noise. Just execution.", "PERFORMATIVE_FRAGMENTS"),
    ],
)
def test_taxonomy_detects_rhetorical_overcomposition(text: str, code: str) -> None:
    assert code in {issue.code for issue in _report(text).issues}


def test_recent_corpus_checks_repeat_opening_and_ending() -> None:
    report = _report(
        "The interesting part is the export path. Own the loop.",
        recent_posts=("The interesting part is the fee path. Own the loop.",),
    )
    codes = {issue.code for issue in report.issues}
    assert "RECENT_OPENING_SIMILARITY" in codes
    assert "RECENT_ENDING_SIMILARITY" in codes


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("The part that matters most is the correction history.", "UNSUPPORTED_SUPERLATIVE"),
        (
            "Most companies will understand this only after switching providers.",
            "IMAGINARY_AWARENESS_GAP",
        ),
        (
            "The feedback loop is institutional memory, and letting it leave the building "
            "hands over the keys to the company.",
            "METAPHOR_STACKING",
        ),
        ("Evals are the operating system of organizational judgment.", "INSIGHT_PACKAGING"),
        (
            "Companies spent a decade securing data and are now casually giving away prompts.",
            "HISTORICAL_CONTRAST_FRAME",
        ),
        ("This is far more dangerous than teams think.", "UNSUPPORTED_STAKES"),
        (
            "The API stores evals, corrections, traces, and workflow context as organizational "
            "intelligence.",
            "OVERCOMPRESSED_THEORY",
        ),
        (
            (
                "The API stores evals. That correction history becomes company learning "
                "and then a moat."
            ),
            "CONCRETE_TO_ABSTRACT_DRIFT",
        ),
        (
            "basically, the feedback loop is the company learning out loud.",
            "FALSE_CASUALNESS",
        ),
        ("Companies should make this the baseline for every AI contract.", "UNSUPPORTED_INFERENCE"),
    ],
)
def test_plainness_taxonomy_detects_remaining_insight_content(text: str, code: str) -> None:
    assert code in {issue.code for issue in _report(text).issues}
