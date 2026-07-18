from __future__ import annotations

from app.schemas.content import GenerationVariant
from app.services.editorial_quality import assess_variant


def _variant(text: str) -> GenerationVariant:
    return GenerationVariant(
        label="semantic-lint",
        text=text,
        character_count=len(text),
        similarity_risk=0.0,
        angle_id="angle-1",
        angle_type="plain_update",
        evidence_ids=("evidence-1",),
        confidence="medium",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
    )


def test_semantic_lint_detects_announcement_before_content() -> None:
    report = assess_variant(
        _variant("The interesting part is what happened next. Renewals fell after launch.")
    )

    assert "ANNOUNCEMENT_BEFORE_CONTENT" in {issue.code for issue in report.issues}


def test_semantic_lint_detects_false_agency_and_hidden_actor() -> None:
    report = assess_variant(_variant("The market decided the feature was unnecessary."))
    codes = {issue.code for issue in report.issues}

    assert {"FALSE_AGENCY", "HIDDEN_ACTOR", "ACTORLESS_CAUSALITY"} <= codes


def test_semantic_lint_keeps_literal_technical_subject() -> None:
    report = assess_variant(_variant("The API returned a 429 after the third request."))
    codes = {issue.code for issue in report.issues}

    assert "FALSE_AGENCY" not in codes
    assert "HIDDEN_ACTOR" not in codes


def test_semantic_lint_detects_unasked_reassurance() -> None:
    report = assess_variant(_variant("The first version can be rough, and that's okay."))
    codes = {issue.code for issue in report.issues}

    assert {
        "UNASKED_REASSURANCE",
        "PERMISSION_GRANTING_ENDING",
        "EMOTIONAL_HAND_HOLDING",
    } <= codes
