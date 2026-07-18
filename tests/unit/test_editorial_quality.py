"""Editorial quality gate regression tests."""

from app.schemas.content import GenerationVariant
from app.services.editorial_quality import assess_variant, repair_feedback


def _variant(text: str) -> GenerationVariant:
    return GenerationVariant(
        label="candidate",
        text=text,
        character_count=0,
        similarity_risk=0.0,
        angle_id="angle-1",
        angle_type="plain_update",
        evidence_ids=("evidence-1",),
        confidence="medium",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
    )


def test_editorial_gate_rejects_stock_ai_summary_language() -> None:
    assessment = assess_variant(
        _variant(
            "There is a report that a launch may happen. Details matter more than hype, "
            "and it remains to be seen what comes next."
        )
    )

    assert not assessment.acceptable
    assert "stock_opening" in assessment.blocking_flags
    assert "noise_cliche" in assessment.blocking_flags
    assert "generic_takeaway" in assessment.blocking_flags


def test_editorial_gate_accepts_specific_source_backed_angle() -> None:
    assessment = assess_variant(
        _variant(
            "A token launch without published eligibility rules turns every user action into a "
            "guess. Polymarket can avoid that by defining the snapshot date before incentives "
            "change behavior."
        )
    )

    assert assessment.acceptable
    assert assessment.blocking_flags == ()


def test_repair_feedback_names_failed_patterns_without_copying_source_text() -> None:
    bad = assess_variant(_variant("If this happens, details matter more than hype."))

    feedback = repair_feedback((bad,))

    assert "CONDITIONAL_ESCAPE" in feedback
    assert "PLACEHOLDER_COMMENTARY" in feedback
    assert "stock_opening" not in feedback
    assert "Fix only the listed issues" in feedback
    assert "Keep usable concrete sentences unchanged" in feedback
    assert "revision_status=insufficient_context" in feedback


def test_editorial_gate_rejects_report_about_report_personal_post() -> None:
    assessment = assess_variant(
        _variant(
            "One report says Jannik Sinner lost the opening set before defeating "
            "Alexander Zverev and defending his Wimbledon title. If that account is "
            "accurate, the match's turning point was not the start but the response after "
            "it. The result is specific; the report does not explain which changes in play "
            "produced the comeback."
        )
    )

    codes = {issue.code for issue in assessment.issues}
    assert not assessment.passed
    assert {
        "VAGUE_ATTRIBUTION",
        "MEDIA_BRAND_VOICE",
        "GENERIC_VERIFICATION",
        "CONDITIONAL_ESCAPE",
        "EXPLANATION_LEAKAGE",
        "SOURCE_SUMMARY_ONLY",
    } <= codes
    assert assessment.plainness_score < 1.0


def test_editorial_gate_accepts_direct_literal_sports_result() -> None:
    assessment = assess_variant(
        _variant("Jannik Sinner lost the opening set and still beat Alexander Zverev.")
    )

    codes = {issue.code for issue in assessment.issues}
    assert "VAGUE_ATTRIBUTION" not in codes
    assert "GENERIC_VERIFICATION" not in codes
    assert "EXPLANATION_LEAKAGE" not in codes
    assert "MEDIA_BRAND_VOICE" not in codes


def test_source_quality_angle_may_discuss_a_specific_source_limit() -> None:
    variant = _variant(
        "The match report gives the result but does not include point-by-point data."
    ).model_copy(update={"angle_type": "source_quality"})

    assessment = assess_variant(variant)
    codes = {issue.code for issue in assessment.issues}

    assert "EXPLANATION_LEAKAGE" not in codes
    assert "SOURCE_SUMMARY_ONLY" not in codes


def test_editorial_gate_treats_insufficient_context_copy_as_empty_thesis() -> None:
    assessment = assess_variant(_variant("insufficient context."))

    codes = {issue.code for issue in assessment.issues}
    assert not assessment.passed
    assert "EMPTY_THESIS" in codes
    issue = next(issue for issue in assessment.issues if issue.code == "EMPTY_THESIS")
    assert issue.suggested_action == "regenerate_angle"
