from __future__ import annotations

from app.domain.enums import GenerationMode
from app.schemas.content import EvidenceItem, EvidencePacket, GenerationVariant, QualityReport
from app.services.editorial_quality import assess_variant


def _report(
    text: str,
    *,
    mode: GenerationMode = GenerationMode.TREND_CONTEXT,
    evidence_text: str = "The source describes corrections, evals, and model-provider workflows.",
    factual_claims: tuple[str, ...] = (),
    uncertainty_markers: tuple[str, ...] = (),
) -> QualityReport:
    packet = EvidencePacket(
        mode=mode,
        topic="AI model portability",
        items=(
            EvidenceItem(
                evidence_id="evidence",
                source_type="x_post",
                text=evidence_text,
                verification_status="single_source",
            ),
        ),
        language="en",
        requested_format="reply" if mode is GenerationMode.REPLY else "short_post",
    )
    return assess_variant(
        GenerationVariant(
            label="regression",
            text=text,
            similarity_risk=0.0,
            angle_id="angle",
            angle_type="plain_update",
            evidence_ids=("evidence",),
            confidence="low",
            factual_claims=factual_claims,
            attributed_claims=(),
            uncertainty_markers=uncertainty_markers,
        ),
        evidence_packet=packet,
    )


def _rejected_codes(text: str) -> set[str]:
    report = _report(text)
    assert not report.passed
    return {issue.code for issue in report.issues}


def test_polymarket_topic_label_regression_is_rejected() -> None:
    codes = _rejected_codes(
        "If Polymarket TGE happens, details will matter more than the noise around it. "
        "Criteria, snapshots, distribution — that is what trust depends on."
    )
    assert {"CONDITIONAL_ESCAPE", "PLACEHOLDER_COMMENTARY"} <= codes
    assert "MECHANICAL_TRICOLON" in codes


def test_hormuz_verification_boilerplate_is_rejected() -> None:
    codes = _rejected_codes(
        "There is a report that the US military announced the opening of the Strait. "
        "The information requires independent verification."
    )
    assert {"VAGUE_ATTRIBUTION", "GENERIC_VERIFICATION"} <= codes


def test_ai_lock_in_overcomposition_is_rejected_for_scope_and_rhetoric() -> None:
    report = _report(
        "The part most companies will realize too late: AI lock-in is not really the model. "
        "It is the months of corrections, evals, traces and workflow context. "
        "You can swap an API in a weekend. You cannot export a year of encoded judgment. "
        "Own it, keep it portable, and never let one provider become the only place the company "
        "gets smarter."
    )
    codes = {issue.code for issue in report.issues}
    assert not report.passed
    assert {
        "REVERSAL_HOOK",
        "IMAGINARY_AWARENESS_GAP",
        "MECHANICAL_SYMMETRY",
        "MECHANICAL_TRICOLON",
        "CONCRETE_TO_ABSTRACT_DRIFT",
    } <= codes


def test_reply_correction_history_regression_triggers_plainness_gate() -> None:
    report = _report(
        "companies spent years locking down their data and are now casually exporting the part "
        "that matters most: how their people correct, judge and improve the model.\n\n"
        "that correction history is basically the company learning out loud. letting it leave "
        "the building feels a lot riskier than most teams realize.",
        mode=GenerationMode.REPLY,
        evidence_text=(
            "Model requests can include prompts, corrections, and evals that shape expected "
            "behavior."
        ),
        factual_claims=("Model requests can include corrections and evals.",),
    )
    codes = {issue.code for issue in report.issues}
    assert not report.passed
    assert {
        "HISTORICAL_CONTRAST_FRAME",
        "UNSUPPORTED_SUPERLATIVE",
        "MECHANICAL_TRICOLON",
        "INSIGHT_PACKAGING",
        "METAPHOR_STACKING",
        "IMAGINARY_AWARENESS_GAP",
        "UNSUPPORTED_STAKES",
        "REPLY_OVERPRODUCTION",
        "REDUNDANT_PARAGRAPH",
        "FALSE_CASUALNESS",
        "CONCRETE_TO_ABSTRACT_DRIFT",
    } <= codes
    assert report.plainness_score < 0.5
    assert report.rhetorical_density_score > 0.5
    assert report.reply_fit_score < 0.5


def test_already_natural_literal_reply_is_not_over_rewritten() -> None:
    report = _report(
        "The evals are probably harder to move than the API integration.",
        mode=GenerationMode.REPLY,
        evidence_text="Switching APIs is easier than moving a year of evals and corrections.",
        factual_claims=("The evals are harder to move than the API integration.",),
        uncertainty_markers=("probably",),
    )
    assert report.passed
    assert report.plainness_score == 1.0
    assert report.reply_fit_score == 1.0
    assert report.issues == ()


def test_evidence_supported_superlative_is_not_rejected() -> None:
    report = _report(
        "The hardest benchmark in the supplied results was LongBench.",
        evidence_text="LongBench was the hardest benchmark in the supplied results.",
        factual_claims=("LongBench was the hardest benchmark.",),
    )
    assert "UNSUPPORTED_SUPERLATIVE" not in {issue.code for issue in report.issues}


def test_domain_ranking_terms_are_not_treated_as_rhetorical_superlatives() -> None:
    report = _report(
        "The main function reads a primary key before running the hardest benchmark.",
        evidence_text="The main function reads a primary key before running the hardest benchmark.",
    )
    assert "UNSUPPORTED_SUPERLATIVE" not in {issue.code for issue in report.issues}


def test_evidence_supported_group_awareness_claim_is_contextual() -> None:
    report = _report(
        "Most teams in the survey said they did not understand the migration requirement.",
        evidence_text=(
            "Most teams in the survey said they did not understand the migration requirement."
        ),
    )
    codes = {issue.code for issue in report.issues}
    assert "IMAGINARY_AWARENESS_GAP" not in codes
    assert "UNSUPPORTED_GROUP_GENERALIZATION" not in codes


def test_one_metaphor_is_not_automatically_a_metaphor_stack() -> None:
    report = _report(
        "The eval history is institutional memory.",
        evidence_text="The eval history records prior expected outputs.",
    )
    codes = {issue.code for issue in report.issues}
    assert "INSIGHT_PACKAGING" in codes
    assert "METAPHOR_STACKING" not in codes


def test_plain_lowercase_reply_does_not_trigger_false_casualness() -> None:
    report = _report(
        "the eval history may need its own migration path.",
        mode=GenerationMode.REPLY,
        evidence_text="The eval history may need a separate migration path.",
        uncertainty_markers=("may",),
    )
    assert "FALSE_CASUALNESS" not in {issue.code for issue in report.issues}


def test_correction_history_regression_has_a_short_literal_revision_target() -> None:
    before = (
        "companies spent years locking down their data and are now casually exporting the part "
        "that matters most: how their people correct, judge and improve the model.\n\n"
        "that correction history is basically the company learning out loud. letting it leave "
        "the building feels a lot riskier than most teams realize."
    )
    after = "The corrections and evals are part of the migration when a company changes providers."
    report = _report(
        after,
        mode=GenerationMode.REPLY,
        evidence_text=(
            "Changing providers can require moving the corrections and evals used to shape "
            "expected model behavior."
        ),
        factual_claims=("Changing providers can require moving corrections and evals.",),
    )
    forbidden = {
        "HISTORICAL_CONTRAST_FRAME",
        "UNSUPPORTED_SUPERLATIVE",
        "MECHANICAL_TRICOLON",
        "INSIGHT_PACKAGING",
        "METAPHOR_STACKING",
        "IMAGINARY_AWARENESS_GAP",
        "UNSUPPORTED_STAKES",
        "REPLY_OVERPRODUCTION",
        "REDUNDANT_PARAGRAPH",
        "FALSE_CASUALNESS",
        "CONCRETE_TO_ABSTRACT_DRIFT",
    }

    assert not ({issue.code for issue in report.issues} & forbidden)
    assert "\n\n" not in after
    assert len([part for part in after.split(".") if part.strip()]) <= 2
    assert len(after) < len(before)
