from __future__ import annotations

import pytest

from app.domain.enums import ContentType
from app.schemas.content import GenerationVariant, UntrustedSourceData
from app.services.editorial_quality import assess_variant, inspect_reply_naturalness
from app.services.generation_pipeline import build_generation_context
from app.services.semantic_classifier import inspect_humor_intent, inspect_humor_safety


def _context(source: str):
    return build_generation_context(
        idea_summary="Reply to source",
        idea_explanation={"origin": "reply_to_source", "editorial_intent": "reply_reaction"},
        language="en",
        sources=(
            UntrustedSourceData(
                source_id="source",
                source_type="x_post",
                content=source,
            ),
        ),
        content_type=ContentType.REPLY,
    )


def _variant(
    text: str,
    *,
    contribution_type: str = "plain_observation",
    angle_type: str = "plain_update",
) -> GenerationVariant:
    return GenerationVariant(
        label="reply",
        text=text,
        similarity_risk=0.0,
        angle_id="reply",
        angle_type=angle_type,
        evidence_ids=("source",),
        confidence="high",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
        contribution_id="reply",
        contribution_type=contribution_type,
    )


def test_tragedy_joke_is_blocked_even_when_mislabeled_plain() -> None:
    context = _context("A hospital reported 10 deaths after the outage.")
    variant = _variant("the outage is doing heroic amounts of work")

    naturalness = inspect_reply_naturalness(variant, evidence_packet=context.evidence)
    report = assess_variant(
        variant,
        evidence_packet=context.evidence,
        editorial_intent="reply_reaction",
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert naturalness.humor_intent is not None
    assert naturalness.humor_intent.text_appears_humorous
    assert naturalness.humor_intent.humor_safety_required
    assert naturalness.humor_intent.extraction_conflict
    assert naturalness.humor_safety is not None
    assert not naturalness.humor_safety.suitable_for_humor
    assert not report.passed
    assert {"HUMOR_ON_TRAGEDY", "REPLY_TYPE_MISMATCH"} <= {issue.code for issue in report.issues}


def test_metadata_cannot_suppress_humor_intent() -> None:
    context = _context("A hospital reported 10 deaths after the outage.")
    inspection = inspect_humor_intent(
        context.evidence,
        "the outage really committed to the bit",
        metadata_says_humor=False,
    )

    assert not inspection.metadata_says_humor
    assert inspection.text_appears_humorous
    assert inspection.humor_safety_required
    assert inspection.extraction_conflict


def test_plain_text_without_humorous_function_is_not_reclassified() -> None:
    context = _context("A hospital reported 10 deaths after the outage.")
    variant = _variant("the outage and the response need a serious independent review")

    naturalness = inspect_reply_naturalness(variant, evidence_packet=context.evidence)

    assert naturalness.humor_intent is not None
    assert not naturalness.humor_intent.text_appears_humorous
    assert not naturalness.humor_intent.humor_safety_required
    assert naturalness.humor_safety is None


@pytest.mark.parametrize(
    "source",
    (
        "Ten people lost their lives after the outage.",
        "The founder passed away after a medical emergency.",
        "The attack left twenty people wounded.",
        "An explosion injured dozens of workers.",
        "A blast injured workers at the plant.",
    ),
)
def test_common_harm_language_is_sensitive(source: str) -> None:
    context = _context(source)
    safety = inspect_humor_safety(context.evidence, "achievement unlocked")

    assert safety.sensitive_context
    assert not safety.suitable_for_humor
    assert "HUMOR_ON_TRAGEDY" in safety.issue_codes


@pytest.mark.parametrize(
    "source",
    (
        "The app crash was fixed in version 2.1.",
        "The server crash was fixed after the patch.",
        "The browser crash no longer reproduces.",
        "The market crash erased the morning gains.",
        "The crypto crash cut prices overnight.",
    ),
)
def test_non_human_crash_context_is_not_tragedy(source: str) -> None:
    context = _context(source)
    safety = inspect_humor_safety(context.evidence, "finally a fix for the actual crash")

    assert not safety.sensitive_context
    assert safety.suitable_for_humor
    assert safety.context_confidence == "high"
    assert safety.disambiguation_reason is not None


@pytest.mark.parametrize(
    "source",
    (
        "Five people died in a car crash.",
        "A fatal plane crash was reported overnight.",
        "The train crash injured twelve passengers.",
    ),
)
def test_vehicle_or_fatal_crash_is_sensitive(source: str) -> None:
    context = _context(source)
    safety = inspect_humor_safety(context.evidence, "traffic committed to the bit")

    assert safety.sensitive_context
    assert not safety.suitable_for_humor


def test_ambiguous_sensitive_term_fails_closed_only_for_humor() -> None:
    context = _context("The attack damaged the facility overnight.")
    humorous = inspect_reply_naturalness(
        _variant("the facility really committed to the bit"),
        evidence_packet=context.evidence,
    )
    plain = inspect_reply_naturalness(
        _variant("the damage needs a clear account of what happened"),
        evidence_packet=context.evidence,
    )

    assert humorous.humor_safety is not None
    assert humorous.humor_safety.ambiguous_sensitive_term
    assert not humorous.humor_safety.suitable_for_humor
    assert plain.humor_safety is None


@pytest.mark.parametrize(
    ("recent", "candidate", "family"),
    (
        (
            "nothing says simplicity like six screens",
            "nothing screams simplicity like eight screens",
            "inverse_praise",
        ),
        (
            "fundraising is just sales with extra steps",
            "onboarding is paperwork with better branding",
            "reductive_identity",
        ),
        (
            "fundraising is just sales with extra steps",
            "onboarding, translated: paperwork with a progress bar",
            "reductive_identity",
        ),
        (
            "fundraising is just sales with extra steps",
            "onboarding is the same thing as paperwork, just with nicer copy",
            "reductive_identity",
        ),
    ),
)
def test_additional_bounded_shells_match_recent_history(
    recent: str,
    candidate: str,
    family: str,
) -> None:
    inspection = inspect_reply_naturalness(
        _variant(candidate),
        recent_posts=(recent,),
    )

    assert family in inspection.structural_families
    assert inspection.recent_syntax_reuse
    assert inspection.recent_structural_match


def test_structural_detection_is_independent_of_provider_label() -> None:
    recent = ("nothing says simplicity like six screens",)
    text = "nothing screams simplicity like eight screens"
    plain = inspect_reply_naturalness(_variant(text), recent_posts=recent)
    humor = inspect_reply_naturalness(
        _variant(text, contribution_type="dry_humor", angle_type="concise_joke"),
        recent_posts=recent,
    )

    assert plain.structural_families == humor.structural_families
    assert plain.recent_syntax_reuse
    assert humor.recent_syntax_reuse
