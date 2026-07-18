from __future__ import annotations

import pytest

from app.domain.enums import ContentType
from app.schemas.content import GenerationVariant, UntrustedSourceData
from app.services.editorial_quality import inspect_reply_naturalness, inspect_reply_shell
from app.services.generation_pipeline import build_generation_context
from app.services.semantic_classifier import (
    inspect_human_harm_scope,
    inspect_humor_intent,
    inspect_humor_safety,
)


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


def _variant(text: str, *, contribution_type: str = "plain_observation") -> GenerationVariant:
    return GenerationVariant(
        label="reply",
        text=text,
        similarity_risk=0.0,
        angle_id="reply",
        angle_type="plain_update",
        evidence_ids=("source",),
        confidence="high",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
        contribution_id="reply",
        contribution_type=contribution_type,
    )


@pytest.mark.parametrize(
    "text",
    (
        "nice work from the outage",
        "beautiful work from the outage",
        "exactly what the hospital needed",
        "could not have asked for a better outage",
        "10/10 execution from the outage",
        "a flawless showing from the grid",
        "good stuff from the outage",
        "the outage deserves a raise",
        "the outage earned employee of the month",
        "just what the hospital needed",
        "couldn't have asked for a better outage",
        "10 out of 10 execution from the outage",
        "the outage should get a medal",
        "the outage wins employee of the month",
    ),
)
def test_bounded_evaluative_relations_detect_sarcastic_praise(text: str) -> None:
    context = _context("A hospital reported 10 deaths after the outage.")
    intent = inspect_humor_intent(context.evidence, text, metadata_says_humor=False)
    safety = inspect_humor_safety(context.evidence, text)

    assert intent.text_appears_humorous or "HUMOR_INTENT_UNCERTAIN" in intent.issue_codes
    assert intent.humor_safety_required
    assert intent.evaluative_irony is not None
    assert intent.evaluative_irony.positive_evaluation
    assert intent.evaluative_irony.negative_or_harmful_source_context
    assert intent.evaluative_irony.source_event_polarity == "harmful"
    assert intent.evaluative_irony.evaluation_polarity in {"positive", "rewarding"}
    assert intent.evaluative_irony.evaluation_target
    assert intent.evaluative_irony.target_source_alignment
    assert intent.evaluative_irony.evidence_spans
    assert not safety.suitable_for_humor


def test_literal_praise_of_positive_result_remains_allowed() -> None:
    context = _context("The hospital restored service safely and the outage response succeeded.")
    intent = inspect_humor_intent(
        context.evidence,
        "nice work from the response team",
        metadata_says_humor=False,
    )

    assert not intent.text_appears_humorous
    assert "HUMOR_INTENT_UNCERTAIN" not in intent.issue_codes
    assert intent.evaluative_irony is not None
    assert intent.evaluative_irony.literal_positive_reading_plausible


@pytest.mark.parametrize(
    "source",
    (
        "There were no survivors after the crash.",
        "The blast left no survivors.",
        "Ten people were fatally injured.",
        "The victim suffered mortal injuries.",
        "The fire raised the body count to ten.",
        "Ten bodies were recovered after the collapse.",
        "Three miners never made it out.",
        "The incident ended ten lives.",
        "A worker was left in critical condition.",
        "A child was found lifeless.",
    ),
)
def test_additional_human_harm_relations_are_explicit(source: str) -> None:
    scope = inspect_human_harm_scope(source)
    safety = inspect_humor_safety(_context(source).evidence, "achievement unlocked")

    assert scope.harm_referent == "human"
    assert scope.literal_harm
    assert scope.confidence == "high"
    assert scope.evidence_spans
    assert safety.sensitive_context
    assert not safety.suitable_for_humor


@pytest.mark.parametrize(
    "source",
    (
        "The product died overnight.",
        "The startup died after funding dried up.",
        "The project died in committee.",
        "The meme died last week.",
        "The conversation died immediately.",
        "The deal was killed by regulators.",
        "The proposal was killed in committee.",
        "The brand is dead.",
        "The network died for ten minutes.",
        "The chain died at block 100.",
        "The token died after liquidity vanished.",
    ),
)
def test_clear_non_human_death_relations_are_not_tragedy(source: str) -> None:
    scope = inspect_human_harm_scope(source)
    safety = inspect_humor_safety(_context(source).evidence, "achievement unlocked")

    assert scope.harm_referent == "non_human"
    assert not scope.literal_harm
    assert scope.metaphorical_or_technical_usage
    assert scope.evidence_spans
    assert not safety.sensitive_context
    assert safety.suitable_for_humor


def test_mixed_non_human_and_human_death_relations_remain_sensitive() -> None:
    source = "The server died. Two patients died during the outage."
    scope = inspect_human_harm_scope(source)
    safety = inspect_humor_safety(_context(source).evidence, "achievement unlocked")

    assert scope.harm_referent == "human"
    assert scope.literal_harm
    assert scope.metaphorical_or_technical_usage
    assert safety.sensitive_context
    assert not safety.suitable_for_humor


@pytest.mark.parametrize(
    ("text", "expected_operator"),
    (
        ("onboarding is merely paperwork with better UX", "reduction"),
        ("onboarding is paperwork pretending to be product strategy", "decorated_identity"),
        ("onboarding is paperwork cosplaying as product strategy", "decorated_identity"),
        ("onboarding is paperwork, rebranded", "decorated_identity"),
        ("onboarding equals paperwork plus a progress bar", "decorated_identity"),
        ("strip away the branding: onboarding is paperwork", "reduction"),
        ("onboarding is paperwork in product clothing", "decorated_identity"),
    ),
)
def test_relation_shell_variants_match_recent_reduction(
    text: str,
    expected_operator: str,
) -> None:
    shell = inspect_reply_shell(text)
    inspection = inspect_reply_naturalness(
        _variant(text),
        recent_posts=("fundraising is just sales with extra steps",),
    )

    assert shell.operator == expected_operator
    assert shell.subject_span
    assert shell.complement_span
    assert inspection.recent_syntax_reuse
    assert "reductive_identity" in inspection.structural_families


@pytest.mark.parametrize(
    "text",
    (
        "apparently onboarding was paperwork all along",
        "who knew onboarding was merely paperwork",
    ),
)
def test_reveal_operator_variants_share_reveal_family(text: str) -> None:
    inspection = inspect_reply_naturalness(
        _variant(text),
        recent_posts=("turns out onboarding was just paperwork",),
    )

    assert inspection.reply_shell is not None
    assert inspection.reply_shell.operator == "reveal"
    assert inspection.recent_syntax_reuse


@pytest.mark.parametrize(
    "text",
    (
        "nothing captures simplicity like six approval screens",
        "few things embody simplicity better than six approval screens",
    ),
)
def test_inverse_praise_operator_variants_share_family(text: str) -> None:
    inspection = inspect_reply_naturalness(
        _variant(text),
        recent_posts=("nothing says simplicity like six screens",),
    )

    assert inspection.reply_shell is not None
    assert inspection.reply_shell.operator == "inverse_praise"
    assert inspection.recent_syntax_reuse


def test_fresh_structure_with_same_source_nouns_is_not_blocked() -> None:
    text = "the extra onboarding paperwork adds a measurable completion cost"
    inspection = inspect_reply_naturalness(
        _variant(text),
        recent_posts=("fundraising is just sales with extra steps",),
    )

    assert not inspection.recent_syntax_reuse


def test_literal_equation_is_not_a_repackaging_shell() -> None:
    shell = inspect_reply_shell("revenue equals price plus tax")

    assert shell.operator == "none"


def test_literal_strip_away_definition_is_not_a_reduction_shell() -> None:
    shell = inspect_reply_shell("strip away the insulation: the cable is copper")

    assert shell.operator == "none"


def test_source_specific_nouns_do_not_suppress_relation_reuse() -> None:
    context = _context("Onboarding now uses paperwork and a progress bar.")
    text = "onboarding equals paperwork plus a progress bar"
    inspection = inspect_reply_naturalness(
        _variant(text),
        evidence_packet=context.evidence,
        recent_posts=("fundraising is just sales with extra steps",),
    )

    assert inspection.reply_shell is not None
    assert inspection.reply_shell.source_specific
    assert inspection.recent_syntax_reuse
