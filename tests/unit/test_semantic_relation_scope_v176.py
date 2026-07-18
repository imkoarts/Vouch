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
    ("text", "expected_relation"),
    (
        ("great timing from the outage", "positive_timing"),
        ("could not be prouder of the grid", "admiration"),
        ("a round of applause for the outage", "celebration"),
        ("five stars for the outage", "rating"),
        ("chef's kiss from the grid", "praise"),
        ("the outage outdid itself", "success_attribution"),
        ("the outage really showed up", "success_attribution"),
        ("the outage crushed it", "success_attribution"),
        ("big win for the outage", "success_attribution"),
        ("another triumph for the grid", "celebration"),
    ),
)
def test_v176_evaluative_relations_detect_sarcastic_praise(
    text: str,
    expected_relation: str,
) -> None:
    context = _context("A hospital reported 10 deaths after the outage.")
    intent = inspect_humor_intent(context.evidence, text, metadata_says_humor=False)
    safety = inspect_humor_safety(context.evidence, text)

    assert intent.text_appears_humorous
    assert intent.humor_safety_required
    assert intent.evaluative_irony is not None
    assert intent.evaluative_irony.evaluation_relation == expected_relation
    assert intent.evaluative_irony.positive_evaluation
    assert intent.evaluative_irony.target_source_alignment
    assert intent.evaluative_irony.implicit_sarcasm
    assert intent.evaluative_irony.confidence == "high"
    assert intent.evaluative_irony.evidence_spans
    assert not safety.suitable_for_humor


def test_v176_literal_praise_on_positive_source_remains_allowed() -> None:
    context = _context("The hospital safely restored power and completed the recovery.")
    intent = inspect_humor_intent(
        context.evidence,
        "great work from the response team",
        metadata_says_humor=False,
    )

    assert not intent.text_appears_humorous
    assert not intent.humor_safety_required
    assert intent.evaluative_irony is not None
    assert intent.evaluative_irony.literal_positive_reading_plausible
    assert not intent.evaluative_irony.implicit_sarcasm


def test_v176_unresolved_positive_target_on_harmful_source_fails_closed_for_humor() -> None:
    context = _context("A hospital reported 10 deaths after the outage.")
    intent = inspect_humor_intent(
        context.evidence,
        "five stars for that performance",
        metadata_says_humor=False,
    )

    assert not intent.text_appears_humorous
    assert intent.humor_safety_required
    assert "HUMOR_INTENT_UNCERTAIN" in intent.issue_codes
    assert intent.evaluative_irony is not None
    assert intent.evaluative_irony.positive_evaluation
    assert not intent.evaluative_irony.target_source_alignment


@pytest.mark.parametrize(
    ("source", "expected_state", "expected_event"),
    (
        ("Three workers were declared deceased.", "death", "unknown"),
        ("The worker remains in a coma.", "critical_condition", "medical"),
        ("The crash proved fatal for ten passengers.", "death", "physical_crash"),
        ("Ten passengers were lost in the crash.", "death", "physical_crash"),
        ("Ten people did not make it home after the crash.", "death", "physical_crash"),
        ("The incident left ten people on life support.", "critical_condition", "incident"),
    ),
)
def test_v176_additional_human_harm_relations_are_explicit(
    source: str,
    expected_state: str,
    expected_event: str,
) -> None:
    scope = inspect_human_harm_scope(source)
    safety = inspect_humor_safety(_context(source).evidence, "achievement unlocked")

    assert scope.harm_referent == "human"
    assert scope.harm_state == expected_state
    assert scope.event_type == expected_event
    assert scope.literal_harm
    assert scope.literal_or_metaphorical_usage == "literal"
    assert scope.confidence == "high"
    assert scope.evidence_spans
    assert safety.sensitive_context
    assert not safety.suitable_for_humor


@pytest.mark.parametrize(
    "source",
    (
        "The lost package was recovered.",
        "The fatal build error was fixed.",
        "The database entered a recovery coma.",
    ),
)
def test_v176_harm_terms_without_human_relation_are_not_literal_harm(source: str) -> None:
    scope = inspect_human_harm_scope(source)

    assert scope.harm_referent != "human"
    assert not scope.literal_harm


@pytest.mark.parametrize(
    "source",
    (
        "The build died after the dependency update.",
        "The pipeline died overnight.",
        "The database died during migration.",
        "The node died at block 100.",
        "The model died after the rollout.",
        "The bill was killed in committee.",
        "The launch was dead on arrival.",
    ),
)
def test_v176_clear_non_human_death_relations_are_not_tragedy(source: str) -> None:
    scope = inspect_human_harm_scope(source)
    safety = inspect_humor_safety(_context(source).evidence, "achievement unlocked")

    assert scope.harm_referent == "non_human"
    assert scope.harm_state == "non_human_failure"
    assert scope.literal_or_metaphorical_usage == "metaphorical_or_technical"
    assert not scope.literal_harm
    assert scope.metaphorical_or_technical_usage
    assert scope.evidence_spans
    assert not safety.sensitive_context
    assert safety.suitable_for_humor


def test_v176_mixed_non_human_and_human_death_remains_sensitive() -> None:
    source = "The server died. Two patients died during the outage."
    scope = inspect_human_harm_scope(source)
    safety = inspect_humor_safety(_context(source).evidence, "achievement unlocked")

    assert scope.harm_referent == "human"
    assert scope.harm_state == "death"
    assert scope.literal_harm
    assert scope.metaphorical_or_technical_usage
    assert safety.sensitive_context
    assert not safety.suitable_for_humor


@pytest.mark.parametrize(
    ("text", "expected_operator", "expected_marker"),
    (
        ("onboarding is glorified paperwork", "decorated_identity", "glorified"),
        ("onboarding is fancy paperwork", "decorated_identity", "fancy"),
        ("onboarding is rebranded paperwork", "decorated_identity", "rebranded"),
        ("onboarding is paperwork in disguise", "decorated_identity", "in disguise"),
        (
            "onboarding is paperwork with a fresh coat of paint",
            "decorated_identity",
            "fresh coat of paint",
        ),
        (
            "onboarding: paperwork with better branding",
            "decorated_identity",
            "better branding",
        ),
        (
            "onboarding is nothing but paperwork with better UX",
            "reduction",
            None,
        ),
    ),
)
def test_v176_relation_shell_variants_match_recent_reduction(
    text: str,
    expected_operator: str,
    expected_marker: str | None,
) -> None:
    shell = inspect_reply_shell(text)
    inspection = inspect_reply_naturalness(
        _variant(text),
        recent_posts=("fundraising is just sales with extra steps",),
    )

    assert shell.operator == expected_operator
    assert shell.subject_span
    assert shell.complement_span
    assert shell.repackaging_marker == expected_marker
    assert inspection.recent_syntax_reuse
    assert "reductive_identity" in inspection.structural_families


@pytest.mark.parametrize(
    "text",
    (
        "velocity is distance divided by time",
        "the bill is a proposed law awaiting a vote",
        "onboarding now requires paperwork and identity verification",
    ),
)
def test_v176_literal_definitions_and_clarifications_are_not_shell_reuse(text: str) -> None:
    inspection = inspect_reply_naturalness(
        _variant(text),
        recent_posts=("fundraising is just sales with extra steps",),
    )

    assert not inspection.recent_syntax_reuse


def test_v176_source_specific_nouns_do_not_hide_structural_reuse() -> None:
    context = _context("Onboarding now includes paperwork and a new interface.")
    text = "onboarding is glorified paperwork"
    inspection = inspect_reply_naturalness(
        _variant(text),
        evidence_packet=context.evidence,
        recent_posts=("fundraising is just sales with extra steps",),
    )

    assert inspection.reply_shell is not None
    assert inspection.reply_shell.source_specific
    assert inspection.recent_syntax_reuse


def test_v176_fresh_structure_with_same_nouns_is_not_falsely_blocked() -> None:
    text = "the new onboarding paperwork adds two minutes to completion"
    inspection = inspect_reply_naturalness(
        _variant(text),
        recent_posts=("fundraising is just sales with extra steps",),
    )

    assert not inspection.recent_syntax_reuse


@pytest.mark.parametrize(
    ("text", "expected_relation"),
    (
        ("applause for the outage", "celebration"),
        ("a standing ovation for the grid", "celebration"),
        ("four stars for the outage", "rating"),
        ("so proud of the grid", "admiration"),
        ("a victory for the outage", "success_attribution"),
        ("the outage nailed it", "success_attribution"),
        ("the outage aced it", "success_attribution"),
    ),
)
def test_v176_acceptance_audit_evaluation_variants_share_relations(
    text: str,
    expected_relation: str,
) -> None:
    context = _context("A hospital reported 10 deaths after the outage.")
    intent = inspect_humor_intent(context.evidence, text, metadata_says_humor=False)

    assert intent.text_appears_humorous
    assert intent.humor_safety_required
    assert intent.evaluative_irony is not None
    assert intent.evaluative_irony.evaluation_relation == expected_relation
    assert intent.evaluative_irony.target_source_alignment
    assert intent.evaluative_irony.implicit_sarcasm


@pytest.mark.parametrize(
    ("source", "expected_state", "expected_event"),
    (
        ("Two patients remain on life support.", "critical_condition", "medical"),
        ("Three workers remained in a coma.", "critical_condition", "medical"),
        ("The crash was fatal to ten passengers.", "death", "physical_crash"),
    ),
)
def test_v176_acceptance_audit_result_state_harm_variants(
    source: str,
    expected_state: str,
    expected_event: str,
) -> None:
    scope = inspect_human_harm_scope(source)

    assert scope.harm_referent == "human"
    assert scope.harm_state == expected_state
    assert scope.event_type == expected_event
    assert scope.literal_harm


@pytest.mark.parametrize(
    "text",
    (
        "onboarding: glorified paperwork",
        "onboarding is paperwork, but fancier",
        "onboarding is paperwork wearing better branding",
    ),
)
def test_v176_acceptance_audit_repackaging_variants_share_family(text: str) -> None:
    shell = inspect_reply_shell(text)
    inspection = inspect_reply_naturalness(
        _variant(text),
        recent_posts=("fundraising is just sales with extra steps",),
    )

    assert shell.operator == "decorated_identity"
    assert shell.repackaging_marker
    assert inspection.recent_syntax_reuse
