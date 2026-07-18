from __future__ import annotations

import pytest

from app.domain.enums import ContentType
from app.schemas.content import (
    DraftSemanticInspection,
    GenerationVariant,
    SemanticHumorProposal,
    UntrustedSourceData,
)
from app.services.editorial_quality import (
    assess_variant,
    inspect_reply_naturalness,
    inspect_reply_shell,
)
from app.services.generation_pipeline import build_generation_context
from app.services.semantic_classifier import (
    derive_source_semantic_map,
    inspect_draft_deterministic,
    inspect_human_harm_scope,
    inspect_humor_intent,
    inspect_humor_safety,
    merge_semantic_inspections,
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


@pytest.mark.parametrize(
    "text",
    (
        "the outage really delivered",
        "love that for the hospital",
        "excellent work from the outage",
        "mission accomplished",
    ),
)
def test_implicit_positive_evaluation_is_humor_on_fatal_source(text: str) -> None:
    context = _context("A hospital reported 10 deaths after the outage.")
    intent = inspect_humor_intent(context.evidence, text, metadata_says_humor=False)
    report = assess_variant(
        _variant(text),
        evidence_packet=context.evidence,
        editorial_intent="reply_reaction",
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert intent.text_appears_humorous
    assert intent.detected_reaction_type == "sarcasm"
    assert intent.humor_safety_required
    assert intent.evaluative_irony is not None
    assert intent.evaluative_irony.implicit_sarcasm
    assert not report.passed
    assert "HUMOR_ON_TRAGEDY" in {issue.code for issue in report.issues}


def test_medium_confidence_evaluative_irony_on_sensitive_source_fails_closed() -> None:
    context = _context("A hospital reported 10 deaths after the outage.")
    text = "great job, weather"
    intent = inspect_humor_intent(
        context.evidence,
        text,
        metadata_says_humor=False,
    )
    report = assess_variant(
        _variant(text),
        evidence_packet=context.evidence,
        editorial_intent="reply_reaction",
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert intent.detected_reaction_type == "uncertain"
    assert "HUMOR_INTENT_UNCERTAIN" in intent.issue_codes
    assert not report.passed
    assert "HUMOR_INTENT_UNCERTAIN" in {issue.code for issue in report.issues}


def test_literal_positive_evaluation_on_positive_source_is_not_sarcasm() -> None:
    context = _context("The outage was fixed and the hospital restored every service safely.")
    intent = inspect_humor_intent(
        context.evidence,
        "excellent work from the response team",
        metadata_says_humor=False,
    )

    assert not intent.text_appears_humorous
    assert intent.evaluative_irony is not None
    assert intent.evaluative_irony.literal_positive_reading_plausible
    assert not intent.evaluative_irony.implicit_sarcasm


@pytest.mark.parametrize(
    "source",
    (
        "The outage claimed ten lives.",
        "The fire took three lives.",
        "Ten lives were lost after the outage.",
        "Three people succumbed to their injuries.",
        "Two workers did not survive.",
        "The incident left a worker fighting for his life.",
        "A patient was pronounced dead.",
        "The death toll reached four.",
        "Two people perished in the fire.",
    ),
)
def test_additional_fatality_language_is_literal_human_harm(source: str) -> None:
    scope = inspect_human_harm_scope(source)
    safety = inspect_humor_safety(_context(source).evidence, "achievement unlocked")

    assert scope.harm_referent == "human"
    assert scope.literal_harm
    assert safety.sensitive_context
    assert not safety.suitable_for_humor


@pytest.mark.parametrize(
    "source",
    (
        "Dead code was removed.",
        "The dead link was fixed.",
        "The browser process died.",
        "The API killed the worker process.",
        "The campaign killed engagement.",
        "The app killed battery life.",
        "The deal is dead.",
        "The feature is dead.",
        "The server died after the deploy.",
        "The app killed the worker experience.",
        "The campaign killed employee engagement.",
        "The deploy killed the user session.",
    ),
)
def test_non_human_death_language_is_not_tragedy(source: str) -> None:
    safety = inspect_humor_safety(_context(source).evidence, "achievement unlocked")

    assert not safety.sensitive_context
    assert safety.harm_referent == "non_human"
    assert safety.metaphorical_or_technical_usage
    assert safety.suitable_for_humor


def test_mixed_technical_and_real_human_harm_remains_sensitive() -> None:
    source = "The API killed the worker process. Two patients died during the outage."
    safety = inspect_humor_safety(_context(source).evidence, "achievement unlocked")

    assert safety.sensitive_context
    assert safety.harm_referent == "human"
    assert safety.literal_harm
    assert safety.metaphorical_or_technical_usage
    assert not safety.suitable_for_humor


@pytest.mark.parametrize(
    "source",
    (
        "A fire sale starts Friday.",
        "The fire callback runs after the request.",
        "The product saw a growth explosion.",
        "There was an explosion of interest in the launch.",
        "The market collapse erased gains.",
        "A DDoS attack hit the API.",
        "The prompt attack bypassed the first filter.",
    ),
)
def test_bounded_non_harmful_ambiguous_events_are_safe(source: str) -> None:
    safety = inspect_humor_safety(_context(source).evidence, "achievement unlocked")

    assert not safety.sensitive_context
    assert safety.suitable_for_humor
    assert safety.context_confidence == "high"


def test_safe_event_form_does_not_suppress_human_harm_elsewhere() -> None:
    source = "A cyber attack caused two patient deaths."
    safety = inspect_humor_safety(_context(source).evidence, "achievement unlocked")

    assert safety.sensitive_context
    assert safety.literal_harm
    assert not safety.suitable_for_humor


def test_unknown_death_referent_fails_closed_only_for_humor() -> None:
    context = _context("The component killed it overnight.")
    humorous = inspect_reply_naturalness(
        _variant("achievement unlocked"),
        evidence_packet=context.evidence,
    )
    plain = inspect_reply_naturalness(
        _variant("the overnight failure needs a root-cause review"),
        evidence_packet=context.evidence,
    )

    assert humorous.humor_safety is not None
    assert humorous.humor_safety.harm_referent == "unknown"
    assert not humorous.humor_safety.suitable_for_humor
    assert plain.humor_safety is None


@pytest.mark.parametrize(
    ("text", "operator", "family"),
    (
        (
            "onboarding is paperwork dressed up as product strategy",
            "decorated_identity",
            "reductive_identity",
        ),
        ("onboarding is paperwork under a new name", "decorated_identity", "reductive_identity"),
        (
            "call onboarding what it is: paperwork with a progress bar",
            "reduction",
            "reductive_identity",
        ),
        ("as it turns out, onboarding is paperwork", "reveal", "reveal_reduction"),
    ),
)
def test_relational_reply_shells_are_normalized(
    text: str,
    operator: str,
    family: str,
) -> None:
    shell = inspect_reply_shell(text)
    inspection = inspect_reply_naturalness(
        _variant(text),
        recent_posts=(
            "turns out onboarding was just paperwork"
            if operator == "reveal"
            else "fundraising is just sales with extra steps",
        ),
    )

    assert shell.operator == operator
    assert shell.subject_span
    assert shell.complement_span
    assert family in inspection.structural_families
    assert inspection.recent_syntax_reuse


def test_positive_recovery_language_does_not_hide_real_human_harm() -> None:
    context = _context("The hospital restored service, but two patients died during the outage.")
    intent = inspect_humor_intent(
        context.evidence,
        "excellent work from the outage",
        metadata_says_humor=False,
    )

    assert intent.text_appears_humorous
    assert intent.detected_reaction_type == "sarcasm"
    assert intent.evaluative_irony is not None
    assert not intent.evaluative_irony.literal_positive_reading_plausible


@pytest.mark.parametrize(
    "text",
    (
        "OAuth, in other words, is an authorization protocol",
        "A triangle, in other words, is a three-sided polygon",
    ),
)
def test_literal_definition_with_in_other_words_is_not_reductive_shell(text: str) -> None:
    shell = inspect_reply_shell(text)

    assert shell.operator == "none"


def test_reductive_in_other_words_still_maps_to_reduction() -> None:
    shell = inspect_reply_shell("onboarding, in other words, is paperwork with a progress bar")

    assert shell.operator == "reduction"
    assert shell.subject_span == "onboarding"


def _provider_inspection(proposal: SemanticHumorProposal) -> DraftSemanticInspection:
    return DraftSemanticInspection(
        contribution_delivered=True,
        contribution_preserved=True,
        source_echo=False,
        implied_restatement=False,
        internal_repetition=False,
        analytical_packaging=False,
        source_stronger_than_draft=False,
        no_grounded_contribution=False,
        humor_intent_proposal=proposal,
    )


def test_provider_none_cannot_suppress_deterministic_sarcasm() -> None:
    context = _context("A hospital reported 10 deaths after the outage.")
    contribution = context.angles[0]
    text = "the outage really delivered"
    deterministic = inspect_draft_deterministic(
        context.evidence,
        contribution,
        text,
        derive_source_semantic_map(context.evidence),
        commentary_mode=True,
    )
    provider = _provider_inspection(
        SemanticHumorProposal(
            intent="none",
            humorous_treatment=False,
            confidence="high",
        )
    )

    merged = merge_semantic_inspections(deterministic, provider, text=text)

    assert merged.humor_intent_proposal is not None
    assert merged.humor_intent_proposal.humorous_treatment
    assert "HUMOR_INTENT_CONFLICT" in merged.issue_codes
    assert "HUMOR_ON_TRAGEDY" in merged.issue_codes


def test_provider_additional_humor_coverage_runs_application_safety() -> None:
    context = _context("A hospital reported 10 deaths after the outage.")
    contribution = context.angles[0]
    text = "bravo, outage"
    deterministic = inspect_draft_deterministic(
        context.evidence,
        contribution,
        text,
        derive_source_semantic_map(context.evidence),
        commentary_mode=True,
    )
    provider = _provider_inspection(
        SemanticHumorProposal(
            intent="sarcasm",
            humorous_treatment=True,
            confidence="high",
            evidence_spans=("bravo, outage",),
        )
    )

    merged = merge_semantic_inspections(deterministic, provider, text=text)

    assert merged.humor_intent_proposal is not None
    assert merged.humor_intent_proposal.intent == "sarcasm"
    assert "HUMOR_ON_TRAGEDY" in merged.issue_codes


def test_provider_humor_span_must_be_exact() -> None:
    context = _context("A hospital reported 10 deaths after the outage.")
    contribution = context.angles[0]
    text = "bravo, outage"
    deterministic = inspect_draft_deterministic(
        context.evidence,
        contribution,
        text,
        derive_source_semantic_map(context.evidence),
        commentary_mode=True,
    )
    provider = _provider_inspection(
        SemanticHumorProposal(
            intent="sarcasm",
            humorous_treatment=True,
            confidence="high",
            evidence_spans=("not in final text",),
        )
    )

    merged = merge_semantic_inspections(deterministic, provider, text=text)

    assert "HUMOR_INTENT_CONFLICT" in merged.issue_codes
    assert "HUMOR_ON_TRAGEDY" not in merged.issue_codes
