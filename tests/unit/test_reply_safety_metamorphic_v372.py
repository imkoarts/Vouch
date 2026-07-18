from __future__ import annotations

import pytest

from app.domain.enums import ContentType
from app.schemas.content import GenerationVariant, UntrustedSourceData
from app.services.draft_service import RetryDisposition, retry_disposition
from app.services.editorial_quality import assess_variant, inspect_reply_naturalness
from app.services.generation_pipeline import build_generation_context
from app.services.semantic_classifier import inspect_humor_safety, inspect_reaction_function


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
    contribution_type: str = "dry_humor",
    angle_type: str = "concise_joke",
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
    ("source", "reply", "category"),
    (
        (
            "A hospital reported 10 deaths after the outage.",
            "the outage is doing heroic amounts of work",
            "death_or_serious_injury",
        ),
        (
            "Five people died in the crash.",
            "traffic really committed to the bit",
            "death_or_serious_injury",
        ),
        (
            "The earthquake left hundreds injured.",
            "infrastructure speedrun",
            "personal_tragedy",
        ),
        (
            "A missing child was found dead.",
            "plot twist nobody needed",
            "personal_tragedy",
        ),
    ),
)
def test_tragedy_humor_is_never_eligible(
    source: str,
    reply: str,
    category: str,
) -> None:
    context = _context(source)
    variant = _variant(reply)

    safety = inspect_humor_safety(context.evidence, reply)
    report = assess_variant(
        variant,
        evidence_packet=context.evidence,
        editorial_intent="reply_reaction",
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert safety.sensitive_context
    assert category in safety.sensitive_categories
    assert not safety.suitable_for_humor
    assert not report.passed
    assert "HUMOR_ON_TRAGEDY" in {issue.code for issue in report.issues}


def test_source_specificity_and_grounding_do_not_override_humor_safety() -> None:
    context = _context("A hospital reported 10 deaths after the outage.")
    reply = "the outage is doing heroic amounts of work"

    reaction = inspect_reaction_function(context.evidence, reply)
    naturalness = inspect_reply_naturalness(
        _variant(reply),
        evidence_packet=context.evidence,
    )

    assert reaction.grounded
    assert reaction.source_specific
    assert naturalness.source_specific
    assert naturalness.humor_safety is not None
    assert not naturalness.humor_safety.suitable_for_humor


def test_humor_safety_block_is_semantic_abstention_not_style_retry() -> None:
    assert retry_disposition(("HUMOR_ON_TRAGEDY",)) is RetryDisposition.NO_POST


def test_serious_injuries_are_sensitive_but_price_war_is_not() -> None:
    serious = _context("Five people suffered serious injuries after the outage.")
    market = _context("The company started a price war with a cheaper plan.")

    assert not inspect_humor_safety(
        serious.evidence, "the outage really committed to the bit"
    ).suitable_for_humor
    assert inspect_humor_safety(
        market.evidence, "the pricing team finally found the volume knob"
    ).suitable_for_humor


def test_sensitive_source_can_target_institutional_wording_without_targeting_harm() -> None:
    context = _context(
        "After an outage caused 10 deaths, the agency statement called it a minor interruption."
    )
    safety = inspect_humor_safety(
        context.evidence,
        "the agency statement's wording is doing a lot of work",
    )

    assert safety.sensitive_context
    assert safety.targets_public_claim_or_system
    assert not safety.targets_victim_or_harm
    assert safety.suitable_for_humor


def test_harmless_self_deprecating_humor_remains_suitable() -> None:
    context = _context("The app added a low-verbosity setting.")
    safety = inspect_humor_safety(
        context.evidence,
        "my drafts just lost their favorite hiding place",
    )

    assert safety.self_deprecating
    assert safety.suitable_for_humor


def test_non_harmful_source_is_not_flagged_as_tragedy() -> None:
    context = _context("The company added six approval screens.")
    reply = "nothing says simplicity like six more screens"

    safety = inspect_humor_safety(context.evidence, reply)

    assert not safety.sensitive_context
    assert safety.suitable_for_humor
    assert "HUMOR_ON_TRAGEDY" not in safety.issue_codes


def test_hospital_administration_topic_without_harm_is_not_overblocked() -> None:
    context = _context("The hospital added six approval screens to its scheduling system.")

    safety = inspect_humor_safety(
        context.evidence,
        "nothing says simple scheduling like six approval screens",
    )

    assert not safety.sensitive_context
    assert safety.suitable_for_humor


@pytest.mark.parametrize(
    "candidate",
    (
        "onboarding boils down to paperwork in a nicer font",
        "onboarding comes down to paperwork with a progress bar",
        "onboarding is little more than paperwork wearing product copy",
    ),
)
def test_reductive_identity_metamorphs_match_recent_shell(candidate: str) -> None:
    inspection = inspect_reply_naturalness(
        _variant(candidate, contribution_type="plain_observation", angle_type="plain_update"),
        recent_posts=("fundraising is just sales with extra steps",),
    )

    assert "reductive_identity" in inspection.structural_families
    assert inspection.recent_structural_match
    assert inspection.recent_syntax_reuse
    assert inspection.generic_reusability


def test_inverse_praise_paraphrase_matches_recent_shell() -> None:
    inspection = inspect_reply_naturalness(
        _variant(
            "few things communicate simplicity better than six more screens",
            contribution_type="plain_observation",
            angle_type="plain_update",
        ),
        recent_posts=("nothing says simplicity like six screens",),
    )

    assert "inverse_praise" in inspection.structural_families
    assert inspection.recent_syntax_reuse
    assert inspection.quote_card_risk


def test_reveal_paraphrase_matches_recent_shell() -> None:
    inspection = inspect_reply_naturalness(
        _variant(
            "apparently onboarding is paperwork in product copy",
            contribution_type="plain_observation",
            angle_type="plain_update",
        ),
        recent_posts=("turns out onboarding was just paperwork",),
    )

    assert "reveal_reduction" in inspection.structural_families
    assert inspection.recent_syntax_reuse


def test_changed_nouns_tense_and_punctuation_do_not_hide_reuse() -> None:
    inspection = inspect_reply_naturalness(
        _variant(
            "At this point, compliance comes down to paperwork with a dashboard!",
            contribution_type="plain_observation",
            angle_type="plain_update",
        ),
        recent_posts=("fundraising was just sales with extra steps.",),
    )

    assert inspection.recent_syntax_reuse


def test_source_specificity_is_separate_from_structural_reuse() -> None:
    context = _context("Onboarding now includes six more screens.")
    candidate = "onboarding boils down to six screens in a nicer font"
    inspection = inspect_reply_naturalness(
        _variant(candidate, contribution_type="plain_observation", angle_type="plain_update"),
        evidence_packet=context.evidence,
        recent_posts=("fundraising is just sales with extra steps",),
    )

    assert inspection.source_specific
    assert inspection.recent_structural_match
    assert inspection.quote_card_risk


def test_same_source_nouns_with_fresh_structure_are_not_reuse() -> None:
    context = _context("Onboarding now includes six more screens.")
    candidate = "six more screens make onboarding slower for the people using it"
    inspection = inspect_reply_naturalness(
        _variant(candidate, contribution_type="plain_observation", angle_type="plain_update"),
        evidence_packet=context.evidence,
        recent_posts=("onboarding is just paperwork with extra steps",),
    )

    assert inspection.source_specific
    assert not inspection.recent_syntax_reuse
    assert not inspection.generic_reusability


def test_single_isolated_shell_is_not_automatically_recent_reuse() -> None:
    inspection = inspect_reply_naturalness(
        _variant(
            "onboarding boils down to paperwork in a nicer font",
            contribution_type="plain_observation",
            angle_type="plain_update",
        ),
        recent_posts=(),
    )

    assert not inspection.recent_syntax_reuse
    assert inspection.generic_reusability
