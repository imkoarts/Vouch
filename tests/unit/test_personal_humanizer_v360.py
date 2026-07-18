from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine

from app.config import Settings
from app.domain.enums import ContentType, GenerationMode
from app.providers.responses_provider import (
    BASE_GENERATION_INSTRUCTIONS,
    FORMAT_CONTRACTS,
    INITIAL_HUMANIZER_RUNTIME,
    MODE_CONTRACTS,
    PERSONAL_HUMANIZER_RUNTIME,
    REPLY_MICRO_RUNTIME,
)
from app.schemas.content import (
    AngleCandidate,
    EvidenceItem,
    EvidencePacket,
    GenerationRequest,
    GenerationResult,
    GenerationVariant,
    HumanizerRevisionTarget,
    MediaPlan,
    QualityIssue,
    QualityReport,
)
from app.services.doctor import DoctorService
from app.services.draft_service import DraftService
from app.services.editorial_quality import assess_variant, inspect_reply_naturalness
from app.services.generation_pipeline import build_generation_context
from app.services.humanizer_runtime import HumanizerRuntimeLoader
from app.services.semantic_classifier import (
    DeterministicSemanticInspector,
    derive_source_semantic_map,
    inspect_reaction_function,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _packet(source: str) -> EvidencePacket:
    return EvidencePacket(
        mode=GenerationMode.REPLY,
        topic="reply diversity",
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


def _variant(
    text: str,
    *,
    angle_id: str = "reply-plain-observation",
    angle_type: str = "plain_update",
    contribution_type: str | None = "plain_observation",
) -> GenerationVariant:
    return GenerationVariant(
        label=angle_id,
        text=text,
        similarity_risk=0.0,
        angle_id=angle_id,
        angle_type=angle_type,
        evidence_ids=("source",),
        confidence="high",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
        contribution_id=angle_id,
        contribution_type=contribution_type,
    )


def _report(*issues: str) -> QualityReport:
    return QualityReport(
        passed=True,
        specificity_score=1.0,
        evidence_score=1.0,
        naturalness_score=1.0,
        distinctiveness_score=1.0,
        issues=tuple(
            QualityIssue(
                code=code,
                severity="warning",
                message=code,
                suggested_action="flatten",
            )
            for code in issues
        ),
    )


def _angle(angle_id: str, contribution_type: str, angle_type: str) -> AngleCandidate:
    return AngleCandidate(
        angle_id=angle_id,
        angle_type=angle_type,
        thesis="Use the grounded reply family for this source.",
        evidence_ids=("source",),
        why_interesting="Grounded reply candidate.",
        confidence="high",
        contribution_type=contribution_type,
    )


def test_bundled_humanizer_is_version_360_and_contracts_are_coherent() -> None:
    runtime = HumanizerRuntimeLoader().load()
    reply_range = f"4{chr(8211)}35 words"

    assert runtime.source == "bundled"
    assert runtime.version == "3.6.0"
    assert reply_range in runtime.initial_contract
    assert reply_range in runtime.revision_contract
    assert "least constructed" in runtime.initial_contract.casefold()
    assert "least constructed" in runtime.revision_contract.casefold()
    assert "micro-reply-examples.md" in runtime.reference_files


def test_all_canonical_reply_prompts_use_the_360_policy() -> None:
    canonical = "\n".join(
        (
            BASE_GENERATION_INSTRUCTIONS,
            INITIAL_HUMANIZER_RUNTIME,
            PERSONAL_HUMANIZER_RUNTIME,
            REPLY_MICRO_RUNTIME,
            MODE_CONTRACTS["reply"],
            FORMAT_CONTRACTS["reply"],
        )
    )
    normalized = canonical.casefold().replace(chr(8211), "-")

    assert "3.6.0" in canonical
    assert "4-35 words" in normalized
    assert "6-18 words" not in normalized
    assert "one lowercase sentence" not in normalized
    assert "strongest survivor" not in normalized
    assert "stop after the punchline" not in normalized
    assert "lowercase is optional" in normalized
    assert "two when" in normalized


def test_reply_planner_exposes_four_families_before_semantic_filtering() -> None:
    from app.schemas.content import UntrustedSourceData

    context = build_generation_context(
        idea_summary="Reply to source",
        idea_explanation={"origin": "reply_to_source"},
        language="en",
        sources=(
            UntrustedSourceData(
                source_id="source",
                source_type="x_post",
                content="The social manager has ten other jobs and no budget.",
            ),
        ),
        content_type=ContentType.REPLY,
    )

    assert [item.contribution_type for item in context.angles] == [
        "plain_observation",
        "direct_response",
        "dry_humor",
        "contextual_extension",
    ]


@pytest.mark.asyncio
async def test_reply_semantic_filter_retains_zero_to_three_grounded_candidates() -> None:
    from app.schemas.content import UntrustedSourceData

    context = build_generation_context(
        idea_summary="Reply to source",
        idea_explanation={"origin": "reply_to_source"},
        language="en",
        sources=(
            UntrustedSourceData(
                source_id="source",
                source_type="x_post",
                content="The social manager has ten other jobs and no budget.",
            ),
        ),
        content_type=ContentType.REPLY,
    )
    service = DraftService.__new__(DraftService)
    service.deterministic_semantic_inspector = DeterministicSemanticInspector()
    service.semantic_inspector = service.deterministic_semantic_inspector

    filtered, _, inspections = await service._validate_context_semantics(context)

    assert 0 <= len(filtered.angles) <= 3
    assert {item.contribution_type for item in filtered.angles} <= {
        "plain_observation",
        "direct_response",
        "specific_qualification",
        "genuine_question",
        "dry_humor",
        "contextual_extension",
    }
    assert any(not item.grounded for item in inspections)


def test_grounded_twenty_to_thirty_five_word_reply_is_not_overproduction() -> None:
    packet = _packet("The social manager handles ten other jobs and has no budget.")
    text = (
        "half the time the social manager is good, they just gave them 10 other jobs and no budget"
    )
    report = assess_variant(
        _variant(text),
        evidence_packet=packet,
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )
    codes = {issue.code for issue in report.issues}

    assert "REPLY_OVERPRODUCTION" not in codes
    assert "FALSE_BREVITY" not in codes
    assert report.reply_fit_score >= 0.8


def test_necessary_two_sentence_reply_is_allowed() -> None:
    packet = _packet("The model can migrate, but correction history cannot transfer.")
    text = (
        "the model can move. the correction history still cannot, which is the part "
        "the team trained"
    )
    report = assess_variant(
        _variant(text, angle_type="mechanism", contribution_type="contextual_extension"),
        evidence_packet=packet,
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert "REPLY_OVERPRODUCTION" not in {issue.code for issue in report.issues}
    assert report.reply_fit_score >= 0.75


def test_lowercase_is_optional_for_a_natural_reply() -> None:
    packet = _packet("The export includes scores but does not include test cases.")
    report = assess_variant(
        _variant("The scores help, but the missing test cases still block a rerun."),
        evidence_packet=packet,
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert "FALSE_CASUALNESS" not in {issue.code for issue in report.issues}
    assert "REPLY_OVERPRODUCTION" not in {issue.code for issue in report.issues}


def test_plain_observation_scores_above_polished_quote_card() -> None:
    packet = _packet("A social manager has ten other jobs and no budget.")
    request = GenerationRequest(
        idea_summary="Reply",
        content_type=ContentType.REPLY,
        language="en",
        generation_mode=GenerationMode.REPLY,
        evidence_packet=packet,
        angle_candidates=(
            _angle("plain", "plain_observation", "plain_update"),
            _angle("humor", "dry_humor", "concise_joke"),
        ),
    )
    plain = _variant(
        (
            "half the time the social manager is good, they just gave them ten other "
            "jobs and no budget"
        ),
        angle_id="plain",
    )
    polished = _variant(
        (
            "brands will spend millions building a product then hand the account to "
            "whoever is least busy"
        ),
        angle_id="humor",
        angle_type="concise_joke",
        contribution_type="dry_humor",
    )
    polished_report = assess_variant(
        polished,
        evidence_packet=packet,
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert {"QUOTE_CARD_REPLY", "ONE_LINER_FACTORY"} & {
        issue.code for issue in polished_report.issues
    }
    assert DraftService._variant_selection_score(plain, _report(), request) > (
        DraftService._variant_selection_score(polished, polished_report, request)
    )


def test_real_question_is_a_low_constructed_reply_type() -> None:
    packet = _packet("The export includes scores but does not say whether test cases are included.")
    variant = _variant(
        "do the test cases come with the export, or just the scores?",
        angle_type="specific_question",
        contribution_type="genuine_question",
    )
    naturalness = inspect_reply_naturalness(variant, evidence_packet=packet)

    assert naturalness.response_type == "question"
    assert naturalness.constructedness == "low"
    assert not naturalness.punchline_compulsion


def test_recent_syntax_reuse_is_structural_when_nouns_change() -> None:
    packet = _packet("The product adds steps to onboarding.")
    report = assess_variant(
        _variant("onboarding is just paperwork with extra steps"),
        evidence_packet=packet,
        recent_posts=("fundraising is just sales with extra steps",),
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert "RECENT_SYNTAX_REUSE" in {issue.code for issue in report.issues}


@pytest.mark.parametrize(
    ("draft", "recent"),
    (
        (
            "sources confirm the budget has left the group chat",
            "sources confirm the deadline has left the group chat",
        ),
        (
            "the budget detail has entered the group chat",
            "the pricing detail has entered the group chat",
        ),
        (
            "bold strategy, apparently support can run on fumes",
            "sure, why not, apparently marketing can run on fumes",
        ),
    ),
)
def test_recent_reply_family_reuse_detects_newsroom_metaphor_and_sarcasm(
    draft: str,
    recent: str,
) -> None:
    packet = _packet("The team gave the launch no budget and no support coverage.")
    report = assess_variant(
        _variant(draft),
        evidence_packet=packet,
        recent_posts=(recent,),
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert "RECENT_SYNTAX_REUSE" in {issue.code for issue in report.issues}


def test_quote_card_reply_and_punchline_compulsion_emit_real_issues() -> None:
    packet = _packet("Hungary removed President Sulyok as part of dismantling Orbán's influence.")
    report = assess_variant(
        _variant(
            "presidential removal, achievement unlocked",
            angle_type="concise_joke",
            contribution_type="dry_humor",
        ),
        evidence_packet=packet,
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )
    codes = {issue.code for issue in report.issues}

    assert "PUNCHLINE_COMPULSION" in codes
    assert "ONE_LINER_FACTORY" in codes


def test_false_brevity_emits_blocking_issue() -> None:
    packet = _packet("The plan works only when a human reviews the result.")
    report = assess_variant(
        _variant("works. mostly:"),
        evidence_packet=packet,
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    issue = next(item for item in report.issues if item.code == "FALSE_BREVITY")
    assert issue.severity == "error"
    assert not report.passed


@pytest.mark.parametrize(
    "text",
    (
        "removing the president, but call it a rebrand",
        "anti-Orbán campaign speedrun",
        "presidential removal, achievement unlocked",
        "nothing says dismantling influence like removing the president",
    ),
)
def test_formulaic_wordplay_does_not_bypass_reaction_grounding(text: str) -> None:
    packet = _packet(
        "Hungary's parliament removed President Sulyok as part of dismantling Orbán's influence."
    )

    inspection = inspect_reaction_function(packet, text)

    assert not inspection.grounded
    assert inspection.source_echo or inspection.generic_template_only


def test_genuine_short_humor_remains_allowed() -> None:
    packet = _packet("The model added a low verbosity setting.")
    text = "finally an official anti-yap switch"
    inspection = inspect_reaction_function(packet, text)
    report = assess_variant(
        _variant(text, angle_type="concise_joke", contribution_type="dry_humor"),
        evidence_packet=packet,
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert inspection.grounded
    assert "PUNCHLINE_COMPULSION" not in {issue.code for issue in report.issues}
    assert inspect_reply_naturalness(
        _variant(text, angle_type="concise_joke", contribution_type="dry_humor"),
        evidence_packet=packet,
    ).source_specific


def test_doctor_reports_bundled_source_and_version_360(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        config_dir=PROJECT_ROOT / "config",
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        mock_mode=True,
        publish_enabled=False,
    )
    engine = create_engine("sqlite+pysqlite:///:memory:")
    checks = {item.name: item for item in DoctorService(settings, engine).run()}

    assert checks["humanizer"].status == "PASS"
    assert "source=bundled" in checks["humanizer"].message
    assert "version=3.6.0" in checks["humanizer"].message


def test_invalid_external_runtime_falls_back_atomically_to_bundled_360(tmp_path: Path) -> None:
    runtime = HumanizerRuntimeLoader(configured_path=tmp_path / "missing" / "SKILL.md").load()
    reply_range = f"4{chr(8211)}35 words"

    assert runtime.source == "bundled"
    assert runtime.version == "3.6.0"
    assert reply_range in runtime.initial_contract
    assert reply_range in runtime.revision_contract


def test_source_map_is_unchanged_by_reply_runtime_upgrade() -> None:
    packet = _packet("Alice signed the bill on Tuesday.")
    source_map = derive_source_semantic_map(packet)

    assert any(item.text == "Alice signed the bill on Tuesday." for item in source_map.propositions)


@pytest.mark.asyncio
async def test_humanizer_revision_cannot_turn_plain_reply_into_punchline() -> None:
    packet = _packet("The social manager handles ten other jobs and has no budget.")
    candidate = _angle("plain", "plain_observation", "plain_update")
    original = _variant(
        "the social manager has ten other jobs and no budget",
        angle_id="plain",
    )
    revised = _variant(
        "staffing is just chaos with extra steps",
        angle_id="plain",
    )
    request = GenerationRequest(
        idea_summary="reply",
        content_type=ContentType.REPLY,
        language="en",
        generation_mode=GenerationMode.REPLY,
        editorial_intent="reply_reaction",
        evidence_packet=packet,
        angle_candidates=(candidate,),
        revision_targets=(
            HumanizerRevisionTarget(
                variant=original,
                issue_codes=("RECENT_SYNTAX_REUSE",),
            ),
        ),
        revision_issue_codes=("RECENT_SYNTAX_REUSE",),
        metadata={"request_id": "revision-shape"},
    )
    result = GenerationResult(
        idea_summary="reply",
        recommended_format=ContentType.REPLY,
        variants=(revised,),
        recommended_variant=0,
        recommendation_reason="revision",
        media_plan=MediaPlan(),
    )
    service = DraftService.__new__(DraftService)
    service.deterministic_semantic_inspector = DeterministicSemanticInspector()
    service.semantic_inspector = service.deterministic_semantic_inspector
    service._semantic_inspection_cache = {}

    inspections = await service._inspect_result_semantics(result, request)

    assert inspections[0].no_grounded_contribution
    assert "CONTRIBUTION_CHANGED_DURING_REVISION" in inspections[0].issue_codes
    assert "PUNCHLINE_COMPULSION" in inspections[0].issue_codes
