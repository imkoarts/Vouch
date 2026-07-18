from __future__ import annotations

from itertools import count

from app.domain.enums import ContentType
from app.schemas.content import (
    GenerationRequest,
    GenerationResult,
    GenerationVariant,
    MediaPlan,
    QualityIssue,
    QualityReport,
)
from app.services.humanizer import HumanizerService

_ANGLE_IDS = count(1)


def _variant(text: str, *, parts: tuple[str, ...] = ()) -> GenerationVariant:
    return GenerationVariant(
        label="test",
        text=text,
        parts=parts,
        character_count=len(text),
        similarity_risk=0.0,
        angle_id=f"angle-{next(_ANGLE_IDS)}",
        angle_type="plain_update",
        evidence_ids=("legacy-evidence",),
        confidence="medium",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
    )


def _revision_inputs() -> tuple[GenerationRequest, GenerationResult, tuple[QualityReport, ...]]:
    request = GenerationRequest(
        idea_summary="Check natural tone",
        content_type=ContentType.REPLY,
        language="en",
    )
    variants = tuple(
        GenerationVariant(
            label=f"target-{index}",
            text=f"The real lesson is target {index}.",
            similarity_risk=0.0,
            angle_id=f"angle-{index}",
            angle_type=angle.angle_type,
            evidence_ids=angle.evidence_ids,
            confidence=angle.confidence,
            factual_claims=(),
            attributed_claims=(),
            uncertainty_markers=(),
        )
        for index, angle in enumerate(request.angle_candidates, start=1)
    )
    result = GenerationResult(
        idea_summary=request.idea_summary,
        recommended_format=request.content_type,
        variants=variants,
        recommended_variant=0,
        recommendation_reason="test",
        media_plan=MediaPlan(),
    )
    reports = tuple(
        QualityReport(
            passed=False,
            specificity_score=0.8,
            evidence_score=1.0,
            naturalness_score=0.4,
            distinctiveness_score=1.0,
            issues=(
                QualityIssue(
                    code="INSIGHT_PACKAGING",
                    severity="error",
                    message="Packaged insight",
                    suggested_action="flatten",
                ),
            ),
        )
        for _ in variants
    )
    return request, result, reports


def test_initial_generation_does_not_receive_humanizer_rules() -> None:
    request, _, _ = _revision_inputs()

    prepared = HumanizerService().prepare_request(request)

    assert prepared is request
    assert prepared.revision_targets == ()
    assert "humanizer" not in prepared.metadata


def test_humanizer_revision_receives_source_variants_and_named_issue_codes() -> None:
    request, result, reports = _revision_inputs()

    prepared = HumanizerService().prepare_revision_request(
        request,
        result=result,
        reports=reports,
        feedback="Fix only INSIGHT_PACKAGING.",
    )

    assert prepared is not request
    assert prepared.metadata["humanizer"] == "compact_conditional"
    assert prepared.metadata["humanizer_issue_codes"] == ["INSIGHT_PACKAGING"]
    assert prepared.revision_issue_codes == ("INSIGHT_PACKAGING",)
    assert tuple(target.variant.text for target in prepared.revision_targets) == tuple(
        variant.text for variant in result.variants
    )
    assert all(target.issue_codes == ("INSIGHT_PACKAGING",) for target in prepared.revision_targets)
    assert "claim personal experience" not in prepared.instructions[-1].casefold()
    assert "do not replace deleted rhetoric" in prepared.instructions[-1].casefold()


def test_humanizer_cleanup_preserves_facts_and_recomputes_character_count() -> None:
    text = (
        "It is important to note that the launch is scheduled for July 17, 2026."
        "\n\n\nWe will review the result."
    )
    result = GenerationResult(
        idea_summary="Launch",
        recommended_format=ContentType.SHORT_POST,
        variants=(_variant(text), _variant(text), _variant(text)),
        recommended_variant=0,
        recommendation_reason="test",
        media_plan=MediaPlan(),
    )

    processed = HumanizerService(mode="prompt_and_cleanup").process_result(result)

    assert "It is important to note" not in processed.variants[0].text
    assert "July 17, 2026" in processed.variants[0].text
    assert "\n\n\n" not in processed.variants[0].text
    assert processed.variants[0].character_count == len(processed.variants[0].text)


def test_humanizer_does_not_translate_english_phrases_during_cleanup() -> None:
    text = "This could be a game changer for a team of five."
    result = GenerationResult(
        idea_summary="English",
        recommended_format=ContentType.SHORT_POST,
        variants=(_variant(text), _variant(text), _variant(text)),
        recommended_variant=0,
        recommendation_reason="test",
        media_plan=MediaPlan(),
    )

    processed = HumanizerService(mode="prompt_and_cleanup").process_result(result)

    assert processed.variants[0].text == text


def test_humanizer_never_turns_a_valid_variant_into_empty_text() -> None:
    text = "It is important to note that"
    result = GenerationResult(
        idea_summary="Fallback",
        recommended_format=ContentType.SHORT_POST,
        variants=(_variant(text), _variant(text), _variant(text)),
        recommended_variant=0,
        recommendation_reason="test",
        media_plan=MediaPlan(),
    )

    processed = HumanizerService(mode="prompt_and_cleanup").process_result(result)

    assert processed.variants[0].text == text


def test_humanizer_prefers_configured_codex_skill_and_references(tmp_path) -> None:
    skill_dir = tmp_path / "personal-humanizer"
    references = skill_dir / "references"
    references.mkdir(parents=True)
    skill = skill_dir / "SKILL.md"
    skill.write_text("Lead with the useful point.", encoding="utf-8")
    (references / "runtime-contract.md").write_text("Avoid fake depth.", encoding="utf-8")
    (references / "semantic-lint.md").write_text("Use concrete details.", encoding="utf-8")
    request, result, reports = _revision_inputs()

    prepared = HumanizerService(
        external_skill_path=skill, include_references=True
    ).prepare_revision_request(
        request,
        result=result,
        reports=reports,
        feedback="Fix only INSIGHT_PACKAGING.",
    )

    assert prepared.metadata["humanizer_source"] == "external_runtime_contract"
    assert "Avoid fake depth" in prepared.instructions[-1]
    assert "Use concrete details" in prepared.instructions[-1]


def test_humanizer_uses_bundled_fallback_when_external_skill_is_missing(tmp_path) -> None:
    request, result, reports = _revision_inputs()

    prepared = HumanizerService(
        external_skill_path=tmp_path / "missing" / "SKILL.md"
    ).prepare_revision_request(
        request,
        result=result,
        reports=reports,
        feedback="Fix only INSIGHT_PACKAGING.",
    )

    assert prepared.metadata["humanizer_source"] == "bundled_reply_runtime"
    assert "deletion-first revision" in prepared.instructions[-1].casefold()
    assert "x_reply_micro" in prepared.instructions[-1].casefold()


def test_humanizer_prefers_external_compact_runtime_contract(tmp_path) -> None:
    skill_dir = tmp_path / "personal-humanizer"
    references = skill_dir / "references"
    references.mkdir(parents=True)
    skill = skill_dir / "SKILL.md"
    skill.write_text("Long canonical skill text.", encoding="utf-8")
    (references / "runtime-contract.md").write_text(
        "Compact deletion-first reply contract.", encoding="utf-8"
    )
    request, result, reports = _revision_inputs()

    prepared = HumanizerService(external_skill_path=skill).prepare_revision_request(
        request,
        result=result,
        reports=reports,
        feedback="Fix only INSIGHT_PACKAGING.",
    )

    assert prepared.metadata["humanizer_source"] == "external_runtime_contract"
    assert "Compact deletion-first reply contract" in prepared.instructions[-1]
    assert "Long canonical skill text" not in prepared.instructions[-1]
