from __future__ import annotations

from app.domain.enums import ContentType, GenerationMode
from app.schemas.content import GenerationRequest, GenerationResult, GenerationVariant, MediaPlan
from app.services.editorial_quality import assess_result


def _variant(
    text: str,
    *,
    angle_id: str,
    angle_type: str,
    evidence_ids: tuple[str, ...],
) -> GenerationVariant:
    return GenerationVariant(
        label=angle_id,
        text=text,
        character_count=len(text),
        similarity_risk=0.0,
        angle_id=angle_id,
        angle_type=angle_type,
        evidence_ids=evidence_ids,
        source_post_ids=evidence_ids,
        confidence="medium",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
    )


def _request() -> GenerationRequest:
    request = GenerationRequest(
        idea_summary="One Messi video moment",
        content_type=ContentType.SHORT_POST,
        language="en",
        generation_mode=GenerationMode.SOURCE_POST,
    )
    return request.model_copy(
        update={
            "metadata": {
                "single_source_anchor": True,
                "anchor_evidence_id": "2076265333803913327",
                "anchor_author_username": "bdll_10",
                "source_context_strategy": "attach_source_media",
            }
        }
    )


def test_messi_source_collage_is_rejected_even_when_sentences_are_grounded() -> None:
    request = _request()
    angles = request.angle_candidates
    bad = (
        "@bdll_10 calls a Messi moment the only time Messi looked arrogant. "
        "@FIFAWC26Updates highlights the last four with a trophy. The contrast is narrow "
        "but clear: one line fixes attention on Messi's manner, while the other fixes it on "
        "the remaining tournament field."
    )
    result = GenerationResult(
        idea_summary=request.idea_summary,
        recommended_format=request.content_type,
        variants=(
            _variant(
                bad,
                angle_id=angles[0].angle_id,
                angle_type=angles[0].angle_type,
                evidence_ids=("2076265333803913327", "2076600215105404990"),
            ),
            _variant(
                "@bdll_10 says this is the only arrogant Messi moment.",
                angle_id=angles[1].angle_id,
                angle_type=angles[1].angle_type,
                evidence_ids=("2076265333803913327",),
            ),
            _variant(
                "the smile makes the whole accusation look optimistic",
                angle_id=angles[2].angle_id,
                angle_type=angles[2].angle_type,
                evidence_ids=("2076265333803913327",),
            ),
        ),
        recommended_variant=0,
        recommendation_reason="test",
        media_plan=MediaPlan(),
    )

    reports = assess_result(result, request)
    first_codes = {issue.code for issue in reports[0].issues}
    second_codes = {issue.code for issue in reports[1].issues}

    assert "MULTI_SOURCE_COLLAGE" in first_codes
    assert "SOURCE_AS_SUBJECT" in first_codes
    assert "SOURCE_AS_SUBJECT" in second_codes


def test_single_source_voice_without_source_narration_is_not_flagged() -> None:
    request = _request()
    angles = request.angle_candidates
    texts = (
        "Messi looked less arrogant than someone trying not to laugh at the accusation.",
        "The evidence for arrogant Messi is apparently one smile and a very generous caption.",
        "Eight seconds of Messi smiling somehow became a character trial.",
    )
    result = GenerationResult(
        idea_summary=request.idea_summary,
        recommended_format=request.content_type,
        variants=tuple(
            _variant(
                text,
                angle_id=angle.angle_id,
                angle_type=angle.angle_type,
                evidence_ids=("2076265333803913327",),
            )
            for text, angle in zip(texts, angles, strict=True)
        ),
        recommended_variant=0,
        recommendation_reason="test",
        media_plan=MediaPlan(),
    )

    reports = assess_result(result, request)
    for report in reports:
        codes = {issue.code for issue in report.issues}
        assert "MULTI_SOURCE_COLLAGE" not in codes
        assert "SOURCE_AS_SUBJECT" not in codes
        assert "MISSING_SOURCE_CONTEXT" not in codes
