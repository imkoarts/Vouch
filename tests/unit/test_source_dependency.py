from __future__ import annotations

from app.schemas.content import QualityReport, UntrustedSourceData
from app.services.draft_service import DraftService
from app.services.source_dependency import (
    SourceDependencyAssessment,
    assess_source_dependency,
)


def _source(text: str, *, media: bool = False) -> UntrustedSourceData:
    return UntrustedSourceData(
        source_id="2076668264755732487",
        content=text,
        verification_status="single_source",
        media_metadata={"items": [{"type": "video"}]} if media else {},
    )


def _passed_report() -> QualityReport:
    return QualityReport(
        passed=True,
        thesis_score=1.0,
        specificity_score=1.0,
        evidence_score=1.0,
        naturalness_score=1.0,
        plainness_score=1.0,
        rhetorical_density_score=0.0,
        reply_fit_score=1.0,
        distinctiveness_score=1.0,
    )


def test_putin_wording_commentary_requires_quote_context() -> None:
    source = _source(
        "Putin: Ukrainian strikes have caused some problems with Russian oil products, "
        "but the situation will gradually improve."
    )
    draft = (
        "“Some problems” is carrying a lot of freight. Putin says Ukrainian strikes have "
        "caused problems with Russian oil products, then says the situation will gradually "
        "improve. The sentence acknowledges a present difficulty while holding onto confidence "
        "about the next phase, with the details kept at a polite distance. A careful way to "
        "describe trouble without letting it occupy too much of the sentence."
    )

    assessment = assess_source_dependency(
        draft_text=draft,
        source=source,
        angle_type="dry_observation",
        angle_context_preference="quote_post",
    )

    assert assessment.recommended_format == "quote_post"
    assert assessment.source_dependency >= 0.65
    assert assessment.wording_commentary is True
    assert "reason for the reaction without the source" in assessment.missing_context


def test_icc_event_post_is_standalone_but_added_value_is_weak() -> None:
    source = _source(
        "JUST IN: Trump administration launches campaign to isolate the International "
        "Criminal Court & pressure countries to reject its authority over Americans."
    )
    draft = (
        "The Trump administration is launching a campaign to isolate the International "
        "Criminal Court and press countries to reject its authority over Americans. The "
        "immediate effect is unusually concrete: the argument is not just about a court's "
        "jurisdiction, but whether other governments will be asked to take a position on it."
    )

    assessment = assess_source_dependency(
        draft_text=draft,
        source=source,
        angle_type="practical_implication",
        angle_context_preference="standalone_post",
    )

    assert assessment.recommended_format == "standalone_post"
    assert assessment.standalone_clarity >= 0.9
    assert assessment.wording_commentary is False
    assert assessment.added_value_score < 0.46


def test_borderline_reaction_requests_one_minimum_context_rewrite() -> None:
    assessment = assess_source_dependency(
        draft_text="That is a careful way to admit the problem without naming the scale.",
        source=_source(
            "A company said the outage affected its payment API and service is recovering."
        ),
        angle_type="plain_update",
        angle_context_preference="either",
    )

    assert assessment.recommended_format == "rewrite_with_context"
    assert assessment.rewrite_instruction
    assert assessment.missing_context


def test_attached_source_media_can_carry_visual_context() -> None:
    assessment = assess_source_dependency(
        draft_text="this might be the only time messi looked even slightly arrogant",
        source=_source("the only time i saw messi arrogant", media=True),
        angle_type="dry_observation",
        angle_context_preference="quote_post",
        source_context_strategy="attach_source_media",
    )

    assert assessment.recommended_format == "standalone_post"
    assert assessment.reason.startswith("The attached source media")


def test_source_dependency_report_codes_are_deterministic() -> None:
    quote = SourceDependencyAssessment(
        recommended_format="quote_post",
        source_dependency=0.9,
        standalone_clarity=0.3,
        wording_commentary=True,
        reason="comments on wording",
        added_value_score=0.8,
    )
    gap = SourceDependencyAssessment(
        recommended_format="rewrite_with_context",
        source_dependency=0.5,
        standalone_clarity=0.4,
        wording_commentary=False,
        reason="missing event",
        added_value_score=0.8,
    )
    weak = SourceDependencyAssessment(
        recommended_format="standalone_post",
        source_dependency=0.1,
        standalone_clarity=1.0,
        wording_commentary=False,
        reason="standalone",
        added_value_score=0.3,
    )

    reports, _ = DraftService._reports_with_source_dependency(
        (_passed_report(), _passed_report(), _passed_report()),
        (quote, gap, weak),
        allow_quote_fallback=False,
    )

    assert reports[0].passed is True
    assert {issue.code for issue in reports[0].issues} == {"QUOTE_CONTEXT_REQUIRED"}
    assert reports[1].passed is False
    assert {issue.code for issue in reports[1].issues} == {"STANDALONE_CONTEXT_GAP"}
    assert reports[2].passed is False
    assert {issue.code for issue in reports[2].issues} == {"WEAK_ADDED_VALUE"}


def test_context_rewrite_falls_back_to_quote_after_bounded_attempt() -> None:
    gap = SourceDependencyAssessment(
        recommended_format="rewrite_with_context",
        source_dependency=0.5,
        standalone_clarity=0.4,
        wording_commentary=False,
        reason="missing event",
        added_value_score=0.8,
    )

    reports, assessments = DraftService._reports_with_source_dependency(
        (_passed_report(),),
        (gap,),
        allow_quote_fallback=True,
    )

    assert reports[0].passed is True
    assert {issue.code for issue in reports[0].issues} == {"QUOTE_CONTEXT_REQUIRED"}
    assert assessments[0] is not None
    assert assessments[0].recommended_format == "quote_post"


def test_single_source_standalone_claim_requires_fact_check() -> None:
    from app.domain.enums import ContentType, GenerationMode
    from app.schemas.content import (
        AngleCandidate,
        EvidenceItem,
        EvidencePacket,
        GenerationRequest,
        GenerationVariant,
    )

    packet = EvidencePacket(
        mode=GenerationMode.SOURCE_POST,
        topic="ICC campaign",
        items=(
            EvidenceItem(
                evidence_id="source-1",
                source_type="x_post",
                text="The administration launches a campaign against the ICC.",
                verification_status="single_source",
            ),
        ),
        language="en",
    )
    angle = AngleCandidate(
        angle_id="source-event-bounded-interpretation",
        angle_type="practical_implication",
        thesis="Explain one bounded implication.",
        evidence_ids=("source-1",),
        why_interesting="test",
        confidence="medium",
    )
    request = GenerationRequest(
        idea_summary="ICC campaign",
        content_type=ContentType.LONG_POST,
        language="en",
        generation_mode=GenerationMode.SOURCE_POST,
        evidence_packet=packet,
        angle_candidates=(
            angle,
            angle.model_copy(update={"angle_id": "a2"}),
            angle.model_copy(update={"angle_id": "a3"}),
        ),
        metadata={"single_source_anchor": True},
    )
    variant = GenerationVariant(
        label="event",
        text="The administration is launching a campaign against the ICC.",
        similarity_risk=0.0,
        angle_id=angle.angle_id,
        angle_type=angle.angle_type,
        evidence_ids=("source-1",),
        confidence="medium",
        factual_claims=("The administration is launching a campaign against the ICC.",),
        attributed_claims=(),
        uncertainty_markers=(),
    )
    standalone = SourceDependencyAssessment(
        recommended_format="standalone_post",
        source_dependency=0.1,
        standalone_clarity=1.0,
        wording_commentary=False,
        reason="standalone",
        added_value_score=0.8,
    )
    quote = standalone.model_copy(update={"recommended_format": "quote_post"})

    assert DraftService._deterministic_fact_check_required(variant, request, standalone) is True
    assert DraftService._deterministic_fact_check_required(variant, request, quote) is False


def test_quote_post_commentary_uses_short_effective_length_even_when_medium_selected() -> None:
    from app.domain.enums import PostLengthMode
    from app.services.post_length import resolve_post_length

    selection = resolve_post_length(PostLengthMode.MEDIUM)
    quote = SourceDependencyAssessment(
        recommended_format="quote_post",
        source_dependency=0.9,
        standalone_clarity=0.4,
        wording_commentary=True,
        reason="comments on exact source wording",
        added_value_score=0.8,
    )
    standalone = quote.model_copy(update={"recommended_format": "standalone_post"})
    text = "that phrase is doing more work than the rest of the statement"

    assert DraftService._fits_post_length(text, selection, quote) is True
    assert DraftService._fits_post_length(text, selection, standalone) is False


def test_source_dependency_reports_keep_one_assessment_per_variant() -> None:
    quote = SourceDependencyAssessment(
        recommended_format="quote_post",
        source_dependency=0.9,
        standalone_clarity=0.3,
        wording_commentary=True,
        reason="comments on wording",
        added_value_score=0.8,
    )

    reports, assessments = DraftService._reports_with_source_dependency(
        (_passed_report(), _passed_report(), _passed_report()),
        (quote, quote, quote),
        allow_quote_fallback=False,
    )

    assert len(reports) == 3
    assert len(assessments) == 3
