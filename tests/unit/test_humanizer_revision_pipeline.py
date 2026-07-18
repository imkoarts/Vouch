from __future__ import annotations

from types import MethodType

import pytest

from app.domain.enums import ContentType, PostLengthMode
from app.schemas.content import (
    GenerationRequest,
    GenerationResult,
    GenerationVariant,
    MediaPlan,
    QualityIssue,
    QualityReport,
)
from app.services.draft_service import (
    DraftService,
    EditorialQualityError,
    PostLengthMismatchError,
)
from app.services.editorial_quality import assess_result
from app.services.humanizer import HumanizerService
from app.services.post_length import resolve_post_length


def _variant(request: GenerationRequest, index: int, text: str) -> GenerationVariant:
    angle = request.angle_candidates[index]
    return GenerationVariant(
        label=f"variant-{index + 1}",
        text=text,
        similarity_risk=0.0,
        angle_id=angle.angle_id,
        angle_type=angle.angle_type,
        evidence_ids=angle.evidence_ids,
        confidence=angle.confidence,
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
    )


def _result(request: GenerationRequest, texts: tuple[str, str, str]) -> GenerationResult:
    return GenerationResult(
        idea_summary=request.idea_summary,
        recommended_format=request.content_type,
        variants=tuple(_variant(request, index, text) for index, text in enumerate(texts)),
        recommended_variant=0,
        recommendation_reason="test",
        media_plan=MediaPlan(),
    )


def _report(*issues: QualityIssue, passed: bool = False) -> QualityReport:
    return QualityReport(
        passed=passed,
        thesis_score=1.0,
        specificity_score=1.0,
        evidence_score=1.0,
        naturalness_score=1.0 if passed else 0.2,
        plainness_score=1.0 if passed else 0.2,
        rhetorical_density_score=0.0 if passed else 0.8,
        reply_fit_score=1.0 if passed else 0.2,
        distinctiveness_score=1.0,
        issues=issues,
    )


class _RecordingProvider:
    name = "recording"
    model = "test"

    def __init__(self, results: tuple[GenerationResult, ...]) -> None:
        self.results = list(results)
        self.requests: list[GenerationRequest] = []

    async def generate_structured(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return self.results.pop(0)


def _service(provider: _RecordingProvider) -> DraftService:
    service = object.__new__(DraftService)
    service.provider = provider
    service.humanizer = HumanizerService()
    service.editorial_quality_retry_count = 1
    return service


@pytest.mark.asyncio
async def test_quality_retry_revises_exact_source_variants_with_named_codes() -> None:
    request = GenerationRequest(
        idea_summary="Model portability",
        content_type=ContentType.REPLY,
        language="en",
    )
    initial = _result(
        request,
        (
            "The real moat is the correction history.",
            "Most teams miss the eval history.",
            "Companies spent years securing data and now export it.",
        ),
    )
    revised = _result(
        request,
        (
            "The correction history can be harder to move than the API.",
            "The eval history may need a separate migration path.",
            "Corrections can remain tied to one provider.",
        ),
    )
    issue = QualityIssue(
        code="INSIGHT_PACKAGING",
        severity="error",
        message="Packaged insight",
        suggested_action="flatten",
    )
    failed = tuple(_report(issue) for _ in range(3))
    passed = tuple(_report(passed=True) for _ in range(3))
    provider = _RecordingProvider((initial, revised))
    service = _service(provider)
    calls = 0

    def select_variant(
        self: DraftService,
        result: GenerationResult,
        current_request: GenerationRequest,
        selection: object,
    ) -> tuple[GenerationVariant, int, tuple[QualityReport, ...]]:
        nonlocal calls
        del self, selection
        calls += 1
        if calls == 1:
            raise EditorialQualityError(failed)
        assert current_request.revision_issue_codes == ("INSIGHT_PACKAGING",)
        assert tuple(target.variant.text for target in current_request.revision_targets) == tuple(
            variant.text for variant in initial.variants
        )
        return result.variants[0], 0, passed

    service._select_variant = MethodType(select_variant, service)

    await service._generate_quality_checked(request, None)

    assert len(provider.requests) == 2
    assert provider.requests[0].revision_targets == ()
    assert provider.requests[1].revision_targets
    assert provider.requests[1].metadata["humanizer_revision"] is True


@pytest.mark.asyncio
async def test_report_about_report_draft_is_sent_to_bounded_humanizer_revision() -> None:
    request = GenerationRequest(
        idea_summary="Wimbledon final",
        content_type=ContentType.LONG_POST,
        language="en",
        editorial_intent="explain_topic",
    )
    bad_text = (
        "One report says Jannik Sinner lost the opening set before defeating Alexander "
        "Zverev and defending his Wimbledon title. If that account is accurate, the "
        "match's turning point was not the start but the response after it. The result is "
        "specific; the report does not explain which changes in play produced the comeback."
    )
    initial = _result(request, (bad_text, bad_text, bad_text))
    revised = _result(
        request,
        (
            "Jannik Sinner lost the opening set and still defended the title against "
            "Alexander Zverev. The result is enough for the post; the evidence does not "
            "support a claim about which tactical change produced the comeback.",
            "Jannik Sinner dropped the opening set before beating Alexander Zverev and "
            "defending the title. Nothing in the supplied evidence supports a tactical "
            "explanation for the comeback.",
            "Jannik Sinner recovered from losing the first set to beat Alexander Zverev and "
            "defend the title. Keep the post on that result rather than inventing the change "
            "that produced it.",
        ),
    )
    provider = _RecordingProvider((initial, revised))
    service = _service(provider)
    calls = 0

    def select_variant(
        self: DraftService,
        result: GenerationResult,
        current_request: GenerationRequest,
        selection: object,
    ) -> tuple[GenerationVariant, int, tuple[QualityReport, ...]]:
        nonlocal calls
        del self, selection
        calls += 1
        if calls == 1:
            reports = assess_result(result, current_request)
            raise EditorialQualityError(reports)
        required = {
            "VAGUE_ATTRIBUTION",
            "MEDIA_BRAND_VOICE",
            "GENERIC_VERIFICATION",
            "CONDITIONAL_ESCAPE",
            "EXPLANATION_LEAKAGE",
            "SOURCE_SUMMARY_ONLY",
        }
        assert required <= set(current_request.revision_issue_codes)
        assert all(target.variant.text == bad_text for target in current_request.revision_targets)
        return result.variants[0], 0, tuple(_report(passed=True) for _ in range(3))

    service._select_variant = MethodType(select_variant, service)

    await service._generate_quality_checked(request, None)

    assert len(provider.requests) == 2
    assert provider.requests[1].metadata["humanizer_revision"] is True


@pytest.mark.asyncio
async def test_empty_thesis_is_redrafted_upstream_without_humanizer() -> None:
    request = GenerationRequest(
        idea_summary="Model portability",
        content_type=ContentType.REPLY,
        language="en",
    )
    initial = _result(request, ("Empty.", "Empty two.", "Empty three."))
    redrafted = _result(
        request,
        (
            "The eval history may need a separate migration path.",
            "The correction log can remain provider-specific.",
            "The API is not the only part that must move.",
        ),
    )
    issue = QualityIssue(
        code="EMPTY_THESIS",
        severity="error",
        message="No contribution",
        suggested_action="regenerate_angle",
    )
    failed = tuple(_report(issue) for _ in range(3))
    passed = tuple(_report(passed=True) for _ in range(3))
    provider = _RecordingProvider((initial, redrafted))
    service = _service(provider)
    calls = 0

    def select_variant(
        self: DraftService,
        result: GenerationResult,
        current_request: GenerationRequest,
        selection: object,
    ) -> tuple[GenerationVariant, int, tuple[QualityReport, ...]]:
        nonlocal calls
        del self, current_request, selection
        calls += 1
        if calls == 1:
            raise EditorialQualityError(failed)
        return result.variants[0], 0, passed

    service._select_variant = MethodType(select_variant, service)

    await service._generate_quality_checked(request, None)

    assert len(provider.requests) == 2
    assert provider.requests[1].revision_targets == ()
    assert provider.requests[1].revision_issue_codes == ()
    assert provider.requests[1].metadata.get("humanizer_revision") is None
    assert "Replan from the supplied evidence" in (provider.requests[1].feedback or "")
    assert "status=no_post_needed" in (provider.requests[1].feedback or "")


def test_revision_integrity_rejects_changed_claim_boundary() -> None:
    request = GenerationRequest(
        idea_summary="Model portability",
        content_type=ContentType.REPLY,
        language="en",
    )
    source = _result(
        request,
        (
            "The correction history may be harder to move than the API.",
            "The eval history may need a separate migration path.",
            "Corrections may remain tied to one provider.",
        ),
    )
    reports = tuple(
        _report(
            QualityIssue(
                code="UNNECESSARY_METAPHOR",
                severity="warning",
                message="Review literal wording",
                suggested_action="flatten",
            )
        )
        for _ in range(3)
    )
    revision_request = HumanizerService().prepare_revision_request(
        request,
        result=source,
        reports=reports,
        feedback="Fix only UNNECESSARY_METAPHOR.",
    )
    changed = source.model_copy(
        update={
            "variants": tuple(
                variant.model_copy(
                    update={
                        "revision_status": "revised",
                        "factual_claims": ("A new unsupported fact.",),
                    }
                )
                for variant in source.variants
            )
        }
    )

    assessed = assess_result(changed, revision_request)

    assert all("UNSUPPORTED_FACT" in {issue.code for issue in report.issues} for report in assessed)


def test_no_change_needed_cannot_modify_source_text() -> None:
    request = GenerationRequest(
        idea_summary="Model portability",
        content_type=ContentType.REPLY,
        language="en",
    )
    source = _result(
        request,
        (
            "The correction history may be harder to move than the API.",
            "The eval history may need a separate migration path.",
            "Corrections may remain tied to one provider.",
        ),
    )
    reports = tuple(
        _report(
            QualityIssue(
                code="UNNECESSARY_METAPHOR",
                severity="warning",
                message="Review literal wording",
                suggested_action="flatten",
            )
        )
        for _ in range(3)
    )
    revision_request = HumanizerService().prepare_revision_request(
        request,
        result=source,
        reports=reports,
        feedback="Fix only UNNECESSARY_METAPHOR.",
    )
    modified = source.model_copy(
        update={
            "variants": tuple(
                variant.model_copy(
                    update={
                        "text": variant.text + " Extra conclusion.",
                        "revision_status": "no_change_needed",
                    }
                )
                for variant in source.variants
            )
        }
    )

    assessed = assess_result(modified, revision_request)

    assert all("FORMAT_MISMATCH" in {issue.code for issue in report.issues} for report in assessed)


def test_insufficient_context_revision_is_blocked_from_persistence() -> None:
    request = GenerationRequest(
        idea_summary="Model portability",
        content_type=ContentType.REPLY,
        language="en",
    )
    source = _result(
        request,
        (
            "The correction history may be harder to move than the API.",
            "The eval history may need a separate migration path.",
            "Corrections may remain tied to one provider.",
        ),
    )
    reports = tuple(
        _report(
            QualityIssue(
                code="UNSUPPORTED_STAKES",
                severity="error",
                message="Missing risk mechanism",
                suggested_action="ground",
            )
        )
        for _ in range(3)
    )
    revision_request = HumanizerService().prepare_revision_request(
        request,
        result=source,
        reports=reports,
        feedback="Fix only UNSUPPORTED_STAKES.",
    )
    insufficient = source.model_copy(
        update={
            "variants": tuple(
                variant.model_copy(update={"revision_status": "insufficient_context"})
                for variant in source.variants
            )
        }
    )

    assessed = assess_result(insufficient, revision_request)

    assert all(
        "INSUFFICIENT_CONTEXT" in {issue.code for issue in report.issues} for report in assessed
    )
    assert all(not report.passed for report in assessed)


def test_revision_must_preserve_links() -> None:
    request = GenerationRequest(
        idea_summary="Model portability",
        content_type=ContentType.REPLY,
        language="en",
    )
    source = _result(
        request,
        (
            "The migration note is at https://example.test/a.",
            "The eval note is at https://example.test/b.",
            "The correction note is at https://example.test/c.",
        ),
    )
    reports = tuple(
        _report(
            QualityIssue(
                code="UNNECESSARY_METAPHOR",
                severity="warning",
                message="Review literal wording",
                suggested_action="flatten",
            )
        )
        for _ in range(3)
    )
    revision_request = HumanizerService().prepare_revision_request(
        request,
        result=source,
        reports=reports,
        feedback="Fix only UNNECESSARY_METAPHOR.",
    )
    removed = source.model_copy(
        update={
            "variants": tuple(
                variant.model_copy(
                    update={
                        "text": variant.text.split(" at ", 1)[0] + ".",
                        "revision_status": "revised",
                    }
                )
                for variant in source.variants
            )
        }
    )

    assessed = assess_result(removed, revision_request)

    assert all("UNSUPPORTED_FACT" in {issue.code for issue in report.issues} for report in assessed)


@pytest.mark.asyncio
async def test_length_mismatch_uses_bounded_in_place_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = GenerationRequest(
        idea_summary="Length repair",
        content_type=ContentType.LONG_POST,
        language="en",
        minimum_characters=300,
        maximum_characters=500,
        post_length_mode="medium",
    )
    short_texts = (
        "A concrete observation that is much too short.",
        "A second supported observation that is also too short.",
        "A third literal observation that still misses the range.",
    )
    repaired_texts = tuple(text + " " + ("Supported detail. " * 20) for text in short_texts)
    initial = _result(request, short_texts)
    revised = _result(request, repaired_texts)
    provider = _RecordingProvider((initial, revised))
    service = _service(provider)
    service.minimum_specificity_score = 0.0
    service.minimum_evidence_score = 0.0
    service.minimum_naturalness_score = 0.0
    service.maximum_recent_similarity = 1.0

    passed = tuple(_report(passed=True) for _ in range(3))
    monkeypatch.setattr("app.services.draft_service.assess_result", lambda *args, **kwargs: passed)

    result, selected, _, _ = await service._generate_quality_checked(
        request, resolve_post_length(PostLengthMode.MEDIUM)
    )

    assert result is revised
    assert 300 <= len(selected.text) <= 500
    assert len(provider.requests) == 2
    revision_request = provider.requests[1]
    assert revision_request.metadata["humanizer_revision"] is True
    assert revision_request.revision_issue_codes == ("FORMAT_MISMATCH",)
    assert tuple(target.variant.text for target in revision_request.revision_targets) == short_texts
    assert "Actual lengths:" in (revision_request.feedback or "")
    assert "do not pad with source recap" in (revision_request.feedback or "")


@pytest.mark.asyncio
async def test_length_mismatch_remains_strict_after_bounded_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = GenerationRequest(
        idea_summary="Length repair",
        content_type=ContentType.LONG_POST,
        language="en",
        minimum_characters=300,
        maximum_characters=500,
        post_length_mode="medium",
    )
    short = _result(request, ("Too short one.", "Too short two.", "Too short three."))
    provider = _RecordingProvider((short, short))
    service = _service(provider)
    service.minimum_specificity_score = 0.0
    service.minimum_evidence_score = 0.0
    service.minimum_naturalness_score = 0.0
    service.maximum_recent_similarity = 1.0
    passed = tuple(_report(passed=True) for _ in range(3))
    monkeypatch.setattr("app.services.draft_service.assess_result", lambda *args, **kwargs: passed)

    with pytest.raises(PostLengthMismatchError, match="actual weighted lengths"):
        await service._generate_quality_checked(request, resolve_post_length(PostLengthMode.MEDIUM))

    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_reviewable_editorial_errors_are_kept_after_one_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = GenerationRequest(
        idea_summary="A useful quoted source",
        content_type=ContentType.SHORT_POST,
        language="en",
        post_length_mode="short",
    )
    overlong = _result(request, tuple("A" * 320 for _ in range(3)))
    repaired = _result(
        request,
        (
            "The useful part is the constraint this puts on the next decision.",
            "This changes which tradeoff matters first.",
            "The result is more useful than the framing around it.",
        ),
    )
    provider = _RecordingProvider((overlong, repaired))
    service = _service(provider)
    service.minimum_specificity_score = 0.0
    service.minimum_evidence_score = 0.0
    service.minimum_naturalness_score = 0.0
    service.maximum_recent_similarity = 1.0
    passed = tuple(_report(passed=True) for _ in range(3))
    advisory = tuple(
        _report(
            QualityIssue(
                code="ANALYTICAL_PACKAGING",
                severity="error",
                message="Too packaged",
                suggested_action="flatten",
            ),
            QualityIssue(
                code="IMPLIED_RESTATEMENT",
                severity="error",
                message="Too close to the source implication",
                suggested_action="delete",
            ),
            QualityIssue(
                code="SOURCE_ECHO",
                severity="error",
                message="Source echo",
                suggested_action="ground",
            ),
        )
        for _ in range(3)
    )
    reports = iter((passed, advisory))
    monkeypatch.setattr(
        "app.services.draft_service.assess_result",
        lambda *args, **kwargs: next(reports),
    )

    result, selected, _, quality = await service._generate_quality_checked(
        request, resolve_post_length(PostLengthMode.SHORT)
    )

    assert result is repaired
    assert selected.text == repaired.variants[0].text
    assert quality[0].passed is False
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_interpretive_label_from_provider_is_kept_for_human_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = GenerationRequest(
        idea_summary="A grounded source that supports a reviewable reaction",
        content_type=ContentType.SHORT_POST,
        language="en",
        post_length_mode="short",
    )
    generated = _result(
        request,
        (
            "The useful part is how this changes the next decision.",
            "This is less about the headline and more about the constraint.",
            "The reaction is grounded, but the label can still be edited.",
        ),
    )
    provider = _RecordingProvider((generated, generated, generated))
    service = _service(provider)
    service.editorial_quality_retry_count = 2
    service.minimum_specificity_score = 0.0
    service.minimum_evidence_score = 0.0
    service.minimum_naturalness_score = 0.0
    service.maximum_recent_similarity = 1.0
    reports = tuple(
        _report(
            QualityIssue(
                code="UNSUPPORTED_INTERPRETIVE_LABEL",
                severity="error",
                message="The label needs operator review",
                suggested_action="ground",
            ),
            QualityIssue(
                code="SOURCE_STRONGER_THAN_DRAFT",
                severity="error",
                message="The source is stronger than the reaction",
                suggested_action="flatten",
            ),
        )
        for _ in range(3)
    )
    monkeypatch.setattr(
        "app.services.draft_service.assess_result",
        lambda *args, **kwargs: reports,
    )

    result, selected, _, quality = await service._generate_quality_checked(
        request, resolve_post_length(PostLengthMode.SHORT)
    )

    assert result is generated
    assert selected.text == generated.variants[0].text
    assert quality[0].passed is False
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_valid_review_draft_survives_a_regressing_length_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = GenerationRequest(
        idea_summary="Preserve the bounded draft",
        content_type=ContentType.SHORT_POST,
        language="en",
        post_length_mode="short",
    )
    initial = _result(
        request,
        (
            "This is a grounded observation that a person can still improve before approval.",
            "The source supports a smaller and more direct reaction.",
            "The constraint matters more than a polished thesis.",
        ),
    )
    overlong = _result(request, tuple("B" * 320 for _ in range(3)))
    provider = _RecordingProvider((initial, overlong))
    service = _service(provider)
    service.minimum_specificity_score = 0.0
    service.minimum_evidence_score = 0.0
    service.minimum_naturalness_score = 0.0
    service.maximum_recent_similarity = 1.0
    advisory = tuple(
        _report(
            QualityIssue(
                code="LOW_INFORMATION_GAIN",
                severity="error",
                message="Limited added value",
                suggested_action="ground",
            ),
            QualityIssue(
                code="PROMISED_INSIGHT_NOT_DELIVERED",
                severity="error",
                message="The insight is weak",
                suggested_action="flatten",
            ),
        )
        for _ in range(3)
    )
    passed = tuple(_report(passed=True) for _ in range(3))
    reports = iter((advisory, passed))
    monkeypatch.setattr(
        "app.services.draft_service.assess_result",
        lambda *args, **kwargs: next(reports),
    )

    result, selected, _, quality = await service._generate_quality_checked(
        request, resolve_post_length(PostLengthMode.SHORT)
    )

    assert result is initial
    assert selected.text == initial.variants[0].text
    assert quality[0].passed is False
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_unsupported_fact_never_becomes_a_reviewable_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = GenerationRequest(
        idea_summary="Do not weaken factual validation",
        content_type=ContentType.SHORT_POST,
        language="en",
    )
    generated = _result(request, ("A supported length.",) * 3)
    provider = _RecordingProvider((generated, generated))
    service = _service(provider)
    hard = tuple(
        _report(
            QualityIssue(
                code="UNSUPPORTED_FACT",
                severity="error",
                message="Invented fact",
                suggested_action="ground",
            )
        )
        for _ in range(3)
    )
    monkeypatch.setattr(
        "app.services.draft_service.assess_result",
        lambda *args, **kwargs: hard,
    )

    with pytest.raises(EditorialQualityError, match="UNSUPPORTED_FACT"):
        await service._generate_quality_checked(request, None)

    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_mixed_quality_and_length_failures_share_one_bounded_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = GenerationRequest(
        idea_summary="Mixed repair",
        content_type=ContentType.LONG_POST,
        language="en",
        minimum_characters=300,
        maximum_characters=500,
        post_length_mode="medium",
    )
    initial_texts = (
        "A" * 350,
        "B" * 220,
        "C" * 520,
    )
    repaired_texts = (
        "D" * 350,
        "E" * 360,
        "F" * 370,
    )
    initial = _result(request, initial_texts)
    revised = _result(request, repaired_texts)
    provider = _RecordingProvider((initial, revised))
    service = _service(provider)
    service.minimum_specificity_score = 0.0
    service.minimum_evidence_score = 0.0
    service.minimum_naturalness_score = 0.0
    service.maximum_recent_similarity = 1.0

    insight_issue = QualityIssue(
        code="INSIGHT_PACKAGING",
        severity="error",
        message="Packaged insight",
        suggested_action="flatten",
    )
    first_reports = (
        _report(insight_issue),
        _report(passed=True),
        _report(passed=True),
    )
    second_reports = tuple(_report(passed=True) for _ in range(3))
    reports = iter((first_reports, second_reports))
    monkeypatch.setattr(
        "app.services.draft_service.assess_result",
        lambda *args, **kwargs: next(reports),
    )

    result, selected, _, _ = await service._generate_quality_checked(
        request, resolve_post_length(PostLengthMode.MEDIUM)
    )

    assert result is revised
    assert selected.text == repaired_texts[0]
    revision_request = provider.requests[1]
    assert set(revision_request.revision_issue_codes) == {
        "FORMAT_MISMATCH",
        "INSIGHT_PACKAGING",
    }
    assert revision_request.revision_targets[0].issue_codes == ("INSIGHT_PACKAGING",)
    assert revision_request.revision_targets[1].issue_codes == ("FORMAT_MISMATCH",)
    assert revision_request.revision_targets[2].issue_codes == ("FORMAT_MISMATCH",)


@pytest.mark.asyncio
async def test_refusal_markers_are_redrafted_before_length_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = GenerationRequest(
        idea_summary="A current event with usable source evidence",
        content_type=ContentType.LONG_POST,
        language="en",
        minimum_characters=300,
        maximum_characters=500,
        post_length_mode="medium",
    )
    refusal = _result(
        request,
        (
            "insufficient context.",
            "insufficient context.",
            "insufficient context.",
        ),
    )
    publishable_texts = (
        "A concrete result can support a real post without turning the source into the subject. "
        "The useful angle is the response after the setback, because that is the part visible in "
        "the supplied evidence. The post can stay with that contrast and avoid inventing a hidden "
        "mechanism, tactical explanation, or claim about what an unnamed report failed to show.",
        "The event already contains enough tension for a personal observation: an early setback "
        "did not decide the outcome. That gives the post a clear center without a news lead or "
        "an accuracy disclaimer. The rest should develop the same point with the named result "
        "and stop "
        "before it becomes a general lesson about resilience.",
        "A narrow evidence packet does not require a refusal. Start with the concrete action and "
        "result, then add one bounded interpretation about why the sequence is interesting. "
        "That is "
        "enough to reach the requested length without vague attribution, generic background, a new "
        "fact, or a sentence explaining what the source does not contain.",
    )
    redrafted = _result(request, publishable_texts)
    provider = _RecordingProvider((refusal, redrafted))
    service = _service(provider)
    service.minimum_specificity_score = 0.0
    service.minimum_evidence_score = 0.0
    service.minimum_naturalness_score = 0.0
    service.maximum_recent_similarity = 1.0

    empty_issue = QualityIssue(
        code="EMPTY_THESIS",
        severity="error",
        message="The provider returned no publishable observation.",
        suggested_action="regenerate_angle",
    )
    first_reports = tuple(_report(empty_issue) for _ in range(3))
    second_reports = tuple(_report(passed=True) for _ in range(3))
    reports = iter((first_reports, second_reports))
    monkeypatch.setattr(
        "app.services.draft_service.assess_result",
        lambda *args, **kwargs: next(reports),
    )

    result, selected, _, _ = await service._generate_quality_checked(
        request, resolve_post_length(PostLengthMode.MEDIUM)
    )

    assert result is redrafted
    assert 300 <= len(selected.text) <= 500
    assert len(provider.requests) == 2
    redraft_request = provider.requests[1]
    assert redraft_request.revision_targets == ()
    assert redraft_request.metadata["upstream_redraft"] is True
    assert redraft_request.metadata["upstream_issue_codes"] == ["EMPTY_THESIS", "FORMAT_MISMATCH"]
    assert "Replan from the supplied evidence" in (redraft_request.feedback or "")
    assert "status=no_post_needed" in (redraft_request.feedback or "")
    assert "300-500 weighted X characters" in (redraft_request.feedback or "")
