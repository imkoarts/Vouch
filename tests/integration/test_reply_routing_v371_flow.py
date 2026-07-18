from __future__ import annotations

import pytest

from app.domain.enums import ContentType, GenerationMode
from app.providers.mock_provider import MockLLMProvider
from app.schemas.content import (
    CanonicalClaim,
    ContributionInspection,
    GenerationRequest,
    GenerationVariant,
    ReactionFunctionInspection,
    UntrustedSourceData,
)
from app.services.draft_service import DraftService
from app.services.editorial_quality import assess_variant
from app.services.generation_pipeline import build_generation_context
from app.services.semantic_classifier import (
    DeterministicSemanticInspector,
    derive_source_semantic_map,
    inspect_reaction_function,
)


def _context(source: str):
    return build_generation_context(
        idea_summary="Reply",
        idea_explanation={"origin": "reply_to_source", "editorial_intent": "reply_reaction"},
        language="en",
        sources=(UntrustedSourceData(source_id="source", source_type="x_post", content=source),),
        content_type=ContentType.REPLY,
    )


def _variant(text: str) -> GenerationVariant:
    return GenerationVariant(
        label="humor",
        text=text,
        similarity_risk=0.0,
        angle_id="reply-dry-humor",
        angle_type="concise_joke",
        evidence_ids=("source",),
        confidence="high",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
        contribution_id="reply-dry-humor",
        contribution_type="dry_humor",
    )


@pytest.mark.asyncio
async def test_hyphenated_anti_yap_candidate_is_reachable_end_to_end() -> None:
    context = _context("A model added a low-verbosity setting.")
    service = DraftService.__new__(DraftService)
    service.deterministic_semantic_inspector = DeterministicSemanticInspector()
    service.semantic_inspector = service.deterministic_semantic_inspector
    service.recent_corpus_limit = 0

    filtered, _, _ = await service._validate_context_semantics(context)
    variant = _variant("finally an official anti-yap switch")
    report = assess_variant(
        variant,
        evidence_packet=filtered.evidence,
        editorial_intent="reply_reaction",
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert "dry_humor" in {item.contribution_type for item in filtered.angles}
    assert inspect_reaction_function(filtered.evidence, variant.text).grounded
    assert "REACTION_LABEL_WITHOUT_FUNCTION" not in {item.code for item in report.issues}


@pytest.mark.asyncio
async def test_supported_nda_joke_reachable_and_unsupported_inference_blocked() -> None:
    service = DraftService.__new__(DraftService)
    service.deterministic_semantic_inspector = DeterministicSemanticInspector()
    service.semantic_inspector = service.deterministic_semantic_inspector
    service.recent_corpus_limit = 0
    supported = _context("A founder broke two NDAs in one tweet.")
    unsupported = _context("A founder disclosed two confidential partnership details in one tweet.")

    filtered, _, _ = await service._validate_context_semantics(supported)
    reply = _variant("bro broke two NDAs in one tweet")
    supported_report = assess_variant(
        reply,
        evidence_packet=filtered.evidence,
        editorial_intent="reply_reaction",
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )
    unsupported_report = assess_variant(
        reply,
        evidence_packet=unsupported.evidence,
        editorial_intent="reply_reaction",
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert "dry_humor" in {item.contribution_type for item in filtered.angles}
    assert inspect_reaction_function(filtered.evidence, reply.text).grounded
    assert supported_report.passed
    assert not unsupported_report.passed
    assert {item.code for item in unsupported_report.issues} & {
        "UNSUPPORTED_FACT",
        "EVIDENCE_INVALID",
        "REACTION_LABEL_WITHOUT_FUNCTION",
    }


class _AllGroundedInspector:
    def __init__(self, *, strong_humor: bool = False) -> None:
        self.strong_humor = strong_humor

    async def inspect_source(self, packet):
        return derive_source_semantic_map(packet)

    async def inspect_contributions(self, packet, contributions, source_map):
        del packet, source_map
        return tuple(
            ContributionInspection(
                contribution_id=item.angle_id,
                canonical_proposition=CanonicalClaim(
                    claim_id=item.angle_id,
                    exact_text=item.thesis,
                    kind="interpretation",
                    subject=item.supporting_relations[0].subject,
                    predicate=item.supporting_relations[0].predicate,
                    object=item.supporting_relations[0].object,
                    polarity="positive",
                    modality="asserted",
                ),
                already_covered=False,
                requires_new_assumption=False,
                grounded=True,
                reaction_function_present=(
                    item.contribution_type == "dry_humor" and self.strong_humor
                ),
                source_specificity_present=True,
                reaction_inspection=(
                    ReactionFunctionInspection(
                        reaction_type="wordplay",
                        source_specific=True,
                        source_echo=False,
                        unsupported_detail=False,
                        generic_template_only=False,
                        grounded=True,
                    )
                    if item.contribution_type == "dry_humor" and self.strong_humor
                    else None
                ),
            )
            for item in contributions
        )

    async def inspect_draft(self, *args, **kwargs):
        raise AssertionError("not used")


@pytest.mark.asyncio
async def test_all_family_ranking_is_shared_with_mock_provider() -> None:
    context = _context("Funny update! Parliament removed President Sulyok.")
    service = DraftService.__new__(DraftService)
    service.deterministic_semantic_inspector = _AllGroundedInspector()
    service.semantic_inspector = service.deterministic_semantic_inspector
    service.recent_corpus_limit = 0

    filtered, _, _ = await service._validate_context_semantics(context)
    request = GenerationRequest(
        idea_summary="reply",
        content_type=ContentType.REPLY,
        language="en",
        generation_mode=GenerationMode.REPLY,
        editorial_intent="reply_reaction",
        evidence_packet=filtered.evidence,
        source_coverage=filtered.source_coverage,
        available_evidence=filtered.available_evidence,
        content_mode=filtered.content_mode,
        signal_decision=filtered.signal_decision,
        angle_candidates=filtered.angles,
        literal_plans=filtered.literal_plans,
    )

    result = await MockLLMProvider().generate_structured(request)

    assert [item.contribution_type for item in filtered.angles] == [
        "plain_observation",
        "direct_response",
        "contextual_extension",
    ]
    assert [item.contribution_type for item in result.variants] == [
        "plain_observation",
        "direct_response",
        "contextual_extension",
    ]


def test_changed_connectors_still_lower_repeated_shell_recommendation() -> None:
    packet = _context("The product added more onboarding paperwork.").evidence
    recent = ("fundraising is just sales with extra steps",)
    repeated = GenerationVariant(
        label="repeated",
        text="onboarding amounts to paperwork with a progress bar",
        similarity_risk=0.0,
        angle_id="repeated",
        angle_type="plain_update",
        evidence_ids=("source",),
        confidence="high",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
        contribution_id="repeated",
        contribution_type="plain_observation",
    )
    fresh = repeated.model_copy(
        update={
            "label": "fresh",
            "angle_id": "fresh",
            "contribution_id": "fresh",
            "text": "the extra paperwork makes the onboarding change harder to justify",
        }
    )
    request = GenerationRequest(
        idea_summary="reply",
        content_type=ContentType.REPLY,
        language="en",
        generation_mode=GenerationMode.REPLY,
        evidence_packet=packet,
        recent_posts=recent,
    )
    repeated_report = assess_variant(
        repeated,
        evidence_packet=packet,
        recent_posts=recent,
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )
    fresh_report = assess_variant(
        fresh,
        evidence_packet=packet,
        recent_posts=recent,
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert "RECENT_SYNTAX_REUSE" in {item.code for item in repeated_report.issues}
    assert DraftService._variant_selection_score(fresh, fresh_report, request) > (
        DraftService._variant_selection_score(repeated, repeated_report, request)
    )


@pytest.mark.asyncio
async def test_regeneration_uses_reply_specific_recent_history(tmp_path) -> None:
    """The real regeneration path must request reply-only history for a reply draft."""

    from tests.integration.test_mock_flow import _generated_draft, _runtime

    class RecordingMockProvider(MockLLMProvider):
        def __init__(self) -> None:
            super().__init__()
            self.requests: list[GenerationRequest] = []

        async def generate_structured(self, request: GenerationRequest):
            self.requests.append(request)
            return await super().generate_structured(request)

    session, _, services = _runtime(tmp_path)
    try:
        draft_id = await _generated_draft(session, services)
        draft = services.drafts.get(draft_id)
        # Reproject the existing safe mock draft as a reply so reconciliation and the
        # authoritative artifact metadata remain aligned before regeneration.
        draft.content_type = ContentType.REPLY
        session.flush()
        services.projections.rebuild(draft_id)
        session.commit()

        provider = RecordingMockProvider()
        services.drafts.provider = provider
        requested_types: list[ContentType] = []

        def reply_specific_history(content_type: ContentType) -> tuple[str, ...]:
            requested_types.append(content_type)
            return ("reply history sentinel",)

        services.drafts._recent_corpus_for_content_type = reply_specific_history

        regenerated = await services.drafts.regenerate(
            draft_id,
            feedback="Keep the reply grounded and plain.",
            actor="test",
        )

        assert regenerated.content_type is ContentType.REPLY
        assert requested_types == [ContentType.REPLY]
        assert provider.requests
        assert provider.requests[-1].recent_posts == ("reply history sentinel",)
    finally:
        session.close()


def test_unsupported_nda_inference_is_blocked_by_final_factual_inspection() -> None:
    context = _context("A founder disclosed two confidential partnership details in one tweet.")
    reply = _variant("bro broke two NDAs in one tweet")

    report = assess_variant(
        reply,
        evidence_packet=context.evidence,
        editorial_intent="reply_reaction",
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert not report.passed
    assert {item.code for item in report.issues} & {
        "UNSUPPORTED_FACT",
        "EVIDENCE_INVALID",
        "REACTION_LABEL_WITHOUT_FUNCTION",
    }


@pytest.mark.asyncio
async def test_mock_provider_preserves_application_ranked_family_order() -> None:
    context = _context("Funny update! Parliament removed President Sulyok.")
    service = DraftService.__new__(DraftService)
    service.deterministic_semantic_inspector = _AllGroundedInspector()
    service.semantic_inspector = service.deterministic_semantic_inspector
    service.recent_corpus_limit = 0

    filtered, _, _ = await service._validate_context_semantics(context)
    request = GenerationRequest(
        idea_summary="reply",
        content_type=ContentType.REPLY,
        language="en",
        generation_mode=GenerationMode.REPLY,
        editorial_intent="reply_reaction",
        evidence_packet=filtered.evidence,
        source_coverage=filtered.source_coverage,
        available_evidence=filtered.available_evidence,
        content_mode=filtered.content_mode,
        signal_decision=filtered.signal_decision,
        angle_candidates=filtered.angles,
        literal_plans=filtered.literal_plans,
    )

    result = await MockLLMProvider().generate_structured(request)

    assert [item.contribution_type for item in result.variants] == [
        "plain_observation",
        "direct_response",
        "contextual_extension",
    ]


def test_structurally_fresh_plain_reply_beats_repeated_one_liner() -> None:
    packet = _context("The product added more onboarding paperwork.").evidence
    recent = ("fundraising is just sales with extra steps",)
    repeated = GenerationVariant(
        label="repeated",
        text="onboarding is basically paperwork in a nicer font",
        similarity_risk=0.0,
        angle_id="repeated-plain",
        angle_type="plain_update",
        evidence_ids=("source",),
        confidence="high",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
        contribution_id="repeated-plain",
        contribution_type="plain_observation",
    )
    fresh = repeated.model_copy(
        update={
            "label": "fresh",
            "angle_id": "fresh-plain",
            "contribution_id": "fresh-plain",
            "text": "the extra paperwork makes the onboarding change harder to justify",
        }
    )
    request = GenerationRequest(
        idea_summary="reply",
        content_type=ContentType.REPLY,
        language="en",
        generation_mode=GenerationMode.REPLY,
        evidence_packet=packet,
        recent_posts=recent,
    )
    repeated_report = assess_variant(
        repeated,
        evidence_packet=packet,
        recent_posts=recent,
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )
    fresh_report = assess_variant(
        fresh,
        evidence_packet=packet,
        recent_posts=recent,
        minimum_specificity=0.0,
        minimum_evidence=0.0,
        minimum_naturalness=0.0,
    )

    assert DraftService._variant_selection_score(fresh, fresh_report, request) > (
        DraftService._variant_selection_score(repeated, repeated_report, request)
    )
