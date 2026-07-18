from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.database import build_engine
from app.domain.enums import ContentType, DraftStatus, FactCheckStatus
from app.models import Base, Draft, DraftVersion
from app.schemas.content import (
    CanonicalClaim,
    ContributionInspection,
    GenerationVariant,
    ReactionFunctionInspection,
    UntrustedSourceData,
)
from app.services.draft_service import DraftService
from app.services.editorial_quality import inspect_reply_naturalness
from app.services.factual_inspection import inspect_factual_integrity
from app.services.generation_pipeline import build_generation_context
from app.services.semantic_classifier import (
    DeterministicSemanticInspector,
    derive_source_semantic_map,
    inspect_reaction_function,
    merge_contribution_inspections,
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


def _variant(text: str) -> GenerationVariant:
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
        contribution_type="plain_observation",
    )


@pytest.mark.asyncio
async def test_hyphenated_and_spaced_low_verbosity_keep_same_humor_family() -> None:
    service = DraftService.__new__(DraftService)
    service.deterministic_semantic_inspector = DeterministicSemanticInspector()
    service.semantic_inspector = service.deterministic_semantic_inspector
    service.recent_corpus_limit = 0

    hyphenated, _, _ = await service._validate_context_semantics(
        _context("A model added a low-verbosity setting.")
    )
    spaced, _, _ = await service._validate_context_semantics(
        _context("A model added a low verbosity setting.")
    )

    assert "dry_humor" in {item.contribution_type for item in hyphenated.angles}
    assert {item.contribution_type for item in hyphenated.angles} == {
        item.contribution_type for item in spaced.angles
    }


def test_anti_yap_reaction_is_grounded_for_hyphenated_source() -> None:
    packet = _context("A model added a low-verbosity setting.").evidence

    inspection = inspect_reaction_function(packet, "finally an official anti-yap switch")

    assert inspection.grounded
    assert inspection.reaction_type in {"wordplay", "dry_reframe", "literalization"}


@pytest.mark.asyncio
async def test_explicit_nda_source_keeps_humor_family() -> None:
    service = DraftService.__new__(DraftService)
    service.deterministic_semantic_inspector = DeterministicSemanticInspector()
    service.semantic_inspector = service.deterministic_semantic_inspector
    service.recent_corpus_limit = 0

    filtered, _, _ = await service._validate_context_semantics(
        _context("A founder broke two NDAs in one tweet.")
    )

    assert "dry_humor" in {item.contribution_type for item in filtered.angles}


def test_explicit_nda_register_shift_is_grounded_but_inference_is_not() -> None:
    supported = _context("A founder broke two NDAs in one tweet.").evidence
    unsupported = _context(
        "A founder disclosed two confidential partnership details in one tweet."
    ).evidence
    reply = "bro broke two NDAs in one tweet"

    supported_inspection = inspect_reaction_function(supported, reply)
    unsupported_inspection = inspect_reaction_function(unsupported, reply)

    assert supported_inspection.grounded
    assert supported_inspection.source_specific
    assert not unsupported_inspection.grounded
    assert unsupported_inspection.unsupported_detail


def test_unsupported_nda_accusation_is_blocked_by_factual_adjudication() -> None:
    unsupported = _context(
        "A founder disclosed two confidential partnership details in one tweet."
    ).evidence
    supported = _context("A founder broke two NDAs in one tweet.").evidence
    variant = GenerationVariant(
        label="nda",
        text="bro broke two NDAs in one tweet",
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

    unsupported_report = inspect_factual_integrity(variant, packet=unsupported)
    supported_report = inspect_factual_integrity(variant, packet=supported)

    assert unsupported_report.status == "block"
    assert not unsupported_report.approval_eligible
    assert any(item.support == "unsupported" for item in unsupported_report.claims)
    assert supported_report.status == "pass"
    assert supported_report.approval_eligible


class _AllGroundedInspector:
    def __init__(self, *, strong_humor: bool = False) -> None:
        self.strong_humor = strong_humor

    async def inspect_source(self, packet):
        return derive_source_semantic_map(packet)

    async def inspect_contributions(self, packet, contributions, source_map):
        del packet, source_map
        results = []
        for candidate in contributions:
            relation = candidate.supporting_relations[0]
            claim = CanonicalClaim(
                claim_id=candidate.angle_id,
                exact_text=candidate.thesis,
                kind="interpretation",
                subject=relation.subject,
                predicate=relation.predicate,
                object=relation.object,
                polarity="positive",
                modality="asserted",
            )
            humor = candidate.contribution_type == "dry_humor"
            results.append(
                ContributionInspection(
                    contribution_id=candidate.angle_id,
                    canonical_proposition=claim,
                    already_covered=False,
                    requires_new_assumption=False,
                    unsupported_adjacent_complications=(),
                    grounded=True,
                    reaction_function_present=humor and self.strong_humor,
                    source_specificity_present=True,
                    plain_paraphrase_only=False,
                    reaction_inspection=(
                        ReactionFunctionInspection(
                            reaction_type="wordplay",
                            source_specific=True,
                            source_echo=False,
                            unsupported_detail=False,
                            generic_template_only=False,
                            grounded=True,
                        )
                        if humor and self.strong_humor
                        else None
                    ),
                )
            )
        return tuple(results)

    async def inspect_draft(self, *args, **kwargs):
        raise AssertionError("not used")


def test_live_reaction_affordance_survives_merge_without_weakening_grounding() -> None:
    claim = CanonicalClaim(
        claim_id="reply-dry-humor",
        exact_text="dry humor proposal",
        kind="interpretation",
        subject="source",
        predicate="react",
        object="detail",
        polarity="positive",
        modality="asserted",
    )
    deterministic = ContributionInspection(
        contribution_id="reply-dry-humor",
        canonical_proposition=claim,
        already_covered=False,
        requires_new_assumption=False,
        grounded=True,
        reaction_function_present=False,
        source_specificity_present=True,
    )
    provider = deterministic.model_copy(
        update={
            "reaction_function_present": True,
            "source_specificity_present": True,
        }
    )

    merged = merge_contribution_inspections((deterministic,), (provider,))[0]

    assert merged.grounded
    assert merged.reaction_function_present


def test_live_reaction_affordance_cannot_override_provider_grounding_rejection() -> None:
    claim = CanonicalClaim(
        claim_id="reply-dry-humor",
        exact_text="dry humor proposal",
        kind="interpretation",
        subject="source",
        predicate="react",
        object="detail",
        polarity="positive",
        modality="asserted",
    )
    deterministic = ContributionInspection(
        contribution_id="reply-dry-humor",
        canonical_proposition=claim,
        already_covered=False,
        requires_new_assumption=False,
        grounded=True,
        reaction_function_present=False,
        source_specificity_present=True,
    )
    provider = deterministic.model_copy(
        update={
            "grounded": False,
            "reaction_function_present": True,
        }
    )

    merged = merge_contribution_inspections((deterministic,), (provider,))[0]

    assert not merged.grounded
    assert merged.reaction_function_present


@pytest.mark.asyncio
async def test_all_four_grounded_families_are_ranked_before_three_slot_cap() -> None:
    context = _context("Funny update! Parliament removed President Sulyok.")
    service = DraftService.__new__(DraftService)
    service.deterministic_semantic_inspector = _AllGroundedInspector()
    service.semantic_inspector = service.deterministic_semantic_inspector
    service.recent_corpus_limit = 0

    filtered, _, _ = await service._validate_context_semantics(context)

    assert [item.contribution_type for item in filtered.angles] == [
        "plain_observation",
        "direct_response",
        "contextual_extension",
    ]
    assert "reply-dry-humor:LOWER_REPLY_FAMILY_PRIORITY" in (filtered.rejected_candidate_reasons)


@pytest.mark.asyncio
async def test_strong_source_specific_humor_can_outrank_context_when_justified() -> None:
    context = _context("A model added a low-verbosity setting.")
    service = DraftService.__new__(DraftService)
    service.deterministic_semantic_inspector = _AllGroundedInspector(strong_humor=True)
    service.semantic_inspector = service.deterministic_semantic_inspector
    service.recent_corpus_limit = 0

    filtered, _, _ = await service._validate_context_semantics(context)

    assert "dry_humor" in {item.contribution_type for item in filtered.angles}


class _RejectHumorInspector(_AllGroundedInspector):
    async def inspect_contributions(self, packet, contributions, source_map):
        values = await super().inspect_contributions(packet, contributions, source_map)
        return tuple(
            item.model_copy(update={"grounded": False})
            if item.contribution_id == "reply-dry-humor"
            else item
            for item in values
        )


@pytest.mark.asyncio
async def test_semantic_rejection_is_not_reported_as_slot_priority_loss() -> None:
    context = _context("A model added a low-verbosity setting.")
    service = DraftService.__new__(DraftService)
    service.deterministic_semantic_inspector = _RejectHumorInspector()
    service.semantic_inspector = service.deterministic_semantic_inspector
    service.recent_corpus_limit = 0

    filtered, _, _ = await service._validate_context_semantics(context)

    assert "reply-dry-humor:SEMANTIC_CONTRIBUTION_REJECTED" in (filtered.rejected_candidate_reasons)
    assert "reply-dry-humor:LOWER_REPLY_FAMILY_PRIORITY" not in (
        filtered.rejected_candidate_reasons
    )


@pytest.mark.parametrize(
    "candidate",
    (
        "onboarding is basically paperwork in a nicer font",
        "onboarding amounts to paperwork with a progress bar",
        "at this point onboarding is paperwork wearing product copy",
    ),
)
def test_reductive_identity_variants_reuse_same_recent_shell(candidate: str) -> None:
    naturalness = inspect_reply_naturalness(
        _variant(candidate),
        recent_posts=("fundraising is just sales with extra steps",),
    )

    assert naturalness.recent_syntax_reuse


def test_nothing_communicates_maps_to_nothing_says_shell() -> None:
    naturalness = inspect_reply_naturalness(
        _variant("nothing communicates simplicity quite like six more screens"),
        recent_posts=("nothing says product simplicity like six new screens",),
    )

    assert naturalness.recent_syntax_reuse


def test_isolated_structural_shell_is_not_recent_reuse() -> None:
    naturalness = inspect_reply_naturalness(
        _variant("onboarding is basically paperwork in a nicer font"), recent_posts=()
    )

    assert not naturalness.recent_syntax_reuse


def _persist_draft(
    session: Session,
    *,
    draft_id: str,
    content_type: ContentType,
    text: str,
) -> None:
    draft = Draft(
        id=draft_id,
        content_type=content_type,
        status=DraftStatus.NEEDS_REVIEW,
        language="en",
        provider="mock",
        model="mock",
        fact_check_status=FactCheckStatus.NOT_REQUIRED,
        blocking_safety_flags=[],
        media_plan={"type": "none", "required_files": []},
        artifact_path=f"drafts/{draft_id}",
        current_content_hash=f"hash-{draft_id}",
        current_version_number=1,
    )
    version = DraftVersion(
        id=f"{draft_id}-v1",
        draft_id=draft_id,
        version_number=1,
        content={"parts": [text], "weighted_lengths": [len(text)]},
        rendered_text=text,
        content_hash=f"hash-{draft_id}",
        origin="generation",
        provider="mock",
        model="mock",
        generation_metadata={},
    )
    session.add_all((draft, version))
    session.flush()


def test_reply_recent_corpus_excludes_standalone_and_includes_reply(tmp_path: Path) -> None:
    engine = build_engine(f"sqlite:///{(tmp_path / 'reply-history.db').as_posix()}")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        _persist_draft(
            session,
            draft_id="standalone",
            content_type=ContentType.SHORT_POST,
            text="fundraising is just sales with extra steps",
        )
        _persist_draft(
            session,
            draft_id="long-form",
            content_type=ContentType.LONG_POST,
            text="onboarding amounts to paperwork with a progress bar",
        )
        _persist_draft(
            session,
            draft_id="reply",
            content_type=ContentType.REPLY,
            text="support is basically triage in a nicer font",
        )
        service = DraftService.__new__(DraftService)
        service.session = session
        service.recent_corpus_limit = 12

        corpus = service._recent_reply_corpus()

    engine.dispose()
    assert corpus == ("support is basically triage in a nicer font",)


def test_content_type_history_router_is_reply_specific() -> None:
    service = DraftService.__new__(DraftService)
    service._recent_reply_corpus = lambda: ("reply-only history",)
    service._recent_corpus = lambda: ("broad standalone history",)

    assert service._recent_corpus_for_content_type(ContentType.REPLY) == ("reply-only history",)
    assert service._recent_corpus_for_content_type(ContentType.SHORT_POST) == (
        "broad standalone history",
    )
