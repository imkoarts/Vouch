from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.domain.enums import ContentType, GenerationMode
from app.schemas.content import (
    ContributionInspection,
    EvidenceItem,
    EvidencePacket,
    GenerationResult,
    GenerationVariant,
    MediaPlan,
    UntrustedSourceData,
)
from app.services.draft_service import DraftService, RetryDisposition, retry_disposition
from app.services.generation_pipeline import (
    build_generation_context,
    extract_source_coverage,
)
from app.services.semantic_classifier import (
    DeterministicSemanticInspector,
    derive_source_semantic_map,
)
from app.services.semantic_review import assess_added_value, review_semantics

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HUNGARY_SOURCE = (
    "BREAKING: Hungary\u2019s parliament votes to remove President Tamás Sulyok, in the latest "
    "move to dismantle Viktor Orbán\u2019s remaining influence."
)
HUNGARY_BAD = (
    "Hungary\u2019s parliament voting to remove President Tamás Sulyok makes the change concrete: "
    "a named officeholder is being removed, rather than Viktor Orbán\u2019s influence being "
    "discussed in the abstract. The presidency is now part of the political shift itself. "
    "That is a more tangible development than a general argument about influence."
)


def _packet(*texts: str) -> EvidencePacket:
    return EvidencePacket(
        mode=GenerationMode.SOURCE_POST,
        topic="semantic regression",
        language="en",
        items=tuple(
            EvidenceItem(evidence_id=f"e{index}", source_type="x_post", text=text)
            for index, text in enumerate(texts)
        ),
    )


def _variant(index: int = 0) -> GenerationVariant:
    return GenerationVariant(
        label=f"v{index}",
        text="Alice signed the bill on Tuesday.",
        similarity_risk=0.0,
        angle_id=f"angle-{index}",
        angle_type="plain_update",
        evidence_ids=("e0",),
        confidence="high",
        factual_claims=("Alice signed the bill on Tuesday.",),
        attributed_claims=(),
        uncertainty_markers=(),
    )


def test_hungary_bad_output_fails_independent_semantic_review() -> None:
    packet = _packet(HUNGARY_SOURCE)
    report = review_semantics(
        HUNGARY_BAD,
        packet=packet,
        coverage=extract_source_coverage(packet),
        editorial_intent="comment_on_source",
    )

    assert report.decision == "no_post_needed"
    assert {
        "ANALYTICAL_PARAPHRASE_LOOP",
        "ABSTRACT_RESTATEMENT",
        "SAME_POINT_MULTIPLE_WAYS",
        "INVENTED_COMPARISON_TARGET",
        "DEPERSONALIZING_RENAME",
        "SOURCE_FACT_OVEREXPLANATION",
        "LEXICAL_NOVELTY_ONLY",
        "SEMANTIC_SOURCE_ECHO",
        "SENTENCE_WITHOUT_JOB",
    } <= set(report.issue_codes)


def test_hungary_commentary_plans_zero_candidates() -> None:
    context = build_generation_context(
        idea_summary=HUNGARY_SOURCE,
        idea_explanation={"editorial_intent": "comment_on_source"},
        language="en",
        sources=(UntrustedSourceData(source_id="hungary", content=HUNGARY_SOURCE),),
        content_type=ContentType.SHORT_POST,
    )

    assert context.status == "no_post_needed"
    assert context.reason_code == "SOURCE_ALREADY_COVERS_AVAILABLE_POINT"
    assert context.angles == ()
    assert context.literal_plans == ()


@pytest.mark.asyncio
async def test_operator_requested_quote_reaction_uses_deterministic_preflight() -> None:
    context = build_generation_context(
        idea_summary="A model added a low-verbosity setting.",
        idea_explanation={
            "origin": "manual_url",
            "editorial_intent": "quote_reaction",
            "operator_requested_generation": True,
        },
        language="en",
        sources=(
            UntrustedSourceData(
                source_id="manual-source",
                content="A model added a low-verbosity setting.",
                url="https://x.com/example/status/1234567890",
            ),
        ),
        content_type=ContentType.QUOTE_COMMENTARY,
    )

    class RejectingProviderInspector:
        async def inspect_source(self, packet: EvidencePacket):
            return derive_source_semantic_map(packet)

        async def inspect_contributions(self, packet, contributions, source_map):
            deterministic = await DeterministicSemanticInspector().inspect_contributions(
                packet, contributions, source_map
            )
            return tuple(
                ContributionInspection.model_validate(item.model_dump() | {"grounded": False})
                for item in deterministic
            )

        async def inspect_draft(self, *args, **kwargs):
            raise AssertionError("draft inspection is not part of preflight")

    service = DraftService.__new__(DraftService)
    service.deterministic_semantic_inspector = DeterministicSemanticInspector()
    service.semantic_inspector = RejectingProviderInspector()

    safe_fallback, _, inspections = await service._validate_context_semantics(context)
    allowed, _, _ = await service._validate_context_semantics(
        context,
        deterministic_only=True,
    )

    assert safe_fallback.status == "ok"
    assert [item.contribution_type for item in safe_fallback.angles] == ["plain_observation"]
    assert any(item.grounded for item in inspections)
    assert (
        "quote-plain-observation:PROVIDER_DISAGREEMENT_SAFE_QUOTE_FALLBACK"
        in safe_fallback.rejected_candidate_reasons
    )
    assert allowed.status == "ok"
    assert {item.contribution_type for item in allowed.angles} >= {
        "plain_observation",
        "dry_humor",
    }


@pytest.mark.asyncio
async def test_report_event_keeps_grounded_direct_transmission_candidate() -> None:
    context = build_generation_context(
        idea_summary="BREAKING: Company weighs charging traders $100,000 a month.",
        idea_explanation={"editorial_intent": "report_event"},
        language="en",
        sources=(
            UntrustedSourceData(
                source_id="event-source",
                content="BREAKING: Company weighs charging traders $100,000 a month.",
            ),
        ),
        content_type=ContentType.SHORT_POST,
    )
    service = DraftService.__new__(DraftService)
    service.deterministic_semantic_inspector = DeterministicSemanticInspector()
    service.semantic_inspector = service.deterministic_semantic_inspector

    validated, _, inspections = await service._validate_context_semantics(
        context, deterministic_only=True
    )

    assert context.status == "ok"
    assert validated.status == "ok"
    assert validated.angles
    assert all(item.grounded for item in inspections)


def test_report_event_allows_literal_source_echo() -> None:
    packet = _packet("Alice signed the bill on Tuesday.")
    report = review_semantics(
        "Alice signed the bill on Tuesday.",
        packet=packet,
        coverage=extract_source_coverage(packet),
        editorial_intent="report_event",
    )

    assert report.decision == "accept"
    assert not set(report.issue_codes) & {
        "NO_GROUNDED_CONTRIBUTION",
        "SEMANTIC_SOURCE_ECHO",
        "LEXICAL_NOVELTY_ONLY",
    }


def test_abstract_synonym_evasion_is_detected() -> None:
    packet = _packet("The company cut the fee from 2% to 1%.")
    report = review_semantics(
        "The fee reduction turns pricing into part of the product shift itself. That is a more "
        "tangible development than a general discussion about affordability.",
        packet=packet,
        coverage=extract_source_coverage(packet),
        editorial_intent="comment_on_source",
    )

    assert {
        "ABSTRACT_RESTATEMENT",
        "INVENTED_COMPARISON_TARGET",
        "ANALYTICAL_PARAPHRASE_LOOP",
    } <= set(report.issue_codes)


def test_named_actor_depersonalization_is_detected() -> None:
    packet = _packet("Alice signed the bill.")
    report = review_semantics(
        "A named officeholder has now made the policy concrete.",
        packet=packet,
        coverage=extract_source_coverage(packet),
        editorial_intent="comment_on_source",
    )

    assert {"DEPERSONALIZING_RENAME", "ABSTRACT_RESTATEMENT"} <= set(report.issue_codes)


def test_supported_cross_evidence_relation_is_valid_commentary() -> None:
    packet = _packet(
        "The export includes scores but excludes test cases.",
        "The target model requires the test cases to rerun the evaluation.",
    )
    text = "The scores can be exported, but the evaluation cannot be rerun without the test cases."
    coverage = extract_source_coverage(packet)

    value = assess_added_value(
        text,
        packet=packet,
        coverage=coverage,
        editorial_intent="comment_on_source",
    )
    report = review_semantics(
        text,
        packet=packet,
        coverage=coverage,
        editorial_intent="comment_on_source",
    )

    assert value.decision == "pass"
    assert value.supported_new_relations == (text,)
    assert report.decision == "accept"
    assert report.issue_codes == ()


def test_generation_result_accepts_one_or_two_and_models_no_post() -> None:
    one = GenerationResult(
        status="ok",
        idea_summary="one",
        recommended_format=ContentType.SHORT_POST,
        variants=(_variant(0),),
        recommended_variant=0,
        media_plan=MediaPlan(),
    )
    two = one.model_copy(update={"variants": (_variant(0), _variant(1)), "recommended_variant": 1})
    no_post = GenerationResult(
        status="no_post_needed",
        reason_code="NO_GROUNDED_CONTRIBUTION",
        idea_summary="none",
        media_plan=MediaPlan(),
    )

    assert len(one.variants) == 1
    assert len(two.variants) == 2
    assert no_post.variants == ()
    assert no_post.recommended_variant is None
    with pytest.raises(ValidationError):
        GenerationResult(
            status="ok",
            idea_summary="invalid",
            recommended_format=ContentType.SHORT_POST,
            media_plan=MediaPlan(),
        )


@pytest.mark.parametrize(
    ("codes", "expected"),
    [
        (("NO_POST_NEEDED",), RetryDisposition.NO_POST),
        (("UNSUPPORTED_RELATION",), RetryDisposition.INSUFFICIENT_CONTEXT),
        (("UNSUPPORTED_ANGLE",), RetryDisposition.REPLAN),
        (("FORMAT_MISMATCH",), RetryDisposition.LENGTH_REPAIR),
        (("CORPORATE_VOICE",), RetryDisposition.STYLE_REVISION),
        ((), RetryDisposition.ACCEPT),
    ],
)
def test_retry_disposition_is_explicit(codes: tuple[str, ...], expected: RetryDisposition) -> None:
    assert retry_disposition(codes) is expected


@pytest.mark.asyncio
async def test_zero_candidate_editorial_decision_skips_provider(tmp_path: Path) -> None:
    from hashlib import sha256

    from sqlalchemy.orm import Session

    from app.database import build_engine
    from app.domain.enums import IdeaStatus
    from app.models import Base, Idea, IdeaSource, SourcePost
    from app.services.configuration import ConfigurationService
    from app.services.draft_files import DraftArtifactStore
    from app.services.draft_service import DraftService

    class NeverCalledProvider:
        name = "never-called"
        model = "none"
        called = False

        async def generate_structured(self, request: object) -> GenerationResult:
            del request
            self.called = True
            raise AssertionError(
                "provider must not be called when planning returns zero candidates"
            )

    engine = build_engine(f"sqlite:///{(tmp_path / 'no-post.db').as_posix()}")
    Base.metadata.create_all(engine)
    profile = ConfigurationService(PROJECT_ROOT / "config").load_profile()
    with Session(engine, expire_on_commit=False) as session:
        source = SourcePost(
            x_post_id="1001",
            text=HUNGARY_SOURCE,
            text_hash=sha256(HUNGARY_SOURCE.encode()).hexdigest(),
            username="source",
            source_labels=["recent_search"],
            public_metrics={},
            media_metadata={},
            raw_json={},
        )
        session.add(source)
        session.flush()
        idea = Idea(
            status=IdeaStatus.RANKED,
            title="Hungary commentary",
            summary=HUNGARY_SOURCE,
            recommended_format=ContentType.SHORT_POST,
            score=1.0,
            score_components={},
            explanation={"editorial_intent": "comment_on_source"},
            dedupe_key="a" * 64,
        )
        session.add(idea)
        session.flush()
        session.add(
            IdeaSource(
                idea_id=idea.id,
                source_post_id=source.id,
                source_priority=1.0,
                contribution=1.0,
                position=0,
            )
        )
        session.flush()
        provider = NeverCalledProvider()
        service = DraftService(
            session,
            DraftArtifactStore(tmp_path / "drafts"),
            provider,  # type: ignore[arg-type]
            profile=profile,
        )

        draft = await service.generate_from_idea(idea.id)

        assert draft is None
        assert provider.called is False
        assert idea.status is IdeaStatus.ARCHIVED
        decision = idea.explanation["editorial_decision"]
        assert decision["status"] == "no_post_needed"
        assert decision["reason_code"] == "SOURCE_ALREADY_COVERS_AVAILABLE_POINT"
        assert decision["candidate_count"] == 0


def test_source_coverage_records_only_visible_questions_and_recommendations() -> None:
    packet = _packet(
        "Teams should export test cases with scores. Can the evaluation be rerun without them?"
    )

    coverage = extract_source_coverage(packet)

    assert coverage.recommendations == ("Teams should export test cases with scores.",)
    assert coverage.genuinely_unresolved_questions == ("Can the evaluation be rerun without them?",)
