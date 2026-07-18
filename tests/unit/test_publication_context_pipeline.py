from __future__ import annotations

from pathlib import Path
from types import MethodType

import pytest
from sqlalchemy.orm import Session

from app.database import build_engine
from app.domain.enums import ContentType, IdeaStatus
from app.models import Base, Idea, IdeaSource, SourcePost
from app.providers.mock_provider import MockLLMProvider
from app.schemas.content import GenerationResult, GenerationVariant, MediaPlan, QualityReport
from app.services.configuration import ConfigurationService
from app.services.draft_files import DraftArtifactStore
from app.services.draft_service import DraftService

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _report() -> QualityReport:
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


def _idea(session: Session, *, source_text: str, source_id: str, editorial_intent: str) -> Idea:
    post = SourcePost(
        x_post_id=source_id,
        text=source_text,
        text_hash=(source_id[-1] * 64),
        username="source_account",
        url=f"https://x.com/source_account/status/{source_id}",
        public_metrics={},
        source_labels=["home_timeline"],
        media_metadata={},
    )
    session.add(post)
    session.flush()
    idea = Idea(
        status=IdeaStatus.RANKED,
        title="Single source",
        summary="Write one thought from one source anchor.",
        recommended_format=ContentType.SHORT_POST,
        score=1.0,
        score_components={"single_source_anchor": 1.0},
        explanation={
            "origin": "automatic_discovery_single_source",
            "anchor_x_post_id": source_id,
            "anchor_url": post.url,
            "anchor_author_username": post.username,
            "anchor_has_media": False,
            "editorial_intent": editorial_intent,
        },
        dedupe_key=(source_id[-2:] * 32),
    )
    session.add(idea)
    session.flush()
    session.add(
        IdeaSource(
            idea_id=idea.id,
            source_post_id=post.id,
            source_priority=0.5,
            contribution=1.0,
            position=0,
        )
    )
    session.flush()
    return idea


async def _stub_generation(
    self: DraftService,
    request: object,
    selection: object,
    *,
    text: str,
    factual_claims: tuple[str, ...],
) -> tuple[GenerationResult, GenerationVariant, int, tuple[QualityReport, ...]]:
    del self, selection
    angle = request.angle_candidates[0]
    variant = GenerationVariant(
        label="selected",
        text=text,
        similarity_risk=0.0,
        angle_id=angle.angle_id,
        angle_type=angle.angle_type,
        evidence_ids=angle.evidence_ids,
        source_post_ids=angle.evidence_ids,
        confidence=angle.confidence,
        factual_claims=factual_claims,
        attributed_claims=(),
        uncertainty_markers=(),
    )
    variants = tuple(
        variant.model_copy(
            update={
                "angle_id": candidate.angle_id,
                "angle_type": candidate.angle_type,
                "evidence_ids": candidate.evidence_ids,
            }
        )
        for candidate in request.angle_candidates
    )
    result = GenerationResult(
        idea_summary=request.idea_summary,
        recommended_format=request.content_type,
        variants=variants,
        recommended_variant=0,
        recommendation_reason="test",
        media_plan=MediaPlan(),
    )
    return result, variants[0], 0, tuple(_report() for _ in variants)


@pytest.mark.asyncio
async def test_wording_commentary_is_persisted_as_quote_context(tmp_path: Path) -> None:
    engine = build_engine(f"sqlite:///{(tmp_path / 'quote.db').as_posix()}")
    Base.metadata.create_all(engine)
    profile = ConfigurationService(PROJECT_ROOT / "config").load_profile()
    with Session(engine, expire_on_commit=False) as session:
        idea = _idea(
            session,
            source_text=(
                "Putin: Ukrainian strikes have caused some problems with Russian oil products, "
                "but the situation will gradually improve."
            ),
            source_id="2076668264755732487",
            editorial_intent="quote_reaction",
        )
        service = DraftService(
            session,
            DraftArtifactStore(tmp_path / "drafts"),
            MockLLMProvider(),
            profile=profile,
        )
        text = (
            "“Some problems” is doing a lot of work. Putin acknowledges disruption to Russian "
            "oil products, then immediately moves the sentence toward recovery."
        )

        async def stub(self: DraftService, request: object, selection: object):
            return await _stub_generation(
                self,
                request,
                selection,
                text=text,
                factual_claims=("Putin acknowledges disruption to Russian oil products.",),
            )

        service._generate_quality_checked = MethodType(stub, service)
        draft = await service.generate_from_idea(idea.id)
        version = service.current_version(draft)

        assert draft.media_plan["metadata"]["context_strategy"] == "quote_post"
        assessment = draft.media_plan["metadata"]["publication_context"]
        assert assessment["recommended_format"] == "quote_post"
        assert assessment["source_dependency"] >= 0.65
        assert version.generation_metadata["source_context"]["strategy"] == "quote_post"
        assert draft.fact_check_status.value == "not_required"


@pytest.mark.asyncio
async def test_standalone_single_source_event_requires_fact_check(tmp_path: Path) -> None:
    engine = build_engine(f"sqlite:///{(tmp_path / 'standalone.db').as_posix()}")
    Base.metadata.create_all(engine)
    profile = ConfigurationService(PROJECT_ROOT / "config").load_profile()
    with Session(engine, expire_on_commit=False) as session:
        idea = _idea(
            session,
            source_text=(
                "JUST IN: Trump administration launches campaign to isolate the International "
                "Criminal Court and pressure countries to reject its authority over Americans."
            ),
            source_id="2076671652235006142",
            editorial_intent="report_event",
        )
        service = DraftService(
            session,
            DraftArtifactStore(tmp_path / "drafts"),
            MockLLMProvider(),
            profile=profile,
        )
        text = (
            "The Trump administration launched a campaign to isolate the International "
            "Criminal Court and pressure countries to reject its authority over Americans."
        )

        async def stub(self: DraftService, request: object, selection: object):
            return await _stub_generation(
                self,
                request,
                selection,
                text=text,
                factual_claims=(
                    "The Trump administration launched a campaign to isolate the "
                    "International Criminal Court.",
                ),
            )

        service._generate_quality_checked = MethodType(stub, service)
        draft = await service.generate_from_idea(idea.id)
        version = service.current_version(draft)

        assert draft.media_plan["metadata"]["context_strategy"] == "standalone"
        assessment = draft.media_plan["metadata"]["publication_context"]
        assert assessment["recommended_format"] == "standalone_post"
        assert assessment["standalone_clarity"] >= 0.9
        assert draft.fact_check_status.value == "required"
        assert (
            version.generation_metadata["source_context"]["evidence_status"]
            == "requires_primary_or_reputable_confirmation"
        )
