"""Integration coverage for the canonical final-text inspection path."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.container import build_mock_services
from app.database import build_engine
from app.domain.enums import GenerationMode
from app.models import Base, DraftVersion, Idea, IdeaSource, SourcePost
from app.models.enums import ContentType, IdeaStatus
from app.schemas.content import (
    AngleCandidate,
    DraftSemanticInspection,
    EvidenceItem,
    EvidencePacket,
    GenerationRequest,
    GenerationResult,
    GenerationVariant,
    MediaPlan,
)
from app.services.claim_extraction import DeterministicClaimExtractor
from app.services.configuration import ConfigurationService
from app.services.content_mode import route_content_mode
from app.services.draft_files import DraftArtifactStore
from app.services.draft_service import DraftService, DraftValidationError, NoPostDecision
from app.services.final_text_inspection import FinalTextInspectionService
from app.services.idea_collector import IdeaCollector
from app.services.semantic_classifier import DeterministicSemanticInspector

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ForcedTextProvider:
    name = "forced-test"
    model = "forced-test-v1"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def generate_structured(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        angle = request.angle_candidates[0]
        variant = GenerationVariant(
            label="forced",
            text=self.text,
            similarity_risk=0.0,
            angle_id=angle.angle_id,
            angle_type=angle.angle_type,
            evidence_ids=angle.evidence_ids,
            confidence=angle.confidence,
            factual_claims=(),
            attributed_claims=(),
            uncertainty_markers=(),
        )
        return GenerationResult(
            idea_summary=request.idea_summary,
            recommended_format=request.content_type,
            variants=(variant,),
            recommended_variant=0,
            recommendation_reason="Forced offline regression output",
            media_plan=MediaPlan(),
        )


class CountingClaimExtractor:
    def __init__(self) -> None:
        self.delegate = DeterministicClaimExtractor()
        self.calls: list[str] = []

    async def extract_claims(self, text: str, *, context):
        self.calls.append(text)
        return await self.delegate.extract_claims(text, context=context)


class SentinelSemanticInspector:
    def __init__(self) -> None:
        self.delegate = DeterministicSemanticInspector()
        self.draft_calls = 0

    async def inspect_source(self, packet):
        return await self.delegate.inspect_source(packet)

    async def inspect_contributions(self, packet, contributions, source_map):
        return await self.delegate.inspect_contributions(packet, contributions, source_map)

    async def inspect_draft(self, packet, contribution, text, source_map, *, commentary_mode):
        self.draft_calls += 1
        result = await self.delegate.inspect_draft(
            packet,
            contribution,
            text,
            source_map,
            commentary_mode=commentary_mode,
        )
        return result.model_copy(
            update={
                "evidence_spans": (*result.evidence_spans, "provider-semantic-sentinel"),
            }
        )


def _source_idea(session: Session, text: str) -> Idea:
    post = SourcePost(
        x_post_id="2000000000000000001",
        text=text,
        text_hash="a" * 64,
        username="source",
        url="https://x.com/source/status/2000000000000000001",
        source_labels=["test"],
        media_metadata={},
    )
    session.add(post)
    session.flush()
    idea = Idea(
        status=IdeaStatus.RANKED,
        title="Factual update",
        summary=text,
        recommended_format=ContentType.SHORT_POST,
        score=1.0,
        score_components={"test": 1.0},
        explanation={"editorial_intent": "report_event"},
        dedupe_key="b" * 64,
    )
    session.add(idea)
    session.flush()
    session.add(
        IdeaSource(
            idea_id=idea.id,
            source_post_id=post.id,
            source_priority=1.0,
            contribution=1.0,
            position=0,
        )
    )
    session.flush()
    return idea


@pytest.mark.asyncio
async def test_relative_clause_unsupported_fact_blocks_initial_generation(tmp_path: Path) -> None:
    engine = build_engine(f"sqlite:///{(tmp_path / 'initial.db').as_posix()}")
    Base.metadata.create_all(engine)
    profile = ConfigurationService(PROJECT_ROOT / "config").load_profile()
    with Session(engine, expire_on_commit=False) as session:
        idea = _source_idea(session, "Alice bought the company Acme.")
        provider = ForcedTextProvider("Alice bought the company Acme, which Bob founded.")
        service = DraftService(
            session,
            DraftArtifactStore(tmp_path / "drafts"),
            provider,
            profile=profile,
        )

        with pytest.raises(DraftValidationError):
            await service.generate_from_idea(idea.id)

        assert provider.calls >= 1
        assert session.scalars(select(DraftVersion)).all() == []


@pytest.mark.asyncio
async def test_relative_clause_unsupported_fact_blocks_regeneration(tmp_path: Path) -> None:
    engine = build_engine(f"sqlite:///{(tmp_path / 'regen.db').as_posix()}")
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    settings = Settings(
        app_env="test",
        mock_mode=True,
        publish_enabled=True,
        database_url=str(engine.url),
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        config_dir=PROJECT_ROOT / "config",
    )
    services = build_mock_services(session, settings)
    try:
        idea_id = IdeaCollector(
            session,
            source_configuration=services.configuration.sources,
        ).collect_mock()[0]
        draft = await services.drafts.generate_from_idea(idea_id)
        assert draft is not None
        session.commit()
        prior_versions = session.scalars(
            select(DraftVersion).where(DraftVersion.draft_id == draft.id)
        ).all()
        provider = ForcedTextProvider("Alice bought the company Acme, which Bob founded.")
        services.drafts.provider = provider

        with pytest.raises(NoPostDecision):
            await services.drafts.regenerate(
                draft.id,
                feedback="Keep the factual update concise",
            )

        versions = session.scalars(
            select(DraftVersion)
            .where(DraftVersion.draft_id == draft.id)
            .order_by(DraftVersion.version_number)
        ).all()
        assert len(versions) == len(prior_versions)
        assert provider.calls >= 1
    finally:
        session.close()


@pytest.mark.asyncio
async def test_regeneration_runs_live_claim_extractor_and_persists_merged_semantics(
    tmp_path: Path,
) -> None:
    engine = build_engine(f"sqlite:///{(tmp_path / 'providers.db').as_posix()}")
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    settings = Settings(
        app_env="test",
        mock_mode=True,
        publish_enabled=True,
        database_url=str(engine.url),
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        config_dir=PROJECT_ROOT / "config",
    )
    services = build_mock_services(session, settings)
    try:
        claim_extractor = CountingClaimExtractor()
        semantic_inspector = SentinelSemanticInspector()
        services.drafts.claim_extractor = claim_extractor
        services.drafts.semantic_inspector = semantic_inspector
        idea_id = IdeaCollector(
            session,
            source_configuration=services.configuration.sources,
        ).collect_mock()[0]
        draft = await services.drafts.generate_from_idea(idea_id)
        assert draft is not None
        session.commit()
        initial_claim_extraction_calls = len(claim_extractor.calls)
        assert initial_claim_extraction_calls >= 1

        regenerated = await services.drafts.regenerate(
            draft.id,
            feedback="Preserve the evidence-bound checkpoint",
            actor="test",
        )
        version = services.drafts.current_version(regenerated)
        structured = DraftSemanticInspection.model_validate(
            version.generation_metadata["structured_semantic_inspection"]
        )

        assert len(claim_extractor.calls) > initial_claim_extraction_calls
        assert semantic_inspector.draft_calls >= 2
        assert "provider-semantic-sentinel" in structured.evidence_spans
        assert version.generation_metadata["inspection_service"] == "canonical_final_text_v1"
    finally:
        session.close()


@pytest.mark.asyncio
async def test_manual_edit_and_live_reinspection_share_canonical_service(tmp_path: Path) -> None:
    engine = build_engine(f"sqlite:///{(tmp_path / 'edit.db').as_posix()}")
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    settings = Settings(
        app_env="test",
        mock_mode=True,
        publish_enabled=True,
        database_url=str(engine.url),
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        config_dir=PROJECT_ROOT / "config",
    )
    services = build_mock_services(session, settings)
    try:
        idea_id = IdeaCollector(
            session,
            source_configuration=services.configuration.sources,
        ).collect_mock()[0]
        draft = await services.drafts.generate_from_idea(idea_id)
        assert draft is not None
        session.commit()
        services.drafts.claim_extractor = CountingClaimExtractor()
        services.drafts.semantic_inspector = SentinelSemanticInspector()

        markdown = Path(draft.artifact_path) / "draft.md"
        markdown.write_text(
            markdown.read_text(encoding="utf-8") + "\nA bounded wording cleanup.\n",
            encoding="utf-8",
        )
        assert services.drafts.reconcile(draft.id, actor="test") is True
        pending = services.drafts.current_version(services.drafts.get(draft.id))
        assert pending.generation_metadata["inspection_service"] == "canonical_final_text_v1"
        assert (
            pending.generation_metadata["factual_inspection"]["semantic_reinspection_required"]
            is True
        )

        reinspected = await services.drafts.reinspect_current_version(draft.id, actor="test")
        completed = services.drafts.current_version(reinspected)
        assert completed.generation_metadata["inspection_service"] == "canonical_final_text_v1"
        assert (
            completed.generation_metadata["factual_inspection"]["semantic_reinspection_required"]
            is False
        )
    finally:
        session.close()


@pytest.mark.asyncio
async def test_canonical_final_inspection_accepts_supported_range_endpoint() -> None:
    deterministic_claims = DeterministicClaimExtractor()
    deterministic_semantics = DeterministicSemanticInspector()
    service = FinalTextInspectionService(
        deterministic_claim_extractor=deterministic_claims,
        claim_extractor=deterministic_claims,
        deterministic_semantic_inspector=deterministic_semantics,
        semantic_inspector=deterministic_semantics,
    )
    evidence = EvidencePacket(
        mode=GenerationMode.SOURCE_POST,
        topic="range endpoint",
        language="en",
        items=(
            EvidenceItem(
                evidence_id="e1",
                source_type="x_post",
                text="Revenue increased from 5 million to 10 million.",
            ),
        ),
    )
    contribution = AngleCandidate(
        angle_id="a1",
        angle_type="plain_update",
        thesis="Revenue increased from 5 million to 10 million.",
        evidence_ids=("e1",),
        why_interesting="direct factual update",
        confidence="high",
        contribution_type="direct_update",
    )
    final_variant = GenerationVariant(
        label="endpoint",
        text="Revenue increased to 10 million.",
        similarity_risk=0.0,
        angle_id="a1",
        angle_type="plain_update",
        evidence_ids=("e1",),
        confidence="high",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
    )
    result = await service.inspect(
        final_variant.text,
        variant=final_variant,
        packet=evidence,
        content_mode=route_content_mode(
            editorial_intent="report_event",
            generation_mode=GenerationMode.SOURCE_POST,
            requested_format=ContentType.SHORT_POST,
        ),
        contribution=contribution,
    )
    assert result.factual_report.status == "pass"
    assert result.factual_report.approval_eligible is True
    assert result.structured_semantic_inspection.contribution_preserved is True
