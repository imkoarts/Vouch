"""Acceptance coverage for the complete offline moderation vertical slice."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from anyio import Path as AsyncPath
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.container import MockServices, build_mock_services
from app.database import build_engine
from app.models import (
    ApiUsage,
    AuditEvent,
    Base,
    DraftApproval,
    DraftVersion,
    Publication,
)
from app.models.enums import ContentType, DraftStatus, PublicationStatus
from app.services.approval_service import ApprovalGateError
from app.services.artifact_projection import ArtifactProjectionService
from app.services.cost_service import CostEstimateRequiredError
from app.services.idea_collector import IdeaCollector
from app.services.publishing_service import (
    AmbiguousClaimError,
    FaultStage,
    PublicationFaultInjector,
    PublicationGateError,
)
from app.x_api.mock import MockXClient


def _runtime(
    tmp_path: Path,
    *,
    publish_enabled: bool = True,
    writer: MockXClient | None = None,
    publication_fault_injector: PublicationFaultInjector | None = None,
) -> tuple[Session, Settings, MockServices]:
    engine = build_engine(f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    settings = Settings(
        _env_file=None,
        app_env="test",
        mock_mode=True,
        publish_enabled=publish_enabled,
        database_url=str(engine.url),
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        config_dir=Path(__file__).resolve().parents[2] / "config",
    )
    return (
        session,
        settings,
        build_mock_services(
            session,
            settings,
            writer=writer,
            publication_fault_injector=publication_fault_injector,
        ),
    )


async def _generated_draft(session: Session, services: MockServices) -> str:
    idea_ids = IdeaCollector(
        session, source_configuration=services.configuration.sources
    ).collect_mock()
    draft = await services.drafts.generate_from_idea(idea_ids[0])
    session.commit()
    return draft.id


@pytest.mark.asyncio
async def test_full_mock_flow_records_only_a_mock_publication(tmp_path: Path) -> None:
    session, _, services = _runtime(tmp_path)
    try:
        draft_id = await _generated_draft(session, services)
        draft = services.drafts.get(draft_id)
        assert draft.blocking_safety_flags == []
        generations = await AsyncPath(draft.artifact_path, "generations.json").read_text(
            encoding="utf-8"
        )
        assert 1 <= generations.count('"label"') <= 3
        approval = services.approvals.approve(draft_id)
        preview = await services.publishing.preview(draft_id)
        assert preview.estimated_cost_usd == Decimal("0.015")
        records = await services.publishing.publish(
            draft_id,
            confirmation_phrase=preview.confirmation_phrase,
        )
        assert approval.content_hash == preview.content_hash
        assert records and all(record.status is PublicationStatus.SUCCEEDED for record in records)
        assert services.drafts.get(draft_id).status is DraftStatus.PUBLISHED
        assert all(record.response_metadata == {"mock": True} for record in records)
        assert all(record.estimated_cost_usd == Decimal("0.015") for record in records)
        usage = session.scalars(select(ApiUsage)).all()
        assert usage and all(item.estimated_cost_usd == Decimal("0.015") for item in usage)
    finally:
        session.close()


@pytest.mark.asyncio
async def test_publication_requires_approval_and_exact_phrase(tmp_path: Path) -> None:
    session, _, services = _runtime(tmp_path)
    try:
        draft_id = await _generated_draft(session, services)
        with pytest.raises(ApprovalGateError):
            await services.publishing.publish(draft_id, confirmation_phrase="PUBLISH anything")
        services.approvals.approve(draft_id)
        with pytest.raises(PublicationGateError):
            await services.publishing.publish(draft_id, confirmation_phrase="yes")
        assert services.writer.write_calls == []
    finally:
        session.close()


@pytest.mark.asyncio
async def test_manual_edit_revokes_hash_scoped_approval(tmp_path: Path) -> None:
    session, _, services = _runtime(tmp_path)
    try:
        draft_id = await _generated_draft(session, services)
        services.approvals.approve(draft_id)
        draft = services.drafts.get(draft_id)
        path = AsyncPath(draft.artifact_path, "draft.md")
        original = await path.read_text(encoding="utf-8")
        await path.write_text(original + "\nManual clarification.\n", encoding="utf-8")
        assert services.drafts.reconcile(draft_id)
        draft = services.drafts.get(draft_id)
        assert draft.status is DraftStatus.NEEDS_REVIEW
        assert services.approvals.active_approval(draft) is None
        assert (
            session.scalar(
                select(DraftVersion)
                .where(DraftVersion.draft_id == draft_id)
                .order_by(DraftVersion.version_number.desc())
            ).version_number
            == 2
        )
        assert (
            len(
                session.scalars(
                    select(DraftApproval).where(DraftApproval.draft_id == draft_id)
                ).all()
            )
            == 2
        )
        version_count = len(
            session.scalars(select(DraftVersion).where(DraftVersion.draft_id == draft_id)).all()
        )
        assert services.drafts.reconcile(draft_id) is False
        assert (
            len(
                session.scalars(select(DraftVersion).where(DraftVersion.draft_id == draft_id)).all()
            )
            == version_count
        )
    finally:
        session.close()


@pytest.mark.asyncio
async def test_ambiguous_claim_is_never_repeated_automatically(tmp_path: Path) -> None:
    writer = MockXClient(ambiguous_positions={0})
    session, _, services = _runtime(tmp_path, writer=writer)
    try:
        draft_id = await _generated_draft(session, services)
        services.approvals.approve(draft_id)
        phrase = (await services.publishing.preview(draft_id)).confirmation_phrase
        with pytest.raises(AmbiguousClaimError):
            await services.publishing.publish(draft_id, confirmation_phrase=phrase)
        assert len(writer.write_calls) == 1
        with pytest.raises((ApprovalGateError, PublicationGateError, AmbiguousClaimError)):
            await services.publishing.publish(draft_id, confirmation_phrase=phrase)
        assert len(writer.write_calls) == 1
        claims = session.scalars(select(Publication).where(Publication.draft_id == draft_id)).all()
        assert len(claims) == 1
        assert claims[0].status is PublicationStatus.AMBIGUOUS
    finally:
        session.close()


@pytest.mark.asyncio
async def test_publish_enabled_gate_still_applies_in_mock_mode(tmp_path: Path) -> None:
    session, _, services = _runtime(tmp_path, publish_enabled=False)
    try:
        draft_id = await _generated_draft(session, services)
        services.approvals.approve(draft_id)
        phrase = (await services.publishing.preview(draft_id)).confirmation_phrase
        with pytest.raises(PublicationGateError, match="PUBLISH_ENABLED"):
            await services.publishing.publish(draft_id, confirmation_phrase=phrase)
        assert services.writer.write_calls == []
    finally:
        session.close()


@pytest.mark.asyncio
async def test_projection_failure_cannot_repeat_or_undo_a_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _, services = _runtime(tmp_path)
    original_rebuild = ArtifactProjectionService.rebuild
    try:
        draft_id = await _generated_draft(session, services)
        services.approvals.approve(draft_id)
        phrase = (await services.publishing.preview(draft_id)).confirmation_phrase

        def fail_projection(
            projection: ArtifactProjectionService, selected_draft_id: str
        ) -> object:
            del projection, selected_draft_id
            raise OSError("synthetic projection failure")

        monkeypatch.setattr(ArtifactProjectionService, "rebuild", fail_projection)
        records = await services.publishing.publish(draft_id, confirmation_phrase=phrase)

        assert len(services.writer.write_calls) == 1
        assert records[0].status is PublicationStatus.SUCCEEDED
        assert services.drafts.get(draft_id).status is DraftStatus.PUBLISHED
        assert services.projection_coordinator.failures
        assert (
            session.scalar(
                select(AuditEvent).where(
                    AuditEvent.action == "artifact_projection_failed",
                    AuditEvent.entity_id == draft_id,
                )
            )
            is not None
        )

        monkeypatch.setattr(ArtifactProjectionService, "rebuild", original_rebuild)
        services.projections.rebuild(draft_id)
        assert len(services.writer.write_calls) == 1
        projection = await AsyncPath(
            services.drafts.get(draft_id).artifact_path, "publication.json"
        ).read_text(encoding="utf-8")
        assert '"status": "succeeded"' in projection
        assert '"mock": true' in projection
    finally:
        session.close()


@pytest.mark.asyncio
async def test_live_style_budget_with_unknown_cost_blocks_before_writer(
    tmp_path: Path,
) -> None:
    session, _, services = _runtime(tmp_path)
    try:
        draft_id = await _generated_draft(session, services)
        services.approvals.approve(draft_id)
        services.costs.allow_unknown_estimates = False
        services.costs.limits["x_write"] = Decimal("1.00")
        services.costs.estimates.pop("x_write", None)
        preview = await services.publishing.preview(draft_id)
        assert preview.estimated_cost_usd is None

        with pytest.raises(CostEstimateRequiredError, match="unconfigured"):
            await services.publishing.publish(
                draft_id,
                confirmation_phrase=preview.confirmation_phrase,
            )

        assert services.writer.write_calls == []
        assert (
            session.scalars(select(Publication).where(Publication.draft_id == draft_id)).all() == []
        )
    finally:
        session.close()


class _SimulatedProcessCrash(RuntimeError):
    pass


@pytest.mark.parametrize(
    ("crash_stage", "expected_writes", "expected_status"),
    [
        ("after_claim_commit", 0, PublicationStatus.IN_FLIGHT),
        ("after_remote_acceptance", 1, PublicationStatus.IN_FLIGHT),
        ("after_finalize_commit", 1, PublicationStatus.SUCCEEDED),
    ],
)
@pytest.mark.asyncio
async def test_publication_crash_boundaries_never_duplicate_remote_write(
    tmp_path: Path,
    crash_stage: FaultStage,
    expected_writes: int,
    expected_status: PublicationStatus,
) -> None:
    def inject(stage: FaultStage, position: int) -> None:
        if stage == crash_stage and position == 0:
            raise _SimulatedProcessCrash(stage)

    writer = MockXClient()
    session, _, services = _runtime(
        tmp_path,
        writer=writer,
        publication_fault_injector=inject,
    )
    try:
        draft_id = await _generated_draft(session, services)
        services.approvals.approve(draft_id)
        phrase = (await services.publishing.preview(draft_id)).confirmation_phrase

        with pytest.raises(_SimulatedProcessCrash, match=crash_stage):
            await services.publishing.publish(
                draft_id,
                confirmation_phrase=phrase,
            )

        claims = session.scalars(select(Publication).where(Publication.draft_id == draft_id)).all()
        assert len(claims) == 1
        assert claims[0].status is expected_status
        assert len(writer.write_calls) == expected_writes

        services.publishing.fault_injector = None
        with pytest.raises((ApprovalGateError, PublicationGateError, AmbiguousClaimError)):
            await services.publishing.publish(
                draft_id,
                confirmation_phrase=phrase,
            )
        assert len(writer.write_calls) == expected_writes
    finally:
        session.close()


@pytest.mark.asyncio
async def test_regeneration_reuses_untrusted_sources_and_revokes_approval(
    tmp_path: Path,
) -> None:
    session, _, services = _runtime(tmp_path)
    try:
        draft_id = await _generated_draft(session, services)
        services.approvals.approve(draft_id)
        draft = await services.drafts.regenerate(
            draft_id, feedback="Make the checkpoint more concrete"
        )
        session.commit()
        assert draft.status is DraftStatus.NEEDS_REVIEW
        assert draft.blocking_safety_flags == []
        assert services.approvals.active_approval(draft) is None
        versions = session.scalars(
            select(DraftVersion)
            .where(DraftVersion.draft_id == draft_id)
            .order_by(DraftVersion.version_number)
        ).all()
        assert [version.version_number for version in versions] == [1, 2]
        history_text = await AsyncPath(draft.artifact_path, "generations.json").read_text(
            encoding="utf-8"
        )
        history = json.loads(history_text)
        assert [item["version_number"] for item in history["versions"]] == [1, 2]
        assert history["current_version_number"] == 2
        assert "Make the checkpoint more concrete" in history_text
    finally:
        session.close()


@pytest.mark.asyncio
async def test_mock_thread_has_two_parts_and_publishes_as_a_reply_chain(
    tmp_path: Path,
) -> None:
    session, _, services = _runtime(tmp_path)
    try:
        from app.services.manual_generation import create_manual_topic_idea

        idea = create_manual_topic_idea(
            session,
            topic_text=(
                "Explain a two-step manual review workflow: first record the evidence, then "
                "publish only after approval."
            ),
        )
        idea.recommended_format = ContentType.THREAD
        session.flush()
        draft = await services.drafts.generate_from_idea(idea.id)
        session.commit()
        version = services.drafts.current_version(draft)
        assert len(version.content["parts"]) == 2
        markdown = await AsyncPath(draft.artifact_path, "draft.md").read_text(encoding="utf-8")
        assert "X_CONTENT_BOT_THREAD_PART" in markdown
        services.approvals.approve(draft.id)
        preview = await services.publishing.preview(draft.id)
        records = await services.publishing.publish(
            draft.id, confirmation_phrase=preview.confirmation_phrase
        )
        assert len(records) == 2
        assert records[1].parent_post_id == records[0].x_post_id
    finally:
        session.close()


@pytest.mark.asyncio
async def test_generation_rejects_provider_format_different_from_requested_format(
    tmp_path: Path,
) -> None:
    from app.schemas.content import GenerationResult, GenerationVariant, MediaPlan
    from app.services.draft_service import DraftValidationError
    from app.services.manual_generation import create_manual_topic_idea

    class MismatchedProvider:
        name = "mock"
        model = "synthetic"

        async def generate_structured(self, _request):
            text = "A short reply instead of the requested thread."
            variant = GenerationVariant(
                label="synthetic",
                text=text,
                character_count=len(text),
                similarity_risk=0.0,
                angle_id="synthetic-angle",
                angle_type="plain_update",
                evidence_ids=("legacy-evidence",),
                confidence="medium",
                factual_claims=(),
                attributed_claims=(),
                uncertainty_markers=(),
            )
            return GenerationResult(
                idea_summary="synthetic",
                recommended_format=ContentType.SHORT_POST,
                variants=(
                    variant,
                    variant.model_copy(
                        update={"angle_id": "synthetic-angle-2", "angle_type": "skeptical_caveat"}
                    ),
                    variant.model_copy(
                        update={
                            "angle_id": "synthetic-angle-3",
                            "angle_type": "practical_implication",
                        }
                    ),
                ),
                recommended_variant=0,
                recommendation_reason="synthetic mismatch",
                media_plan=MediaPlan(),
            )

    session, _, services = _runtime(tmp_path)
    try:
        services.drafts.provider = MismatchedProvider()
        idea = create_manual_topic_idea(
            session,
            topic_text="A thread is required",
            content_type=ContentType.THREAD,
        )
        with pytest.raises(DraftValidationError, match="different from the requested format"):
            await services.drafts.generate_from_idea(idea.id)
        assert session.scalars(select(DraftVersion)).all() == []
    finally:
        session.close()
