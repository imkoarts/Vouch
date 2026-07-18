from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from sqlalchemy.orm import Session

from app.database import build_engine
from app.domain.content import publishable_content
from app.models import (
    Base,
    Draft,
    DraftApproval,
    DraftVersion,
    Idea,
    IdeaSource,
    Publication,
    Topic,
)
from app.models.enums import (
    ApprovalAction,
    ContentType,
    DraftStatus,
    FactCheckStatus,
    PublicationKind,
    PublicationStatus,
    TopicKind,
)
from app.services.artifact_projection import (
    ArtifactProjectionService,
    RecoveryAction,
    RecoveryIssueKind,
)
from app.services.draft_files import DraftArtifactStore

_CREATED_AT = datetime(2026, 7, 11, 9, 0, tzinfo=UTC)
_MEDIA_PLAN = {"type": "none", "required_files": []}


def _runtime(tmp_path: Path) -> tuple[Session, DraftArtifactStore]:
    engine = build_engine(f"sqlite:///{(tmp_path / 'projection.db').as_posix()}")
    Base.metadata.create_all(engine)
    return (
        Session(engine, expire_on_commit=False),
        DraftArtifactStore(tmp_path / "drafts"),
    )


def _metadata(draft: Draft, text: str, *, source_count: int = 0) -> dict[str, object]:
    return {
        "draft_id": draft.id,
        "status": draft.status.value,
        "created_at": draft.created_at.isoformat(),
        "updated_at": draft.updated_at.isoformat(),
        "content_type": draft.content_type.value,
        "language": draft.language,
        "provider": draft.provider,
        "model": draft.model,
        "source_count": source_count,
        "character_count": len(text),
        "fact_check_status": draft.fact_check_status.value,
        "approved_at": None,
        "content_hash": draft.current_content_hash,
    }


def _bare_draft(
    session: Session,
    store: DraftArtifactStore,
    *,
    draft_id: str,
    text: str,
    create_bundle: bool,
) -> Draft:
    directory = store.draft_directory(draft_id, _CREATED_AT)
    content_hash = store.compute_approval_hash(directory, text, _MEDIA_PLAN)
    draft = Draft(
        id=draft_id,
        content_type=ContentType.SHORT_POST,
        status=DraftStatus.NEEDS_REVIEW,
        language="en",
        provider="mock",
        model="mock-model",
        fact_check_status=FactCheckStatus.NOT_REQUIRED,
        blocking_safety_flags=[],
        media_plan=_MEDIA_PLAN,
        artifact_path=str(directory),
        current_content_hash=content_hash,
        current_version_number=1,
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
    )
    version = DraftVersion(
        id=f"{draft_id}-v1",
        draft_id=draft_id,
        version_number=1,
        content={"parts": [text], "weighted_lengths": [len(text)]},
        rendered_text=text,
        content_hash=content_hash,
        origin="generation",
        provider="mock",
        model="mock-model",
        generation_metadata={"synthetic": True},
        created_at=_CREATED_AT,
    )
    session.add_all((draft, version))
    session.flush()
    if create_bundle:
        store.create_bundle(
            draft_id=draft.id,
            created_at=draft.created_at,
            metadata=_metadata(draft, text),
            content=text,
            media_plan=draft.media_plan,
        )
    return draft


def test_projection_rebuilds_every_db_derived_artifact(tmp_path: Path) -> None:
    session, store = _runtime(tmp_path)
    try:
        idea = Idea(
            id="idea-projection",
            title="Synthetic topic",
            summary="A local synthetic idea",
            recommended_format=ContentType.SHORT_POST,
            score=1.0,
        )
        first_topic = Topic(
            id="topic-one",
            name="Synthetic one",
            kind=TopicKind.EVERGREEN,
            description="First synthetic source",
        )
        second_topic = Topic(
            id="topic-two",
            name="Synthetic two",
            kind=TopicKind.EVERGREEN,
            description="Second synthetic source",
        )
        session.add_all((idea, first_topic, second_topic))
        session.flush()
        session.add_all(
            (
                IdeaSource(
                    id="link-one",
                    idea_id=idea.id,
                    topic_id=first_topic.id,
                    position=0,
                    source_priority=0.8,
                    contribution=0.6,
                ),
                IdeaSource(
                    id="link-two",
                    idea_id=idea.id,
                    topic_id=second_topic.id,
                    position=1,
                    source_priority=0.7,
                    contribution=0.4,
                ),
            )
        )
        directory = store.draft_directory("draft-history", _CREATED_AT)
        first_text = "First synthetic draft"
        current_text = "Current synthetic draft"
        first_hash = store.compute_approval_hash(directory, first_text, _MEDIA_PLAN)
        current_hash = store.compute_approval_hash(directory, current_text, _MEDIA_PLAN)
        draft = Draft(
            id="draft-history",
            idea_id=idea.id,
            content_type=ContentType.SHORT_POST,
            status=DraftStatus.APPROVED,
            language="en",
            provider="mock",
            model="mock-model-v2",
            fact_check_status=FactCheckStatus.NOT_REQUIRED,
            blocking_safety_flags=[],
            media_plan=_MEDIA_PLAN,
            artifact_path=str(directory),
            current_content_hash=current_hash,
            current_version_number=2,
            created_at=_CREATED_AT,
            updated_at=_CREATED_AT + timedelta(minutes=5),
        )
        first_version = DraftVersion(
            id="version-one",
            draft_id=draft.id,
            version_number=1,
            content={"parts": [first_text], "weighted_lengths": [21]},
            rendered_text=first_text,
            content_hash=first_hash,
            origin="generation",
            provider="mock",
            model="mock-model-v1",
            generation_metadata={"selected_variant": 0},
            created_at=_CREATED_AT,
        )
        current_version = DraftVersion(
            id="version-two",
            draft_id=draft.id,
            version_number=2,
            content={"parts": [current_text], "weighted_lengths": [23]},
            rendered_text=current_text,
            content_hash=current_hash,
            origin="regeneration",
            provider="mock",
            model="mock-model-v2",
            feedback="make it clearer",
            generation_metadata={"previous_version": 1},
            created_at=_CREATED_AT + timedelta(minutes=1),
        )
        session.add_all((draft, first_version, current_version))
        session.flush()

        old_approval = DraftApproval(
            id="approval-old",
            draft_id=draft.id,
            draft_version_id=first_version.id,
            content_hash=first_hash,
            action=ApprovalAction.APPROVED,
            actor="operator",
            created_at=_CREATED_AT + timedelta(minutes=2),
        )
        session.add(old_approval)
        session.flush()
        session.add(
            DraftApproval(
                id="approval-revocation",
                draft_id=draft.id,
                draft_version_id=first_version.id,
                content_hash=first_hash,
                action=ApprovalAction.REVOKED,
                actor="operator",
                supersedes_id=old_approval.id,
                reason="superseded by edit",
                created_at=_CREATED_AT + timedelta(minutes=3),
            )
        )
        active_at = _CREATED_AT + timedelta(minutes=4)
        session.add(
            DraftApproval(
                id="approval-active",
                draft_id=draft.id,
                draft_version_id=current_version.id,
                content_hash=current_hash,
                action=ApprovalAction.APPROVED,
                actor="operator",
                created_at=active_at,
            )
        )
        session.add_all(
            (
                Publication(
                    id="publication-failed",
                    draft_id=draft.id,
                    draft_version_id=first_version.id,
                    content_hash=first_hash,
                    kind=PublicationKind.POST,
                    status=PublicationStatus.FAILED,
                    thread_position=0,
                    request_payload_hash="a" * 64,
                    estimated_cost_usd=None,
                    attempt_count=1,
                    error_details="api_key=synthetic-secret-value",
                    created_at=_CREATED_AT + timedelta(minutes=3),
                    updated_at=_CREATED_AT + timedelta(minutes=3),
                ),
                Publication(
                    id="publication-ambiguous",
                    draft_id=draft.id,
                    draft_version_id=current_version.id,
                    content_hash=current_hash,
                    kind=PublicationKind.POST,
                    status=PublicationStatus.AMBIGUOUS,
                    thread_position=0,
                    request_payload_hash="b" * 64,
                    response_metadata={"secret_token": "synthetic-secret-value"},
                    estimated_cost_usd=Decimal("0.01"),
                    attempt_count=1,
                    error_details="synthetic timeout",
                    created_at=_CREATED_AT + timedelta(minutes=5),
                    updated_at=_CREATED_AT + timedelta(minutes=5),
                ),
            )
        )
        session.flush()
        store.create_bundle(
            draft_id=draft.id,
            created_at=draft.created_at,
            metadata=_metadata(draft, current_text, source_count=0),
            content=publishable_content((current_text,), draft.content_type),
            media_plan=draft.media_plan,
        )
        session.commit()

        result = ArtifactProjectionService(session, store).project(draft.id)

        metadata = json.loads((directory / "metadata.json").read_text("utf-8"))
        generations = json.loads((directory / "generations.json").read_text("utf-8"))
        publication = json.loads((directory / "publication.json").read_text("utf-8"))
        markdown = (directory / "draft.md").read_text("utf-8")
        review = (directory / "review.md").read_text("utf-8")

        assert result.fingerprint == metadata["projection"]["fingerprint"]
        assert metadata["status"] == "approved"
        assert metadata["content_hash"] == current_hash
        assert metadata["source_count"] == 2
        assert metadata["approved_at"] == active_at.isoformat()
        assert metadata["active_approval_id"] == "approval-active"
        assert [item["origin"] for item in generations["versions"]] == [
            "generation",
            "regeneration",
        ]
        assert generations["versions"][1]["parts"] == [current_text]
        assert publication["status"] == "approved"
        assert publication["has_failed_outcome"] is True
        assert publication["has_ambiguous_outcome"] is True
        assert publication["status_counts"] == {"ambiguous": 1, "failed": 1}
        assert publication["records"][0]["error_details"] == "api_key=***REDACTED***"
        assert publication["records"][0]["estimated_cost_usd"] is None
        assert publication["records"][1]["response_metadata"]["secret_token"] == ("***REDACTED***")
        assert "source_count: 2" in markdown
        assert current_text in markdown
        assert first_text not in markdown
        assert "approval-active" in review
        assert active_at.isoformat() in review
    finally:
        session.close()


def test_recovery_repairs_only_safe_projection_divergence(tmp_path: Path) -> None:
    session, store = _runtime(tmp_path)
    try:
        stale = _bare_draft(
            session,
            store,
            draft_id="draft-stale",
            text="Stable body",
            create_bundle=True,
        )
        missing = _bare_draft(
            session,
            store,
            draft_id="draft-missing",
            text="Recoverable body",
            create_bundle=False,
        )
        divergent = _bare_draft(
            session,
            store,
            draft_id="draft-divergent",
            text="Database body",
            create_bundle=True,
        )
        quarantined = _bare_draft(
            session,
            store,
            draft_id="draft-quarantined",
            text="Quarantined body",
            create_bundle=True,
        )
        session.commit()
        projection = ArtifactProjectionService(session, store)
        projection.project(divergent.id)
        projection.project(quarantined.id)

        stale_metadata_path = Path(stale.artifact_path) / "metadata.json"
        stale_metadata = json.loads(stale_metadata_path.read_text("utf-8"))
        stale_metadata["status"] = "generated"
        stale_metadata_path.write_text(json.dumps(stale_metadata), encoding="utf-8", newline="\n")
        divergent_markdown = Path(divergent.artifact_path) / "draft.md"
        divergent_markdown.write_text(
            divergent_markdown.read_text("utf-8").replace(
                "Database body", "Uncommitted manual edit"
            ),
            encoding="utf-8",
            newline="\n",
        )

        quarantine_path = (
            store.quarantine_root
            / "20260711T100000.000000Z"
            / _CREATED_AT.date().isoformat()
            / quarantined.id
        )
        quarantine_path.parent.mkdir(parents=True)
        os.replace(Path(quarantined.artifact_path), quarantine_path)
        orphan = store.root / _CREATED_AT.date().isoformat() / "orphan-bundle"
        orphan.mkdir(parents=True)
        (orphan / "draft.md").write_text("synthetic orphan\n", encoding="utf-8")

        initial = projection.scan_recovery()
        issue_pairs = {(issue.kind, issue.draft_id) for issue in initial.issues}
        assert (RecoveryIssueKind.STALE_PROJECTION, stale.id) in issue_pairs
        assert (RecoveryIssueKind.MISSING_BUNDLE, missing.id) in issue_pairs
        assert (RecoveryIssueKind.CONTENT_DIVERGENCE, divergent.id) in issue_pairs
        assert (
            RecoveryIssueKind.QUARANTINE_DIVERGENCE,
            quarantined.id,
        ) in issue_pairs
        assert (RecoveryIssueKind.ORPHAN_BUNDLE, None) in issue_pairs

        report = projection.reconcile_startup()

        assert {
            (action.action, action.draft_id, action.succeeded) for action in report.actions
        } >= {
            (RecoveryAction.PROJECT, stale.id, True),
            (RecoveryAction.RECREATE_BUNDLE, missing.id, True),
        }
        remaining_pairs = {(issue.kind, issue.draft_id) for issue in report.issues}
        assert (RecoveryIssueKind.STALE_PROJECTION, stale.id) not in remaining_pairs
        assert (RecoveryIssueKind.MISSING_BUNDLE, missing.id) not in remaining_pairs
        assert (RecoveryIssueKind.CONTENT_DIVERGENCE, divergent.id) in remaining_pairs
        assert (
            RecoveryIssueKind.QUARANTINE_DIVERGENCE,
            quarantined.id,
        ) in remaining_pairs
        assert (RecoveryIssueKind.ORPHAN_BUNDLE, None) in remaining_pairs
        assert "Uncommitted manual edit" in divergent_markdown.read_text("utf-8")
        assert quarantine_path.is_dir()
        assert orphan.is_dir()
    finally:
        session.close()
