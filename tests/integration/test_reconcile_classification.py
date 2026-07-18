"""Regression coverage for independent text/media reconciliation reasons."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.container import MockServices, build_mock_services
from app.database import build_engine
from app.models import Base, Draft, DraftVersion
from app.models.enums import ContentType, DraftStatus, FactCheckStatus

_TEXT = "Synthetic media draft"
_CREATED_AT = datetime(2026, 7, 11, 13, 0, tzinfo=UTC)


def _runtime(tmp_path: Path, draft_id: str) -> tuple[Session, MockServices, Draft]:
    engine = build_engine(f"sqlite:///{(tmp_path / 'reconcile.db').as_posix()}")
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    settings = Settings(
        app_env="test",
        mock_mode=True,
        publish_enabled=True,
        database_url=str(engine.url),
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        config_dir=Path(__file__).resolve().parents[2] / "config",
    )
    services = build_mock_services(session, settings)
    store = services.drafts.artifacts
    directory = store.draft_directory(draft_id, _CREATED_AT)
    no_media_plan = {"type": "none", "required_files": []}
    initial_hash = store.compute_approval_hash(directory, _TEXT, no_media_plan)
    metadata: dict[str, object] = {
        "draft_id": draft_id,
        "status": DraftStatus.NEEDS_REVIEW.value,
        "created_at": _CREATED_AT.isoformat(),
        "updated_at": _CREATED_AT.isoformat(),
        "content_type": ContentType.SHORT_POST.value,
        "language": "en",
        "provider": "mock",
        "model": "mock-model",
        "source_count": 0,
        "character_count": len(_TEXT),
        "weighted_length": len(_TEXT),
        "fact_check_status": FactCheckStatus.NOT_REQUIRED.value,
        "approved_at": None,
        "content_hash": initial_hash,
    }
    store.create_bundle(
        draft_id=draft_id,
        created_at=_CREATED_AT,
        metadata=metadata,
        content=_TEXT,
        media_plan=no_media_plan,
    )
    media_directory = directory / "media"
    media_directory.mkdir()
    (media_directory / "visual.png").write_bytes(b"initial synthetic bytes")
    media_plan = {
        "type": "image",
        "required_files": ["media/visual.png"],
    }
    store.refresh_media_manifest(directory, media_plan)
    content_hash = store.compute_approval_hash(
        directory, _TEXT, media_plan, require_valid_manifest=True
    )
    metadata["content_hash"] = content_hash
    store.write_markdown(directory, metadata=metadata, content=_TEXT)
    store.update_json(directory, "media_plan.json", media_plan)

    draft = Draft(
        id=draft_id,
        content_type=ContentType.SHORT_POST,
        status=DraftStatus.NEEDS_REVIEW,
        language="en",
        provider="mock",
        model="mock-model",
        fact_check_status=FactCheckStatus.NOT_REQUIRED,
        blocking_safety_flags=[],
        media_plan=media_plan,
        artifact_path=str(directory),
        current_content_hash=content_hash,
        current_version_number=1,
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
    )
    version = DraftVersion(
        id=f"{draft_id}-version-1",
        draft_id=draft_id,
        version_number=1,
        content={"parts": [_TEXT], "weighted_lengths": [len(_TEXT)]},
        rendered_text=_TEXT,
        content_hash=content_hash,
        origin="generation",
        provider="mock",
        model="mock-model",
        generation_metadata={"synthetic": True},
        created_at=_CREATED_AT,
    )
    session.add_all((draft, version))
    session.commit()
    return session, services, draft


@pytest.mark.parametrize(
    ("edit_text", "expected_origin", "expected_content_changed"),
    [
        (False, "media_change", False),
        (True, "manual_edit_and_media_change", True),
    ],
)
def test_reconcile_classifies_media_and_combined_changes_once(
    tmp_path: Path,
    edit_text: bool,
    expected_origin: str,
    expected_content_changed: bool,
) -> None:
    session, services, draft = _runtime(tmp_path, f"draft-{'combined' if edit_text else 'media'}")
    try:
        directory = Path(draft.artifact_path)
        (directory / "media" / "visual.png").write_bytes(b"replacement synthetic bytes")
        if edit_text:
            markdown = directory / "draft.md"
            markdown.write_text(
                markdown.read_text("utf-8").replace(_TEXT, "Edited synthetic draft"),
                encoding="utf-8",
                newline="\n",
            )

        assert services.drafts.reconcile(draft.id) is True
        latest = session.scalar(
            select(DraftVersion)
            .where(DraftVersion.draft_id == draft.id)
            .order_by(DraftVersion.version_number.desc())
        )
        assert latest is not None
        assert latest.origin == expected_origin
        metadata = latest.generation_metadata
        assert metadata["synthetic"] is True
        assert metadata["previous_hash"]
        assert metadata["content_changed"] is expected_content_changed
        assert metadata["media_manifest_changed"] is True
        assert metadata["approval_fingerprint_changed"] is True
        assert metadata["approval_invalidation_reason"] == expected_origin
        assert metadata["edit_inspection"] == {"status": "legacy_inspection_unavailable"}
        session.commit()

        assert services.drafts.reconcile(draft.id) is False
        assert (
            len(
                session.scalars(select(DraftVersion).where(DraftVersion.draft_id == draft.id)).all()
            )
            == 2
        )
    finally:
        session.close()
