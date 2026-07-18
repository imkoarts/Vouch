"""Deterministic offline judge flow with export-only publication evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from anyio import Path as AsyncPath
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.container import build_mock_services
from app.database import build_engine
from app.models import Base, DraftApproval, DraftVersion
from app.services.idea_collector import IdeaCollector
from app.x_api.mock import MockXClient


@dataclass(frozen=True)
class JudgeDemoReport:
    product: str
    mode: str
    draft_id: str
    generated_status: str
    source_visible: bool
    preview_only: bool
    confirmation_phrase: str
    approved_hash: str
    edit_invalidated_approval: bool
    version_after_edit: int
    final_status: str
    blocking_flags: tuple[str, ...]
    remote_write_calls: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def run_judge_demo(output_dir: Path, *, config_dir: Path) -> JudgeDemoReport:
    """Run a fresh synthetic flow; never call publication or any live boundary."""

    root = _prepare_root(output_dir)
    settings = Settings(
        _env_file=None,
        app_env="test",
        mock_mode=True,
        publish_enabled=False,
        database_url=f"sqlite:///{(root / 'judge.db').as_posix()}",
        data_dir=root / "data",
        drafts_dir=root / "drafts",
        logs_dir=root / "logs",
        config_dir=config_dir,
    )
    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)
    try:
        with Session(engine, expire_on_commit=False) as session:
            services = build_mock_services(session, settings)
            idea_ids = IdeaCollector(
                session, source_configuration=services.configuration.sources
            ).collect_mock()
            draft = await services.drafts.generate_from_idea(idea_ids[0], actor="judge_demo")
            if draft is None:  # pragma: no cover - deterministic mock invariant
                raise RuntimeError("Judge fixture did not produce a draft")
            session.commit()
            generated_status = draft.status.value
            source_visible = bool(draft.idea and draft.idea.source_links)
            approval = services.approvals.approve(draft.id, actor="judge_demo")
            preview = await services.publishing.preview(draft.id)

            draft_path = AsyncPath(draft.artifact_path) / "draft.md"
            original = await draft_path.read_text(encoding="utf-8")
            await draft_path.write_text(
                original + "\nThis unsupported edit claims guaranteed results.\n",
                encoding="utf-8",
            )
            changed = services.drafts.reconcile(draft.id, actor="judge_demo")
            refreshed = services.drafts.get(draft.id)
            active = services.approvals.active_approval(refreshed)
            version = session.scalar(
                select(DraftVersion)
                .where(DraftVersion.draft_id == draft.id)
                .order_by(DraftVersion.version_number.desc())
            )
            revoked = session.scalars(
                select(DraftApproval).where(DraftApproval.draft_id == draft.id)
            ).all()
            report = JudgeDemoReport(
                product="Vouch",
                mode="judge/offline",
                draft_id=draft.id,
                generated_status=generated_status,
                source_visible=source_visible,
                preview_only=True,
                confirmation_phrase=preview.confirmation_phrase,
                approved_hash=approval.content_hash,
                edit_invalidated_approval=bool(changed and active is None and len(revoked) >= 2),
                version_after_edit=version.version_number if version is not None else 0,
                final_status=refreshed.status.value,
                blocking_flags=tuple(refreshed.blocking_safety_flags),
                remote_write_calls=(
                    len(services.writer.write_calls)
                    if isinstance(services.writer, MockXClient)
                    else 0
                ),
            )
            session.commit()
            return report
    finally:
        engine.dispose()


def _prepare_root(output_dir: Path) -> Path:
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root
