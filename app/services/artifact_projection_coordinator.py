"""Commit-boundary coordination for DB-authoritative draft projections."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session

from app.models import Draft, DraftApproval, DraftVersion, Publication
from app.services.artifact_projection import ArtifactProjectionService
from app.services.audit import AuditService
from app.services.draft_files import DraftArtifactStore

_COORDINATOR_KEY = "xbot_artifact_projection_coordinator"
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProjectionFailure:
    """Sanitized diagnostic retained after a non-fatal projection failure."""

    draft_id: str
    error_type: str


class ArtifactProjectionCoordinator:
    """Project every changed draft only after its database commit succeeds.

    SQLAlchemy service methods intentionally remain transaction-neutral.  This
    instance-specific listener observes changes made through the composition-root
    session, remembers affected draft IDs, and projects them from a fresh read
    session in ``after_commit``.  Filesystem errors are therefore unable to undo a
    committed publication outcome; they produce a sanitized audit event and can be
    repaired with the explicit rebuild command.
    """

    def __init__(self, session: Session, artifacts: DraftArtifactStore) -> None:
        bind = session.get_bind()
        if not isinstance(bind, (Engine, Connection)):
            raise TypeError("Artifact projection requires a SQLAlchemy engine or connection")
        self.session = session
        self.artifacts = artifacts
        self.bind = bind
        self.pending_draft_ids: set[str] = set()
        self.failures: list[ProjectionFailure] = []

    @classmethod
    def install(
        cls, session: Session, artifacts: DraftArtifactStore
    ) -> ArtifactProjectionCoordinator:
        """Install once for a session and return the active coordinator."""

        existing = session.info.get(_COORDINATOR_KEY)
        if isinstance(existing, cls):
            if existing.artifacts.root != artifacts.root:
                raise RuntimeError("A session cannot project drafts into two artifact roots")
            return existing
        coordinator = cls(session, artifacts)
        session.info[_COORDINATOR_KEY] = coordinator
        event.listen(session, "after_flush", coordinator._collect_after_flush)
        event.listen(session, "after_commit", coordinator._project_after_commit)
        event.listen(session, "after_rollback", coordinator._clear_after_rollback)
        return coordinator

    @staticmethod
    def _draft_id(instance: Any) -> str | None:
        if isinstance(instance, Draft):
            value = instance.id
        elif isinstance(instance, (DraftVersion, DraftApproval, Publication)):
            value = instance.draft_id
        else:
            return None
        return value if isinstance(value, str) and value else None

    def _collect_after_flush(self, session: Session, flush_context: Any) -> None:
        del flush_context
        for instance in (*session.new, *session.dirty, *session.deleted):
            draft_id = self._draft_id(instance)
            if draft_id is not None:
                self.pending_draft_ids.add(draft_id)

    def _clear_after_rollback(self, session: Session) -> None:
        del session
        self.pending_draft_ids.clear()

    def _record_failure(self, draft_id: str, error: Exception) -> None:
        error_type = type(error).__name__
        self.failures.append(ProjectionFailure(draft_id=draft_id, error_type=error_type))
        _LOGGER.error(
            "Draft artifact projection failed after commit for %s (%s)",
            draft_id,
            error_type,
        )
        try:
            with Session(bind=self.bind, expire_on_commit=False) as audit_session:
                AuditService(audit_session).record(
                    "artifact_projection_failed",
                    entity_type="draft",
                    entity_id=draft_id,
                    actor="system",
                    metadata={
                        "error_type": error_type,
                        "recovery": f"drafts rebuild-artifacts {draft_id}",
                    },
                )
                audit_session.commit()
        except Exception as audit_error:  # pragma: no cover - defensive last resort
            _LOGGER.error(
                "Could not persist artifact projection failure audit (%s)",
                type(audit_error).__name__,
            )

    def _project_after_commit(self, session: Session) -> None:
        del session
        draft_ids = tuple(sorted(self.pending_draft_ids))
        self.pending_draft_ids.clear()
        if not draft_ids:
            return
        with Session(bind=self.bind, expire_on_commit=False) as projection_session:
            projection = ArtifactProjectionService(projection_session, self.artifacts)
            for draft_id in draft_ids:
                try:
                    projection.rebuild(draft_id)
                except Exception as error:  # projection failure must never undo commit
                    self._record_failure(draft_id, error)


__all__ = ["ArtifactProjectionCoordinator", "ProjectionFailure"]
