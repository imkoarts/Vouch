"""Append-only audit events with mandatory metadata sanitization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy.orm import Session

from app.domain.security import redact_secrets
from app.models import AuditEvent


class AuditService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def record(
        self,
        action: str,
        *,
        entity_type: str,
        entity_id: str | None,
        actor: str,
        old_status: str | None = None,
        new_status: str | None = None,
        content_hash: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AuditEvent:
        safe = redact_secrets(dict(metadata or {}))
        event = AuditEvent(
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            actor=actor,
            old_status=old_status,
            new_status=new_status,
            content_hash=content_hash,
            event_metadata=safe if isinstance(safe, dict) else {},
        )
        self.session.add(event)
        self.session.flush()
        return event
