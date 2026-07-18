"""Observability history rows reject ORM mutation and deletion."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.domain.enums import ApiDirection, LLMRunStatus
from app.models import ApiUsage, AuditEvent, Base, LLMRun

ObservedRow = AuditEvent | ApiUsage | LLMRun


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as database_session:
        yield database_session
    engine.dispose()


def _audit_event() -> AuditEvent:
    return AuditEvent(action="created", entity_type="draft", actor="local-user")


def _api_usage() -> ApiUsage:
    return ApiUsage(
        provider="mock",
        operation="read",
        direction=ApiDirection.READ,
    )


def _llm_run() -> LLMRun:
    return LLMRun(
        provider="mock",
        model="mock-model",
        operation="generate",
        status=LLMRunStatus.SUCCEEDED,
    )


def _change_audit_event(row: ObservedRow) -> None:
    assert isinstance(row, AuditEvent)
    row.action = "changed"


def _change_api_usage(row: ObservedRow) -> None:
    assert isinstance(row, ApiUsage)
    row.units = 2


def _change_llm_run(row: ObservedRow) -> None:
    assert isinstance(row, LLMRun)
    row.retry_count = 1


@pytest.mark.parametrize(
    ("factory", "mutate"),
    [
        (_audit_event, _change_audit_event),
        (_api_usage, _change_api_usage),
        (_llm_run, _change_llm_run),
    ],
    ids=["audit-event", "api-usage", "llm-run"],
)
def test_observability_rows_cannot_be_updated_or_deleted_via_orm(
    session: Session,
    factory: Callable[[], ObservedRow],
    mutate: Callable[[ObservedRow], None],
) -> None:
    row = factory()
    session.add(row)
    session.commit()
    row_id = row.id

    mutate(row)
    with pytest.raises(RuntimeError, match="rows are append-only"):
        session.commit()
    session.rollback()

    persisted = session.get_one(type(row), row_id)
    session.delete(persisted)
    with pytest.raises(RuntimeError, match="rows are append-only"):
        session.commit()
    session.rollback()
