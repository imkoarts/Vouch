"""Central, fail-closed draft lifecycle state machine."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from app.domain.enums import DraftStatus


class DraftStateMachineError(ValueError):
    """Base class for lifecycle errors safe to expose in logs and the UI."""


class InvalidDraftStatusError(DraftStateMachineError):
    """Raised when persisted or user-supplied state is not recognized."""

    def __init__(self) -> None:
        expected = ", ".join(status.value for status in DraftStatus)
        super().__init__(f"Unknown draft status. Expected one of: {expected}.")


class InvalidDraftTransitionError(DraftStateMachineError):
    """Raised when a transition is outside the explicit allowlist."""

    def __init__(self, current: DraftStatus, target: DraftStatus) -> None:
        self.current = current
        self.target = target
        super().__init__(
            f"Draft transition from '{current.value}' to '{target.value}' is not allowed."
        )


_TRANSITIONS: dict[DraftStatus, frozenset[DraftStatus]] = {
    DraftStatus.GENERATED: frozenset(
        {
            DraftStatus.NEEDS_REVIEW,
            DraftStatus.DELETED,
        }
    ),
    DraftStatus.NEEDS_REVIEW: frozenset(
        {
            DraftStatus.REGENERATING,
            DraftStatus.APPROVED,
            DraftStatus.REJECTED,
            DraftStatus.DELETED,
        }
    ),
    DraftStatus.REGENERATING: frozenset({DraftStatus.NEEDS_REVIEW}),
    DraftStatus.APPROVED: frozenset(
        {
            DraftStatus.NEEDS_REVIEW,
            DraftStatus.REGENERATING,
            DraftStatus.PUBLISH_PENDING_CONFIRMATION,
            DraftStatus.REJECTED,
            DraftStatus.DELETED,
        }
    ),
    DraftStatus.PUBLISH_PENDING_CONFIRMATION: frozenset(
        {
            DraftStatus.APPROVED,
            DraftStatus.NEEDS_REVIEW,
            DraftStatus.PUBLISHED,
            DraftStatus.PUBLISH_FAILED,
        }
    ),
    DraftStatus.PUBLISHED: frozenset({DraftStatus.REMOTE_DELETED}),
    DraftStatus.REJECTED: frozenset(
        {
            DraftStatus.NEEDS_REVIEW,
            DraftStatus.REGENERATING,
            DraftStatus.DELETED,
        }
    ),
    DraftStatus.DELETED: frozenset(),
    DraftStatus.PUBLISH_FAILED: frozenset(
        {
            DraftStatus.APPROVED,
            DraftStatus.NEEDS_REVIEW,
            DraftStatus.DELETED,
        }
    ),
    DraftStatus.REMOTE_DELETED: frozenset(),
}

ALLOWED_DRAFT_TRANSITIONS: Final[Mapping[DraftStatus, frozenset[DraftStatus]]] = MappingProxyType(
    _TRANSITIONS
)


def coerce_draft_status(value: DraftStatus | str) -> DraftStatus:
    """Convert a boundary value without echoing invalid input in an exception."""

    if isinstance(value, DraftStatus):
        return value
    if isinstance(value, str):
        try:
            return DraftStatus(value)
        except ValueError as exc:
            raise InvalidDraftStatusError from exc
    raise InvalidDraftStatusError


def allowed_targets(current: DraftStatus | str) -> frozenset[DraftStatus]:
    """Return the immutable allowlist for a current state."""

    return ALLOWED_DRAFT_TRANSITIONS[coerce_draft_status(current)]


def can_transition(
    current: DraftStatus | str,
    target: DraftStatus | str,
) -> bool:
    """Return whether an explicit, non-idempotent transition is allowed."""

    current_status = coerce_draft_status(current)
    target_status = coerce_draft_status(target)
    return target_status in ALLOWED_DRAFT_TRANSITIONS[current_status]


def assert_transition_allowed(
    current: DraftStatus | str,
    target: DraftStatus | str,
) -> DraftStatus:
    """Return the target status or raise before any caller mutates state."""

    current_status = coerce_draft_status(current)
    target_status = coerce_draft_status(target)
    if target_status not in ALLOWED_DRAFT_TRANSITIONS[current_status]:
        raise InvalidDraftTransitionError(current_status, target_status)
    return target_status
