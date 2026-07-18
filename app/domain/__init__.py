"""Dependency-free domain rules and safety primitives."""

from app.domain.enums import ContentType, DraftStatus, FactCheckStatus, MediaType
from app.domain.state_machine import (
    ALLOWED_DRAFT_TRANSITIONS,
    InvalidDraftStatusError,
    InvalidDraftTransitionError,
    assert_transition_allowed,
    can_transition,
)

__all__ = [
    "ALLOWED_DRAFT_TRANSITIONS",
    "ContentType",
    "DraftStatus",
    "FactCheckStatus",
    "InvalidDraftStatusError",
    "InvalidDraftTransitionError",
    "MediaType",
    "assert_transition_allowed",
    "can_transition",
]
