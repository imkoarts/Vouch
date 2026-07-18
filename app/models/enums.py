"""SQLAlchemy enum adapter and compatibility exports for domain enums.

The enum classes live exclusively in :mod:`app.domain.enums`.  Re-exporting
them here keeps existing persistence callers source-compatible without creating
a second contract.
"""

from __future__ import annotations

from enum import StrEnum

from sqlalchemy import Enum as SAEnum

from app.domain.enums import (
    ApiDirection,
    ApprovalAction,
    ContentSourceKind,
    ContentType,
    DraftStatus,
    FactCheckStatus,
    IdeaStatus,
    LLMRunStatus,
    PublicationKind,
    PublicationStatus,
    ReplyCandidateStatus,
    TopicKind,
    VideoGeneratorMode,
    VideoJobStatus,
    VideoScriptStatus,
)

__all__ = [
    "ApiDirection",
    "ApprovalAction",
    "ContentSourceKind",
    "ContentType",
    "DraftStatus",
    "FactCheckStatus",
    "IdeaStatus",
    "LLMRunStatus",
    "PublicationKind",
    "PublicationStatus",
    "ReplyCandidateStatus",
    "TopicKind",
    "VideoGeneratorMode",
    "VideoJobStatus",
    "VideoScriptStatus",
    "portable_enum",
]


def portable_enum[EnumT: StrEnum](enum_class: type[EnumT], *, name: str) -> SAEnum:
    """Build a CHECK-backed enum that behaves identically on supported DBs."""

    return SAEnum(
        enum_class,
        name=name,
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        values_callable=lambda members: [member.value for member in members],
    )
