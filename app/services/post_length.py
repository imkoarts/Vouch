"""Resolve configured post-length modes before any provider request is sent."""

from __future__ import annotations

from dataclasses import dataclass
from random import SystemRandom

from app.domain.enums import ContentType, PostLengthMode


@dataclass(frozen=True, slots=True)
class PostLengthSelection:
    """A concrete range selected before calling an LLM provider."""

    configured_mode: PostLengthMode
    resolved_mode: PostLengthMode
    minimum: int
    maximum: int
    content_type: ContentType

    @property
    def label(self) -> str:
        labels = {
            PostLengthMode.SHORT: "short",
            PostLengthMode.MEDIUM: "medium",
            PostLengthMode.LONGREAD: "longread",
        }
        return labels[self.resolved_mode]

    def as_metadata(self) -> dict[str, object]:
        return {
            "configured_mode": self.configured_mode.value,
            "resolved_mode": self.resolved_mode.value,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


_CONCRETE_MODES = (
    PostLengthMode.SHORT,
    PostLengthMode.MEDIUM,
    PostLengthMode.LONGREAD,
)

_RANGES: dict[PostLengthMode, tuple[int, int, ContentType]] = {
    PostLengthMode.SHORT: (1, 280, ContentType.SHORT_POST),
    PostLengthMode.MEDIUM: (300, 500, ContentType.LONG_POST),
    PostLengthMode.LONGREAD: (500, 1_000, ContentType.LONG_POST),
}


def resolve_post_length(
    configured_mode: PostLengthMode | str,
    *,
    random_source: SystemRandom | None = None,
) -> PostLengthSelection:
    """Resolve ``random`` to one concrete mode before generation begins."""

    mode = PostLengthMode(configured_mode)
    resolved = mode
    if mode is PostLengthMode.RANDOM:
        resolved = (random_source or SystemRandom()).choice(_CONCRETE_MODES)
    minimum, maximum, content_type = _RANGES[resolved]
    return PostLengthSelection(
        configured_mode=mode,
        resolved_mode=resolved,
        minimum=minimum,
        maximum=maximum,
        content_type=content_type,
    )


def selection_from_metadata(raw: object) -> PostLengthSelection | None:
    """Read a previously persisted selection without trusting arbitrary fields."""

    if not isinstance(raw, dict):
        return None
    try:
        configured = PostLengthMode(str(raw["configured_mode"]))
        resolved = PostLengthMode(str(raw["resolved_mode"]))
    except (KeyError, ValueError):
        return None
    if resolved is PostLengthMode.RANDOM:
        return None
    minimum, maximum, content_type = _RANGES[resolved]
    return PostLengthSelection(
        configured_mode=configured,
        resolved_mode=resolved,
        minimum=minimum,
        maximum=maximum,
        content_type=content_type,
    )
