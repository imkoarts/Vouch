"""One canonical representation for all approval- and publication-relevant content."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from enum import Enum


class PublishableContentError(ValueError):
    pass


def _content_type_value(content_type: str | Enum) -> str:
    value = content_type.value if isinstance(content_type, Enum) else content_type
    if not isinstance(value, str) or not value:
        raise PublishableContentError("Content type must be a non-empty string enum value")
    return value


def _normalize_part(part: str) -> str:
    if not isinstance(part, str):
        raise PublishableContentError("Every publishable part must be text")
    normalized = unicodedata.normalize(
        "NFC", part.replace("\r\n", "\n").replace("\r", "\n")
    ).strip()
    if not normalized:
        raise PublishableContentError("Publishable content cannot contain an empty part")
    return normalized


def publishable_content(
    parts: Sequence[str],
    content_type: str | Enum,
) -> str | tuple[str, ...]:
    """Return the sole canonical shape used by every content hash operation.

    Threads are ordered tuples with at least two parts. Every other supported content type is a
    single string. This prevents the historically dangerous ``"text"`` versus ``("text",)``
    fingerprint divergence.
    """

    normalized = tuple(_normalize_part(part) for part in parts)
    kind = _content_type_value(content_type)
    if kind == "thread":
        if len(normalized) < 2:
            raise PublishableContentError("A thread must contain at least two parts")
        return normalized
    if len(normalized) != 1:
        raise PublishableContentError(f"Content type '{kind}' must contain exactly one part")
    return normalized[0]


def content_parts(content: str | Sequence[str]) -> tuple[str, ...]:
    """Convert an already canonical string/sequence to a non-empty tuple for persistence."""

    raw = (content,) if isinstance(content, str) else tuple(content)
    return tuple(_normalize_part(part) for part in raw)
