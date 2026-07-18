"""Load curated, versioned style examples without training on unreviewed drafts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_MAX_STYLE_EXAMPLES_BYTES = 128 * 1024
_MAX_EXAMPLE_CHARACTERS = 4000


@dataclass(frozen=True, slots=True)
class CuratedStyleExamples:
    approved: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()


def _clean_output(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned[:_MAX_EXAMPLE_CHARACTERS]


def _outputs(items: object) -> tuple[str, ...]:
    if not isinstance(items, list):
        return ()
    values: list[str] = []
    for item in items:
        if isinstance(item, str):
            output = _clean_output(item)
        elif isinstance(item, dict):
            output = _clean_output(item.get("output"))
        else:
            output = None
        if output is not None and output not in values:
            values.append(output)
    return tuple(values)


def load_curated_style_examples(
    path: Path | None,
    *,
    editorial_intent: str,
    max_approved: int,
    max_rejected: int,
) -> CuratedStyleExamples:
    """Load only operator-curated examples for the active editorial intent."""

    if path is None or not path.is_file():
        return CuratedStyleExamples()
    if path.stat().st_size > _MAX_STYLE_EXAMPLES_BYTES:
        raise ValueError(f"Style examples file exceeds {_MAX_STYLE_EXAMPLES_BYTES} bytes: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("style_examples.yml must contain a mapping")
    modes = payload.get("modes", {})
    if not isinstance(modes, dict):
        raise ValueError("style_examples.yml modes must contain a mapping")
    selected: Any = modes.get(editorial_intent, {})
    if not isinstance(selected, dict):
        return CuratedStyleExamples()
    approved = _outputs(selected.get("approved"))[: max(0, max_approved)]
    rejected = _outputs(selected.get("rejected"))[: max(0, max_rejected)]
    return CuratedStyleExamples(approved=approved, rejected=rejected)
