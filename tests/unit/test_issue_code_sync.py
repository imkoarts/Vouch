from __future__ import annotations

import re
from pathlib import Path

from app.services.editorial_quality import QUALITY_ISSUE_CODES

ISSUE_DOC = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "personal-humanizer"
    / "references"
    / "semantic-lint.md"
)


def test_runtime_issue_registry_matches_documented_codes() -> None:
    documented = set(re.findall(r"`([A-Z][A-Z0-9_]+)`", ISSUE_DOC.read_text(encoding="utf-8")))
    assert documented <= QUALITY_ISSUE_CODES


def test_new_plainness_issue_codes_are_registered() -> None:
    assert {
        "UNSUPPORTED_SUPERLATIVE",
        "IMAGINARY_AWARENESS_GAP",
        "METAPHOR_STACKING",
        "INSIGHT_PACKAGING",
        "HISTORICAL_CONTRAST_FRAME",
        "REPLY_OVERPRODUCTION",
        "REDUNDANT_PARAGRAPH",
        "UNSUPPORTED_STAKES",
        "OVERCOMPRESSED_THEORY",
        "ABSTRACTION_LADDER",
        "POLISHED_CAUSAL_ARC",
        "UNNECESSARY_METAPHOR",
        "UNSUPPORTED_GROUP_GENERALIZATION",
        "FALSE_CASUALNESS",
        "CONCRETE_TO_ABSTRACT_DRIFT",
    } <= QUALITY_ISSUE_CODES


def test_packaged_personal_humanizer_manifest_matches_files() -> None:
    import hashlib

    skill_root = ISSUE_DOC.parents[1]
    declared: dict[str, str] = {}
    for line in (skill_root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        declared[relative] = digest

    actual_files = {
        path.relative_to(skill_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in skill_root.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    }

    assert declared == actual_files
