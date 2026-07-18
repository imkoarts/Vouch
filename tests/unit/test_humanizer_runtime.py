from __future__ import annotations

import os
from pathlib import Path

from app.services.humanizer_runtime import HumanizerRuntimeLoader

_REQUIRED = (
    "runtime-contract.md",
    "semantic-lint.md",
    "micro-reply-examples.md",
    "x-post-contract.md",
)


def _skill(root: Path, *, version: str = "9.1.0", marker: str = "first") -> Path:
    references = root / "references"
    references.mkdir(parents=True)
    skill = root / "SKILL.md"
    skill.write_text(
        f"---\nname: personal-humanizer\nversion: {version}\n---\n\nInitial {marker} contract.",
        encoding="utf-8",
    )
    for name in _REQUIRED:
        (references / name).write_text(f"{name} {marker}", encoding="utf-8")
    return skill


def test_external_skill_path_or_directory_loads_one_version_and_hash(tmp_path: Path) -> None:
    skill = _skill(tmp_path)

    direct = HumanizerRuntimeLoader(configured_path=skill, include_references=True).load()
    directory = HumanizerRuntimeLoader(configured_path=tmp_path, include_references=True).load()

    assert direct.source == directory.source == "external"
    assert direct.version == directory.version == "9.1.0"
    assert direct.skill_hash == directory.skill_hash
    assert set(_REQUIRED) <= set(direct.reference_files)
    assert "Initial first contract" in direct.initial_contract
    assert "semantic-lint.md first" in direct.revision_contract
    assert "semantic-lint.md first" in direct.semantic_reviewer_contract


def test_external_skill_cache_invalidates_on_file_change(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    loader = HumanizerRuntimeLoader(configured_path=skill, include_references=True)
    before = loader.load()

    skill.write_text(skill.read_text(encoding="utf-8") + "\nchanged", encoding="utf-8")
    stat = skill.stat()
    os.utime(skill, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    after = loader.load()

    assert after.skill_hash != before.skill_hash
    assert after.version == before.version


def test_broken_external_skill_falls_back_atomically_to_bundled(tmp_path: Path) -> None:
    runtime = HumanizerRuntimeLoader(configured_path=tmp_path / "missing" / "SKILL.md").load()

    assert runtime.source == "bundled"
    assert runtime.initial_contract
    assert runtime.revision_contract
    assert runtime.semantic_reviewer_contract
