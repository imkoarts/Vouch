"""Single runtime source of truth for initial drafting, review, and bounded revision."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Literal

from pydantic import BaseModel, ConfigDict

_MAX_FILE_BYTES = 256 * 1024
_MAX_CONTRACT_WORDS = 6000
_REQUIRED_REFERENCES = (
    "runtime-contract.md",
    "semantic-lint.md",
    "micro-reply-examples.md",
    "x-post-contract.md",
)


def _default_bundled_root() -> Path:
    source_checkout = Path(__file__).resolve().parents[2] / "docs" / "personal-humanizer"
    if (source_checkout / "SKILL.md").is_file():
        return source_checkout
    return Path(__file__).resolve().parents[1] / "_bundled" / "personal-humanizer"


class HumanizerRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["external", "bundled"]
    version: str
    skill_path: str | None
    skill_hash: str
    initial_contract: str
    revision_contract: str
    semantic_reviewer_contract: str
    reference_files: tuple[str, ...] = ()
    semantic_references_enabled: bool = False


@dataclass(slots=True)
class HumanizerRuntimeLoader:
    """Load a skill at request time with a modification-time-aware cache."""

    configured_path: str | Path | None = None
    include_references: bool = False
    bundled_root: Path = field(default_factory=_default_bundled_root)
    prompt_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "prompts"
    )
    _cache_key: tuple[tuple[str, int, int], ...] | None = field(default=None, init=False)
    _cache: HumanizerRuntime | None = field(default=None, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)

    @staticmethod
    def _read(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size > _MAX_FILE_BYTES:
            raise ValueError(f"Humanizer file exceeds {_MAX_FILE_BYTES} bytes: {path}")
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _compact(text: str) -> str:
        words = text.split()
        return " ".join(words[:_MAX_CONTRACT_WORDS]) if len(words) > _MAX_CONTRACT_WORDS else text

    @staticmethod
    def _version(skill_text: str, root: Path) -> str:
        match = re.search(r"(?m)^version:\s*['\"]?([^'\"\s]+)", skill_text)
        if match:
            return match.group(1)
        version_file = root / "VERSION.md"
        if version_file.is_file():
            version_text = version_file.read_text(encoding="utf-8")
            version_match = re.search(r"\b\d+\.\d+(?:\.\d+)?\b", version_text)
            if version_match:
                return version_match.group(0)
        return "unknown"

    @staticmethod
    def _hash(files: list[tuple[Path, str]]) -> str:
        digest = hashlib.sha256()
        for path, text in sorted(files, key=lambda item: str(item[0])):
            digest.update(str(path.name).encode())
            digest.update(b"\0")
            digest.update(text.encode())
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _cache_signature(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            sorted(
                (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
                for path in paths
                if path.is_file()
            )
        )

    def _resolve_external(self) -> tuple[Path, Path] | None:
        if not self.configured_path:
            return None
        path = Path(self.configured_path).expanduser()
        if path.is_dir():
            skill = path / "SKILL.md"
            root = path
        else:
            skill = path
            root = path.parent
        if not skill.is_file():
            raise FileNotFoundError(f"Configured Humanizer skill does not exist: {skill}")
        return skill, root

    def _load_from_skill(self, skill: Path, root: Path, *, external: bool) -> HumanizerRuntime:
        skill_text = self._read(skill)
        references = root / "references"
        files: list[tuple[Path, str]] = [(skill, skill_text)]
        ref_text: dict[str, str] = {}
        for name in _REQUIRED_REFERENCES:
            path = references / name
            if path.is_file():
                text = self._read(path)
                files.append((path, text))
                ref_text[name] = text
        long_form = references / "long-form-contract.md"
        if long_form.is_file():
            text = self._read(long_form)
            files.append((long_form, text))
            ref_text[long_form.name] = text
        runtime = ref_text.get("runtime-contract.md", "")
        x_post = ref_text.get("x-post-contract.md", "")
        semantic = ref_text.get("semantic-lint.md", "")
        micro_examples = ref_text.get("micro-reply-examples.md", "")
        long_contract = ref_text.get("long-form-contract.md", "")
        initial = "\n\n".join((skill_text, runtime, micro_examples, x_post, long_contract)).strip()
        revision = "\n\n".join((runtime, semantic, micro_examples, x_post)).strip()
        reviewer = "\n\n".join((runtime, semantic, micro_examples, long_contract)).strip()
        return HumanizerRuntime(
            source="external" if external else "bundled",
            version=self._version(skill_text, root),
            skill_path=str(skill.resolve()) if external else None,
            skill_hash=self._hash(files),
            initial_contract=self._compact(initial),
            revision_contract=self._compact(revision),
            semantic_reviewer_contract=self._compact(reviewer),
            reference_files=tuple(path.name for path, _ in files[1:]),
            semantic_references_enabled=self.include_references,
        )

    def _load_bundled(self) -> HumanizerRuntime:
        skill = self.bundled_root / "SKILL.md"
        if skill.is_file():
            return self._load_from_skill(skill, self.bundled_root, external=False)
        # Packaged wheel fallback when docs are unavailable.
        initial = self._read(self.prompt_root / "humanizer_initial.txt")
        revision = self._read(self.prompt_root / "humanizer.txt")
        reviewer = self._read(self.prompt_root / "critic.txt")
        files = [
            (self.prompt_root / "humanizer_initial.txt", initial),
            (self.prompt_root / "humanizer.txt", revision),
            (self.prompt_root / "critic.txt", reviewer),
        ]
        return HumanizerRuntime(
            source="bundled",
            version="bundled-prompts",
            skill_path=None,
            skill_hash=self._hash(files),
            initial_contract=initial,
            revision_contract=revision,
            semantic_reviewer_contract=reviewer,
            semantic_references_enabled=False,
        )

    def load(self) -> HumanizerRuntime:
        """Return one coherent runtime; external load failures fall back atomically."""

        with self._lock:
            try:
                resolved = self._resolve_external()
                if resolved is not None:
                    skill, root = resolved
                    paths = [skill, *(root / "references").glob("*.md")]
                    signature = self._cache_signature(paths)
                    if self._cache is not None and signature == self._cache_key:
                        return self._cache
                    runtime = self._load_from_skill(skill, root, external=True)
                    self._cache = runtime
                    self._cache_key = signature
                    return runtime
            except (OSError, UnicodeError, ValueError):
                # A broken external skill must never mix versions with bundled revision rules.
                pass
            paths = [
                self.bundled_root / "SKILL.md",
                *(self.bundled_root / "references").glob("*.md"),
                self.prompt_root / "humanizer_initial.txt",
                self.prompt_root / "humanizer.txt",
                self.prompt_root / "critic.txt",
            ]
            signature = self._cache_signature(paths)
            if self._cache is not None and signature == self._cache_key:
                return self._cache
            runtime = self._load_bundled()
            self._cache = runtime
            self._cache_key = signature
            return runtime
