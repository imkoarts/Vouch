"""Human-readable, atomic draft artifact storage."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from app.domain.security import compute_content_hash
from app.domain.state_machine import (
    DraftStateMachineError,
    DraftStatus,
    assert_transition_allowed,
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9-]{1,128}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_THREAD_SEPARATOR = "\n\n<!-- X_CONTENT_BOT_THREAD_PART -->\n\n"
_MEDIA_MANIFEST_VERSION = 1
_MEDIA_METADATA_KEYS = frozenset(
    {"dimensions", "duration_ms", "duration_seconds", "height", "width"}
)
_MIME_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".webm": "video/webm",
    ".webp": "image/webp",
}
_BUNDLE_FILES = (
    "metadata.json",
    "sources.json",
    "generations.json",
    "review.md",
    "media_plan.json",
    "media_manifest.json",
    "video_script.json",
    "publication.json",
)


class DraftArtifactError(RuntimeError):
    """Raised when a draft bundle is invalid or cannot be reconciled safely."""


@dataclass(frozen=True, slots=True)
class DraftFileSnapshot:
    """Authoritative content and media observed during reconciliation."""

    content: str | tuple[str, ...]
    front_matter: dict[str, Any]
    content_hash: str
    content_changed: bool
    media_manifest: dict[str, Any]
    media_manifest_changed: bool
    approval_fingerprint_changed: bool

    @property
    def changed(self) -> bool:
        """Backward-compatible name for approval-relevant divergence."""

        return self.approval_fingerprint_changed


def utc_now() -> datetime:
    return datetime.now(UTC)


def _jsonable(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        _jsonable(value), ensure_ascii=False, indent=2, sort_keys=True, default=str
    )
    _atomic_write_text(path, f"{payload}\n")


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    resolved = _jsonable(value)
    if not isinstance(resolved, Mapping):
        raise DraftArtifactError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in resolved):
        raise DraftArtifactError(f"{name} keys must be strings")
    return dict(resolved)


def _enum_value(value: Any) -> str:
    resolved = getattr(value, "value", value)
    if not isinstance(resolved, str) or not resolved:
        raise DraftArtifactError("Expected a non-empty authoritative string value")
    return resolved


def _normalize_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DraftArtifactError("Media paths must be non-empty relative strings")
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    if (
        normalized.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
        or any(":" in part or "\x00" in part for part in parts)
    ):
        raise DraftArtifactError("Media path is not a safe bundle-relative path")
    path = PurePosixPath(*parts)
    if path.is_absolute():
        raise DraftArtifactError("Media path must be relative to the draft bundle")
    return path.as_posix()


def _fingerprint_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    observed = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    identity_observed = (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
    )
    if identity_before != identity_after or identity_after != identity_observed:
        raise DraftArtifactError("Media file changed while it was being fingerprinted")
    return digest.hexdigest(), observed.st_size


class DraftArtifactStore:
    """Create and reconcile the on-disk contract for every generated draft.

    Database state and approval records remain authoritative. Front matter is a
    readable mirror and cannot grant approval, change content type, or permit a
    write action.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.quarantine_root = (self.root / ".quarantine").resolve()

    def _validate_id(self, draft_id: str) -> None:
        if not _SAFE_ID.fullmatch(draft_id):
            raise DraftArtifactError("Draft ID contains unsafe path characters")

    def draft_directory(self, draft_id: str, created_at: datetime) -> Path:
        self._validate_id(draft_id)
        path = (self.root / created_at.date().isoformat() / draft_id).resolve()
        if self.root not in path.parents:
            raise DraftArtifactError("Draft path escapes the configured drafts directory")
        return path

    def locate(self, artifact_path: str | Path) -> Path:
        path = Path(artifact_path).resolve()
        if self.root != path and self.root not in path.parents:
            raise DraftArtifactError("Draft path escapes the configured drafts directory")
        return path

    @staticmethod
    def render_content(content: str | Sequence[str]) -> str:
        if isinstance(content, str):
            return content.strip()
        parts = [str(part).strip() for part in content]
        if not parts or any(not part for part in parts):
            raise DraftArtifactError("A thread must contain non-empty post parts")
        return _THREAD_SEPARATOR.join(parts)

    @staticmethod
    def parse_content(body: str, content_type: str) -> str | tuple[str, ...]:
        normalized = body.replace("\r\n", "\n").replace("\r", "\n").strip()
        if content_type == "thread":
            parts = tuple(part.strip() for part in normalized.split(_THREAD_SEPARATOR))
            if not parts or any(not part for part in parts):
                raise DraftArtifactError("Thread markers produce an empty post part")
            return parts
        return normalized

    @staticmethod
    def build_front_matter(metadata: Mapping[str, Any]) -> str:
        required = (
            "draft_id",
            "status",
            "created_at",
            "updated_at",
            "content_type",
            "language",
            "provider",
            "model",
            "source_count",
            "character_count",
            "fact_check_status",
            "approved_at",
            "content_hash",
        )
        missing = [name for name in required if name not in metadata]
        if missing:
            raise DraftArtifactError(f"Missing front-matter fields: {', '.join(missing)}")
        clean = {name: _jsonable(metadata[name]) for name in required}
        return yaml.safe_dump(clean, allow_unicode=True, sort_keys=False).strip()

    def write_markdown(
        self,
        directory: Path,
        *,
        metadata: Mapping[str, Any],
        content: str | Sequence[str],
    ) -> None:
        target = self.locate(directory)
        front_matter = self.build_front_matter(metadata)
        body = self.render_content(content)
        _atomic_write_text(target / "draft.md", f"---\n{front_matter}\n---\n\n{body}\n")

    def _resolve_media_file(self, directory: Path, relative_path: str) -> Path:
        base = self.locate(directory)
        parts = PurePosixPath(relative_path).parts
        unresolved = base.joinpath(*parts)
        current = base
        for part in parts:
            current /= part
            if current.is_symlink():
                raise DraftArtifactError("Symlinked media files are not accepted")
        try:
            resolved = unresolved.resolve(strict=True)
        except FileNotFoundError as exc:
            raise DraftArtifactError("A required media file is missing") from exc
        if base not in resolved.parents or not resolved.is_file():
            raise DraftArtifactError("Media file escapes or is not a regular bundle file")
        return resolved

    @staticmethod
    def _required_media_paths(media_plan: Mapping[str, Any] | Any) -> tuple[str, ...]:
        plan = _mapping(media_plan, name="Media plan")
        raw_paths = plan.get("required_files", ())
        if raw_paths is None:
            return ()
        if isinstance(raw_paths, (str, bytes, bytearray)) or not isinstance(raw_paths, Sequence):
            raise DraftArtifactError("Media plan required_files must be a sequence")
        paths = tuple(_normalize_relative_path(value) for value in raw_paths)
        if len(paths) != len(set(paths)):
            raise DraftArtifactError("Media plan contains duplicate required files")
        return paths

    @staticmethod
    def _entry_metadata(media_plan: Mapping[str, Any] | Any, path: str) -> dict[str, Any]:
        plan = _mapping(media_plan, name="Media plan")
        raw_metadata = plan.get("metadata", {})
        if raw_metadata is None:
            return {}
        metadata = _mapping(raw_metadata, name="Media metadata")
        per_file = metadata.get("files", {})
        selected: dict[str, Any] = {}
        if isinstance(per_file, Mapping) and path in per_file:
            selected.update(_mapping(per_file[path], name="Per-file media metadata"))
        selected.update(
            {key: value for key, value in metadata.items() if key in _MEDIA_METADATA_KEYS}
        )
        return {key: _jsonable(value) for key, value in selected.items()}

    def build_media_manifest(
        self,
        directory: Path,
        media_plan: Mapping[str, Any] | Any,
    ) -> dict[str, Any]:
        """Hash every required media file from bytes inside the draft bundle."""

        entries: list[dict[str, Any]] = []
        for relative_path in self._required_media_paths(media_plan):
            file_path = self._resolve_media_file(directory, relative_path)
            sha256, file_size = _fingerprint_file(file_path)
            entry: dict[str, Any] = {
                "relative_path": relative_path,
                "sha256": sha256,
                "mime_type": _MIME_TYPES.get(
                    PurePosixPath(relative_path).suffix.casefold(),
                    "application/octet-stream",
                ),
                "file_size": file_size,
            }
            metadata = self._entry_metadata(media_plan, relative_path)
            if metadata:
                entry["metadata"] = metadata
            entries.append(entry)
        return {"version": _MEDIA_MANIFEST_VERSION, "files": entries}

    @staticmethod
    def _normalize_manifest(manifest: Any) -> dict[str, Any]:
        resolved = _mapping(manifest, name="Media manifest")
        if resolved.get("version") != _MEDIA_MANIFEST_VERSION:
            raise DraftArtifactError("Media manifest version is not supported")
        raw_entries = resolved.get("files")
        if isinstance(raw_entries, (str, bytes, bytearray)) or not isinstance(
            raw_entries, Sequence
        ):
            raise DraftArtifactError("Media manifest files must be a sequence")

        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_entry in raw_entries:
            entry = _mapping(raw_entry, name="Media manifest entry")
            allowed = {"relative_path", "sha256", "mime_type", "file_size", "metadata"}
            if set(entry) - allowed:
                raise DraftArtifactError("Media manifest entry has unknown fields")
            relative_path = _normalize_relative_path(entry.get("relative_path"))
            sha256 = entry.get("sha256")
            mime_type = entry.get("mime_type")
            file_size = entry.get("file_size")
            if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
                raise DraftArtifactError("Media manifest SHA-256 is invalid")
            if not isinstance(mime_type, str) or not mime_type:
                raise DraftArtifactError("Media manifest MIME type is invalid")
            if isinstance(file_size, bool) or not isinstance(file_size, int) or file_size < 0:
                raise DraftArtifactError("Media manifest file size is invalid")
            if relative_path in seen:
                raise DraftArtifactError("Media manifest contains duplicate files")
            seen.add(relative_path)
            normalized_entry: dict[str, Any] = {
                "relative_path": relative_path,
                "sha256": sha256,
                "mime_type": mime_type,
                "file_size": file_size,
            }
            if "metadata" in entry:
                normalized_entry["metadata"] = _mapping(
                    entry["metadata"], name="Media manifest metadata"
                )
            entries.append(normalized_entry)
        return {"version": _MEDIA_MANIFEST_VERSION, "files": entries}

    def load_media_manifest(self, directory: Path) -> dict[str, Any]:
        path = self.locate(directory) / "media_manifest.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DraftArtifactError("Media manifest is missing or malformed") from exc
        return self._normalize_manifest(payload)

    def validate_media_manifest(
        self,
        directory: Path,
        media_plan: Mapping[str, Any] | Any,
        manifest: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Re-read media bytes and fail closed when they differ from the manifest."""

        expected = self.load_media_manifest(directory) if manifest is None else manifest
        normalized = self._normalize_manifest(expected)
        observed = self.build_media_manifest(directory, media_plan)
        if normalized != observed:
            raise DraftArtifactError("Media bytes do not match the approved manifest")
        return normalized

    def validated_media_files(
        self,
        directory: Path,
        media_plan: Mapping[str, Any] | Any,
    ) -> tuple[Path, ...]:
        """Return only bundle-contained files whose bytes match the stored manifest."""

        manifest = self.validate_media_manifest(directory, media_plan)
        return tuple(
            self._resolve_media_file(directory, str(entry["relative_path"]))
            for entry in manifest["files"]
        )

    def refresh_media_manifest(
        self,
        directory: Path,
        media_plan: Mapping[str, Any] | Any,
    ) -> dict[str, Any]:
        """Persist a newly observed manifest after approval has been revoked."""

        target = self.locate(directory)
        manifest = self.build_media_manifest(target, media_plan)
        _atomic_write_json(target / "media_manifest.json", manifest)
        return manifest

    def recompute_approval_hash(
        self,
        directory: Path,
        content: str | Sequence[str],
        media_plan: Mapping[str, Any] | Any,
    ) -> str:
        """Compute the current hash, observing media even after manual replacement."""

        manifest = self.build_media_manifest(directory, media_plan)
        return compute_content_hash(content, media_plan, manifest)

    def compute_approval_hash(
        self,
        directory: Path,
        content: str | Sequence[str],
        media_plan: Mapping[str, Any] | Any,
        *,
        require_valid_manifest: bool = False,
    ) -> str:
        """Compute a draft hash, optionally requiring the persisted manifest.

        Reconciliation deliberately observes current bytes so a manual replacement
        produces a new hash. Approval/publication callers can request strict manifest
        validation before trusting the result.
        """

        if not require_valid_manifest:
            return self.recompute_approval_hash(directory, content, media_plan)
        manifest = self.validate_media_manifest(directory, media_plan)
        return compute_content_hash(content, media_plan, manifest)

    def validate_approval_hash(
        self,
        directory: Path,
        content: str | Sequence[str],
        media_plan: Mapping[str, Any] | Any,
        expected_hash: str,
    ) -> str:
        """Revalidate media and the complete approval fingerprint for publication."""

        manifest = self.validate_media_manifest(directory, media_plan)
        observed_hash = compute_content_hash(content, media_plan, manifest)
        if not isinstance(expected_hash, str) or not hmac.compare_digest(
            observed_hash, expected_hash
        ):
            raise DraftArtifactError("Draft content does not match the approved hash")
        return observed_hash

    def create_bundle(
        self,
        *,
        draft_id: str,
        created_at: datetime,
        metadata: Mapping[str, Any],
        content: str | Sequence[str],
        sources: Any = None,
        generations: Any = None,
        review: str = "",
        media_plan: Any = None,
        video_script: Any = None,
        publication: Any = None,
    ) -> Path:
        """Build a complete sibling directory and reveal it with one atomic rename."""

        directory = self.draft_directory(draft_id, created_at)
        if directory.exists():
            raise DraftArtifactError("Draft artifact directory already exists")
        parent = directory.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{draft_id}.", suffix=".tmp", dir=parent)
        ).resolve()
        plan = media_plan or {"type": "none"}
        try:
            manifest = self.build_media_manifest(temporary, plan)
            content_type = _enum_value(metadata.get("content_type"))
            rendered = self.render_content(content)
            canonical_content = self.parse_content(rendered, content_type)
            observed_hash = compute_content_hash(canonical_content, plan, manifest)
            metadata_hash = metadata.get("content_hash")
            if not isinstance(metadata_hash, str) or not hmac.compare_digest(
                observed_hash, metadata_hash
            ):
                raise DraftArtifactError(
                    "Bundle metadata hash does not cover its content and media manifest"
                )
            self.write_markdown(temporary, metadata=metadata, content=content)
            _atomic_write_json(temporary / "metadata.json", dict(metadata))
            _atomic_write_json(temporary / "sources.json", sources or [])
            _atomic_write_json(temporary / "generations.json", generations or [])
            _atomic_write_text(temporary / "review.md", review.rstrip() + "\n")
            _atomic_write_json(temporary / "media_plan.json", plan)
            _atomic_write_json(temporary / "media_manifest.json", manifest)
            _atomic_write_json(temporary / "video_script.json", video_script or {})
            _atomic_write_json(temporary / "publication.json", publication or {})
            os.replace(temporary, directory)
        except BaseException as exc:
            if temporary.exists():
                resolved = temporary.resolve()
                if self.root not in resolved.parents or not resolved.name.endswith(".tmp"):
                    raise DraftArtifactError("Refusing unsafe temporary bundle cleanup") from exc
                shutil.rmtree(resolved)
            raise
        return directory

    def read_markdown(
        self,
        directory: Path,
        *,
        expected_hash: str,
        expected_content: str | Sequence[str] | None = None,
        media_plan: Mapping[str, Any] | Any,
        expected_content_type: str | Any,
    ) -> DraftFileSnapshot:
        """Reconcile edits using the database content type, never front matter authority."""

        bundle = self.locate(directory)
        path = bundle / "draft.md"
        try:
            text = path.read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError) as exc:
            raise DraftArtifactError("draft.md is missing or is not valid UTF-8") from exc
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if not normalized.startswith("---\n"):
            raise DraftArtifactError("draft.md is missing YAML front matter")
        try:
            raw_front, body = normalized[4:].split("\n---\n", maxsplit=1)
        except ValueError as exc:
            raise DraftArtifactError("draft.md has malformed YAML front matter") from exc
        try:
            front = yaml.safe_load(raw_front) or {}
        except yaml.YAMLError as exc:
            raise DraftArtifactError("draft.md has malformed YAML front matter") from exc
        if not isinstance(front, dict):
            raise DraftArtifactError("draft.md front matter must be a mapping")

        authoritative_type = _enum_value(expected_content_type)
        mirrored_type = front.get("content_type")
        if mirrored_type != authoritative_type:
            raise DraftArtifactError(
                "draft.md content type conflicts with the authoritative database value"
            )
        content = self.parse_content(body, authoritative_type)
        stored_manifest = self.load_media_manifest(bundle)
        observed_manifest = self.build_media_manifest(bundle, media_plan)
        manifest_changed = stored_manifest != observed_manifest
        fingerprint = compute_content_hash(content, media_plan, observed_manifest)
        fingerprint_changed = not isinstance(expected_hash, str) or not hmac.compare_digest(
            fingerprint, expected_hash
        )
        if expected_content is None:
            # Older callers supplied only the combined approval fingerprint.  A media
            # change is independently observable, but a simultaneous text edit cannot
            # be separated without the authoritative DB content.  Domain services
            # always pass expected_content; this fallback preserves the public helper.
            content_changed = fingerprint_changed and not manifest_changed
        else:
            expected_rendered = self.render_content(expected_content)
            expected_canonical = self.parse_content(expected_rendered, authoritative_type)
            content_changed = content != expected_canonical
        return DraftFileSnapshot(
            content=content,
            front_matter=front,
            content_hash=fingerprint,
            content_changed=content_changed,
            media_manifest=observed_manifest,
            media_manifest_changed=manifest_changed,
            approval_fingerprint_changed=fingerprint_changed,
        )

    def update_json(self, directory: Path, filename: str, value: Any) -> None:
        if filename not in _BUNDLE_FILES:
            raise DraftArtifactError("Refusing to write an unknown draft artifact")
        _atomic_write_json(self.locate(directory) / filename, value)

    def quarantine_bundle(
        self,
        directory: Path,
        *,
        current_status: DraftStatus | str,
    ) -> Path:
        """Atomically move a locally deleted bundle into append-only quarantine."""

        path = self.locate(directory)
        if (
            path == self.root
            or self.quarantine_root == path
            or self.quarantine_root in path.parents
        ):
            raise DraftArtifactError("Refusing to quarantine this draft path")
        try:
            assert_transition_allowed(current_status, DraftStatus.DELETED)
        except DraftStateMachineError as exc:
            raise DraftArtifactError(
                "Current draft status does not permit ordinary local deletion"
            ) from exc
        if not path.is_dir():
            raise DraftArtifactError("Draft artifact directory does not exist")

        relative = path.relative_to(self.root)
        stamp = utc_now().strftime("%Y%m%dT%H%M%S.%fZ")
        destination = (self.quarantine_root / stamp / relative).resolve()
        if self.quarantine_root not in destination.parents or destination.exists():
            raise DraftArtifactError("Could not allocate a safe quarantine path")
        destination.parent.mkdir(parents=True, exist_ok=False)
        os.replace(path, destination)
        return destination

    def remove_bundle(
        self,
        directory: Path,
        *,
        current_status: DraftStatus | str,
    ) -> Path:
        """Compatibility name: local removal always means quarantine, never delete."""

        return self.quarantine_bundle(directory, current_status=current_status)
