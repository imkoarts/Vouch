"""Deterministic filesystem projections and DB/filesystem recovery diagnostics.

The SQLAlchemy records are authoritative.  This module deliberately contains no
draft state transitions and performs no remote writes.  Callers should invoke
``project`` from a fresh session after the transaction that changed a draft has
committed.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.domain.content import publishable_content
from app.domain.security import redact_secrets
from app.domain.x_text import weighted_length
from app.models import Draft, DraftApproval, DraftVersion, IdeaSource, Publication
from app.models.enums import ApprovalAction, DraftStatus, PublicationStatus
from app.services import draft_files
from app.services.draft_files import DraftArtifactError, DraftArtifactStore

_PROJECTION_SCHEMA_VERSION = 1
_DATE_DIRECTORY_LENGTH = 10


class ArtifactProjectionError(RuntimeError):
    """Raised when authoritative records cannot form a safe projection."""


class RecoveryIssueKind(StrEnum):
    """Machine-readable classes of DB/filesystem divergence."""

    ORPHAN_BUNDLE = "orphan_bundle"
    MISSING_BUNDLE = "missing_bundle"
    STALE_PROJECTION = "stale_projection"
    CONTENT_DIVERGENCE = "content_divergence"
    QUARANTINE_DIVERGENCE = "quarantine_divergence"
    INVALID_AUTHORITATIVE_STATE = "invalid_authoritative_state"
    INVALID_ARTIFACT_PATH = "invalid_artifact_path"


class RecoveryAction(StrEnum):
    """Explicit actions a recovery issue can require or perform."""

    PROJECT = "project"
    RECREATE_BUNDLE = "recreate_bundle"
    RECONCILE_CONTENT = "reconcile_content"
    QUARANTINE_PRESERVING = "quarantine_preserving"
    RESTORE_FROM_QUARANTINE = "restore_from_quarantine"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """Files written for one authoritative draft snapshot."""

    draft_id: str
    directory: Path
    fingerprint: str
    files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RecoveryIssue:
    """One detected divergence, without draft/source payloads in its details."""

    kind: RecoveryIssueKind
    action: RecoveryAction
    path: Path
    draft_id: str | None = None
    details: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecoveryActionResult:
    """Outcome of one explicitly attempted safe recovery action."""

    action: RecoveryAction
    path: Path
    succeeded: bool
    draft_id: str | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    """Current unresolved issues plus actions attempted by this invocation."""

    issues: tuple[RecoveryIssue, ...]
    actions: tuple[RecoveryActionResult, ...] = ()


@dataclass(frozen=True, slots=True)
class _Projection:
    draft: Draft
    version: DraftVersion
    content: str | tuple[str, ...]
    metadata: dict[str, Any]
    sources: list[dict[str, Any]]
    generations: dict[str, Any]
    review: str
    publication: dict[str, Any]
    fingerprint: str


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def _enum_value(value: Any) -> str:
    resolved = getattr(value, "value", value)
    if not isinstance(resolved, str) or not resolved:
        raise ArtifactProjectionError("Authoritative enum value is invalid")
    return resolved


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"


def _redacted(value: Any) -> Any:
    """Redact defense-in-depth metadata without changing publishable content."""

    return redact_secrets(value)


def _one_line(value: Any) -> str:
    redacted = _redacted(value)
    if redacted is None:
        return ""
    return " ".join(str(redacted).replace("|", "\\|").split())


class ArtifactProjectionService:
    """Build bundles from DB records and reconcile recoverable divergence.

    Projection writes are atomic per file.  A fingerprint is repeated across
    the generated artifacts so a crash between file replacements is detected by
    the next recovery scan.  Filesystem deletion and quarantine moves are never
    performed by this service.
    """

    def __init__(self, session: Session, artifacts: DraftArtifactStore) -> None:
        self.session = session
        self.artifacts = artifacts

    def _draft(self, draft_id: str) -> Draft:
        draft = self.session.scalar(select(Draft).where(Draft.id == draft_id))
        if draft is None:
            raise LookupError("Draft was not found")
        return draft

    def _versions(self, draft: Draft) -> list[DraftVersion]:
        return list(
            self.session.scalars(
                select(DraftVersion)
                .where(DraftVersion.draft_id == draft.id)
                .order_by(DraftVersion.version_number, DraftVersion.id)
            ).all()
        )

    def _approvals(self, draft: Draft) -> list[DraftApproval]:
        return list(
            self.session.scalars(
                select(DraftApproval)
                .where(DraftApproval.draft_id == draft.id)
                .order_by(DraftApproval.created_at, DraftApproval.id)
            ).all()
        )

    def _publications(self, draft: Draft) -> list[Publication]:
        return list(
            self.session.scalars(
                select(Publication)
                .where(Publication.draft_id == draft.id)
                .order_by(
                    Publication.thread_position,
                    Publication.created_at,
                    Publication.id,
                )
            ).all()
        )

    def _source_links(self, draft: Draft) -> list[IdeaSource]:
        if draft.idea_id is None:
            return []
        return list(
            self.session.scalars(
                select(IdeaSource)
                .where(IdeaSource.idea_id == draft.idea_id)
                .options(
                    selectinload(IdeaSource.source_post),
                    selectinload(IdeaSource.topic),
                )
                .order_by(IdeaSource.position, IdeaSource.id)
            ).all()
        )

    @staticmethod
    def _active_approval(
        approvals: list[DraftApproval],
        current_version: DraftVersion,
        current_hash: str,
    ) -> DraftApproval | None:
        revoked_ids = {
            event.supersedes_id
            for event in approvals
            if event.action is ApprovalAction.REVOKED and event.supersedes_id is not None
        }
        candidates = [
            event
            for event in approvals
            if event.action is ApprovalAction.APPROVED
            and event.id not in revoked_ids
            and event.draft_version_id == current_version.id
            and event.content_hash == current_hash
        ]
        return candidates[-1] if candidates else None

    @staticmethod
    def _parts(version: DraftVersion, draft: Draft) -> tuple[str, ...]:
        raw_parts = version.content.get("parts")
        if (
            not isinstance(raw_parts, list)
            or not raw_parts
            or not all(isinstance(part, str) for part in raw_parts)
        ):
            raise ArtifactProjectionError("Draft version has invalid content parts")
        canonical = publishable_content(tuple(raw_parts), draft.content_type)
        return (canonical,) if isinstance(canonical, str) else canonical

    @staticmethod
    def _source_payload(link: IdeaSource) -> dict[str, Any]:
        common: dict[str, Any] = {
            "link_id": link.id,
            "position": link.position,
            "source_priority": link.source_priority,
            "contribution": link.contribution,
            "link_metadata": _redacted(link.link_metadata),
            "trusted": False,
            "instructions_allowed": False,
        }
        if link.source_post is not None:
            source = link.source_post
            common.update(
                {
                    "kind": "untrusted_external_content",
                    "source_type": "x_post",
                    "source_id": source.x_post_id or source.id,
                    "content": source.text,
                    "url": source.url,
                    "author_id": source.author_id,
                    "username": source.username,
                    "language": source.language,
                    "media_metadata": _redacted(source.media_metadata),
                }
            )
            return common
        if link.topic is not None:
            topic = link.topic
            common.update(
                {
                    "kind": "untrusted_external_content",
                    "source_type": "topic",
                    "source_id": topic.id,
                    "content": topic.description or topic.query or topic.name,
                    "topic_name": topic.name,
                    "topic_kind": topic.kind.value,
                }
            )
            return common
        raise ArtifactProjectionError("Idea source link has no source record")

    @staticmethod
    def _generation_payload(
        draft: Draft,
        versions: list[DraftVersion],
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for version in versions:
            parts = ArtifactProjectionService._parts(version, draft)
            generation_metadata = dict(version.generation_metadata)
            generation_result = generation_metadata.pop("result", None)
            raw_lengths = version.content.get("weighted_lengths")
            lengths = (
                raw_lengths
                if isinstance(raw_lengths, list)
                and len(raw_lengths) == len(parts)
                and all(isinstance(item, int) for item in raw_lengths)
                else [weighted_length(part) for part in parts]
            )
            payload.append(
                {
                    "id": version.id,
                    "version_number": version.version_number,
                    "is_current": version.version_number == draft.current_version_number,
                    "created_at": _iso(version.created_at),
                    "origin": version.origin,
                    "content_hash": version.content_hash,
                    "parts": list(parts),
                    "weighted_lengths": lengths,
                    "provider": version.provider,
                    "model": version.model,
                    "feedback": _redacted(version.feedback),
                    "critique": _redacted(version.critique),
                    "result": _redacted(generation_result),
                    "metadata": _redacted(generation_metadata),
                }
            )
        return payload

    @staticmethod
    def _approval_payload(
        approvals: list[DraftApproval], active: DraftApproval | None
    ) -> list[dict[str, Any]]:
        revoked_ids = {
            event.supersedes_id
            for event in approvals
            if event.action is ApprovalAction.REVOKED and event.supersedes_id is not None
        }
        return [
            {
                "id": event.id,
                "created_at": _iso(event.created_at),
                "action": event.action.value,
                "draft_version_id": event.draft_version_id,
                "content_hash": event.content_hash,
                "actor": _redacted(event.actor),
                "reason": _redacted(event.reason),
                "supersedes_id": event.supersedes_id,
                "revoked": event.id in revoked_ids,
                "active": active is not None and event.id == active.id,
            }
            for event in approvals
        ]

    @staticmethod
    def _publication_payload(
        publications: list[Publication], versions: list[DraftVersion]
    ) -> list[dict[str, Any]]:
        version_numbers = {version.id: version.version_number for version in versions}
        return [
            {
                "id": record.id,
                "draft_version_id": record.draft_version_id,
                "version_number": version_numbers.get(record.draft_version_id),
                "content_hash": record.content_hash,
                "kind": record.kind.value,
                "status": record.status.value,
                "thread_position": record.thread_position,
                "request_payload_hash": record.request_payload_hash,
                "x_post_id": record.x_post_id,
                "parent_post_id": record.parent_post_id,
                "account_id": record.account_id,
                "response_metadata": _redacted(record.response_metadata),
                "estimated_cost_usd": (
                    str(record.estimated_cost_usd)
                    if record.estimated_cost_usd is not None
                    else None
                ),
                "attempt_count": record.attempt_count,
                "created_at": _iso(record.created_at),
                "updated_at": _iso(record.updated_at),
                "published_at": _iso(record.published_at),
                "deletion_timestamp": _iso(record.deletion_timestamp),
                "error_details": _redacted(record.error_details),
            }
            for record in publications
        ]

    @staticmethod
    def _review_text(
        draft: Draft,
        version: DraftVersion,
        active: DraftApproval | None,
        approval_events: list[dict[str, Any]],
        fingerprint: str,
    ) -> str:
        flags = ", ".join(draft.blocking_safety_flags) or "none"
        lines = [
            "# Draft review",
            "",
            f"- Draft ID: `{draft.id}`",
            f"- Status: `{draft.status.value}`",
            f"- Current version: `{version.version_number}`",
            f"- Content hash: `{version.content_hash}`",
            f"- Fact check: `{draft.fact_check_status.value}`",
            f"- Blocking safety flags: `{_one_line(flags)}`",
            f"- Active approval: `{active.id if active is not None else 'none'}`",
            f"- Projection fingerprint: `{fingerprint}`",
            "",
            "## Approval history",
            "",
        ]
        if not approval_events:
            lines.append("No approval events.")
        else:
            lines.extend(
                [
                    "| created_at | action | actor | version_id | hash | reason | active |",
                    "|---|---|---|---|---|---|---|",
                ]
            )
            for event in approval_events:
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            _one_line(event["created_at"]),
                            _one_line(event["action"]),
                            _one_line(event["actor"]),
                            _one_line(event["draft_version_id"]),
                            _one_line(event["content_hash"]),
                            _one_line(event["reason"]),
                            "yes" if event["active"] else "no",
                        )
                    )
                    + " |"
                )
        return "\n".join(lines).rstrip() + "\n"

    def _build(self, draft: Draft) -> _Projection:
        versions = self._versions(draft)
        current = next(
            (
                version
                for version in versions
                if version.version_number == draft.current_version_number
            ),
            None,
        )
        if current is None:
            raise ArtifactProjectionError("Authoritative current draft version is missing")
        if not draft.current_content_hash or current.content_hash != draft.current_content_hash:
            raise ArtifactProjectionError(
                "Draft and current version content hashes are inconsistent"
            )
        parts = self._parts(current, draft)
        content = publishable_content(parts, draft.content_type)
        links = self._source_links(draft)
        sources = [self._source_payload(link) for link in links]
        approvals = self._approvals(draft)
        active = self._active_approval(approvals, current, draft.current_content_hash)
        approval_events = self._approval_payload(approvals, active)
        publications = self._publications(draft)
        publication_records = self._publication_payload(publications, versions)
        generation_records = self._generation_payload(draft, versions)
        base_metadata: dict[str, Any] = {
            "draft_id": draft.id,
            "status": draft.status.value,
            "created_at": _iso(draft.created_at),
            "updated_at": _iso(draft.updated_at),
            "content_type": draft.content_type.value,
            "language": draft.language,
            "provider": draft.provider,
            "model": draft.model,
            "source_count": len(links),
            "character_count": sum(len(part) for part in parts),
            "weighted_length": sum(weighted_length(part) for part in parts),
            "fact_check_status": draft.fact_check_status.value,
            "approved_at": _iso(active.created_at) if active is not None else None,
            "content_hash": draft.current_content_hash,
            "current_version_number": draft.current_version_number,
            "row_version": draft.row_version,
            "blocking_safety_flags": list(draft.blocking_safety_flags),
            "active_approval_id": active.id if active is not None else None,
        }
        fingerprint_input = {
            "metadata": base_metadata,
            "sources": sources,
            "generations": generation_records,
            "approval_events": approval_events,
            "publications": publication_records,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_input,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        projection_marker = {
            "schema_version": _PROJECTION_SCHEMA_VERSION,
            "fingerprint": fingerprint,
        }
        metadata = {**base_metadata, "projection": projection_marker}
        generations = {
            "draft_id": draft.id,
            "current_version_number": draft.current_version_number,
            "projection": projection_marker,
            "versions": generation_records,
        }
        counts = Counter(record["status"] for record in publication_records)
        succeeded_records = [
            record
            for record in publication_records
            if record["status"] == PublicationStatus.SUCCEEDED.value
        ]
        publication = {
            "draft_id": draft.id,
            "status": draft.status.value,
            "current_content_hash": draft.current_content_hash,
            "projection": projection_marker,
            "mock": bool(succeeded_records)
            and all(
                record.get("response_metadata", {}).get("mock") is True
                for record in publication_records
                if record["status"] == PublicationStatus.SUCCEEDED.value
            ),
            "post_ids": [
                record["x_post_id"]
                for record in succeeded_records
                if record["x_post_id"] is not None
            ],
            "published_at": next(
                (
                    record["published_at"]
                    for record in reversed(succeeded_records)
                    if record["published_at"] is not None
                ),
                None,
            ),
            "status_counts": dict(sorted(counts.items())),
            "has_failed_outcome": counts[PublicationStatus.FAILED.value] > 0,
            "has_ambiguous_outcome": counts[PublicationStatus.AMBIGUOUS.value] > 0,
            "records": publication_records,
        }
        review = self._review_text(draft, current, active, approval_events, fingerprint)
        return _Projection(
            draft=draft,
            version=current,
            content=content,
            metadata=metadata,
            sources=sources,
            generations=generations,
            review=review,
            publication=publication,
            fingerprint=fingerprint,
        )

    def _expected_texts(self, projection: _Projection) -> dict[str, str]:
        front_matter = self.artifacts.build_front_matter(projection.metadata)
        body = self.artifacts.render_content(projection.content)
        return {
            "draft.md": f"---\n{front_matter}\n---\n\n{body}\n",
            "metadata.json": _json_text(projection.metadata),
            "sources.json": _json_text(projection.sources),
            "generations.json": _json_text(projection.generations),
            "review.md": projection.review,
            "media_plan.json": _json_text(projection.draft.media_plan),
            "publication.json": _json_text(projection.publication),
        }

    def project(self, draft_id: str) -> ProjectionResult:
        """Atomically replace every DB-derived file for one existing bundle."""

        projection = self._build(self._draft(draft_id))
        directory = self.artifacts.locate(projection.draft.artifact_path)
        if not directory.is_dir():
            raise DraftArtifactError("Draft artifact directory does not exist")
        in_quarantine = self.artifacts.quarantine_root == directory or (
            self.artifacts.quarantine_root in directory.parents
        )
        if in_quarantine and projection.draft.status is not DraftStatus.DELETED:
            raise DraftArtifactError("Refusing to project into quarantine")

        self.artifacts.update_json(directory, "sources.json", projection.sources)
        self.artifacts.update_json(directory, "generations.json", projection.generations)
        self.artifacts.update_json(directory, "media_plan.json", projection.draft.media_plan)
        self.artifacts.update_json(directory, "publication.json", projection.publication)
        self.artifacts.update_json(directory, "metadata.json", projection.metadata)
        draft_files._atomic_write_text(directory / "review.md", projection.review)
        # The editable markdown is the final marker.  A crash before this point is
        # detected through the repeated projection fingerprint on the next scan.
        self.artifacts.write_markdown(
            directory,
            metadata=projection.metadata,
            content=projection.content,
        )
        return ProjectionResult(
            draft_id=draft_id,
            directory=directory,
            fingerprint=projection.fingerprint,
            files=tuple(self._expected_texts(projection)),
        )

    def rebuild(self, draft_id: str) -> ProjectionResult:
        """Rebuild an existing projection or safely recreate a missing empty-media bundle."""

        draft = self._draft(draft_id)
        directory = self.artifacts.locate(draft.artifact_path)
        if directory.is_dir():
            return self.project(draft_id)
        return self._recreate_bundle(draft)

    @staticmethod
    def _markdown_body(text: str) -> str | None:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if not normalized.startswith("---\n"):
            return None
        try:
            _, body = normalized[4:].split("\n---\n", maxsplit=1)
        except ValueError:
            return None
        return body.strip()

    def _projection_issues(self, draft: Draft, directory: Path) -> list[RecoveryIssue]:
        try:
            projection = self._build(draft)
        except (ArtifactProjectionError, ValueError, TypeError):
            return [
                RecoveryIssue(
                    kind=RecoveryIssueKind.INVALID_AUTHORITATIVE_STATE,
                    action=RecoveryAction.MANUAL_REVIEW,
                    draft_id=draft.id,
                    path=directory,
                    details=("authoritative_projection_unavailable",),
                )
            ]

        expected = self._expected_texts(projection)
        stale: list[str] = []
        content_divergence = False
        for filename, expected_text in expected.items():
            path = directory / filename
            try:
                actual = path.read_text(encoding="utf-8")
            except (FileNotFoundError, UnicodeDecodeError, OSError):
                stale.append(filename)
                continue
            if filename == "draft.md" and actual != expected_text:
                expected_body = self._markdown_body(expected_text)
                actual_body = self._markdown_body(actual)
                if actual_body is None or actual_body != expected_body:
                    content_divergence = True
                else:
                    stale.append(filename)
                continue
            if filename.endswith(".json"):
                try:
                    if json.loads(actual) != json.loads(expected_text):
                        stale.append(filename)
                except json.JSONDecodeError:
                    stale.append(filename)
            elif actual != expected_text:
                stale.append(filename)

        issues: list[RecoveryIssue] = []
        if content_divergence:
            issues.append(
                RecoveryIssue(
                    kind=RecoveryIssueKind.CONTENT_DIVERGENCE,
                    action=RecoveryAction.RECONCILE_CONTENT,
                    draft_id=draft.id,
                    path=directory,
                    details=("draft.md",),
                )
            )
        if stale:
            issues.append(
                RecoveryIssue(
                    kind=RecoveryIssueKind.STALE_PROJECTION,
                    action=RecoveryAction.PROJECT,
                    draft_id=draft.id,
                    path=directory,
                    details=tuple(sorted(set(stale))),
                )
            )
        return issues

    def _normal_directories(self) -> list[Path]:
        directories: list[Path] = []
        try:
            children = list(self.artifacts.root.iterdir())
        except OSError:
            return directories
        for date_directory in children:
            if (
                date_directory.name.startswith(".")
                or not date_directory.is_dir()
                or len(date_directory.name) != _DATE_DIRECTORY_LENGTH
            ):
                continue
            try:
                candidates = list(date_directory.iterdir())
            except OSError:
                continue
            directories.extend(path for path in candidates if path.is_dir())
        return directories

    def _quarantine_directories(self) -> list[Path]:
        root = self.artifacts.quarantine_root
        if not root.is_dir():
            return []
        directories: list[Path] = []
        for current, child_directories, files in os.walk(root, followlinks=False):
            child_directories[:] = [
                name for name in child_directories if not (Path(current) / name).is_symlink()
            ]
            if "draft.md" in files or "metadata.json" in files:
                path = Path(current)
                try:
                    resolved = path.resolve()
                except OSError:
                    continue
                if root in resolved.parents:
                    directories.append(resolved)
        return directories

    def scan_recovery(self) -> RecoveryReport:
        """Inspect all persisted drafts and bundles without changing either side."""

        drafts = list(self.session.scalars(select(Draft).order_by(Draft.id)).all())
        draft_by_id = {draft.id: draft for draft in drafts}
        expected_paths: dict[Path, Draft] = {}
        issues: list[RecoveryIssue] = []
        quarantine_by_id: dict[str, list[Path]] = {}
        for directory in self._quarantine_directories():
            quarantine_by_id.setdefault(directory.name, []).append(directory)

        for draft in drafts:
            try:
                path = self.artifacts.locate(draft.artifact_path)
            except (DraftArtifactError, OSError):
                issues.append(
                    RecoveryIssue(
                        kind=RecoveryIssueKind.INVALID_ARTIFACT_PATH,
                        action=RecoveryAction.MANUAL_REVIEW,
                        draft_id=draft.id,
                        path=self.artifacts.root,
                        details=("artifact_path_not_contained",),
                    )
                )
                continue
            expected_paths[path] = draft
            quarantined = quarantine_by_id.get(draft.id, [])
            path_is_quarantined = self.artifacts.quarantine_root == path or (
                self.artifacts.quarantine_root in path.parents
            )
            other_quarantined = [item for item in quarantined if item != path]
            if path.is_dir():
                if draft.status is DraftStatus.DELETED and not path_is_quarantined:
                    issues.append(
                        RecoveryIssue(
                            kind=RecoveryIssueKind.QUARANTINE_DIVERGENCE,
                            action=RecoveryAction.QUARANTINE_PRESERVING,
                            draft_id=draft.id,
                            path=path,
                            details=("deleted_db_row_has_active_bundle",),
                        )
                    )
                elif draft.status is not DraftStatus.DELETED and path_is_quarantined:
                    issues.append(
                        RecoveryIssue(
                            kind=RecoveryIssueKind.QUARANTINE_DIVERGENCE,
                            action=RecoveryAction.RESTORE_FROM_QUARANTINE,
                            draft_id=draft.id,
                            path=path,
                            details=("active_db_row_points_to_quarantined_bundle",),
                        )
                    )
                else:
                    issues.extend(self._projection_issues(draft, path))
                if other_quarantined:
                    issues.append(
                        RecoveryIssue(
                            kind=RecoveryIssueKind.QUARANTINE_DIVERGENCE,
                            action=RecoveryAction.MANUAL_REVIEW,
                            draft_id=draft.id,
                            path=other_quarantined[0],
                            details=("duplicate_active_and_quarantined_bundle",),
                        )
                    )
                continue

            if quarantined:
                if draft.status is not DraftStatus.DELETED:
                    issues.append(
                        RecoveryIssue(
                            kind=RecoveryIssueKind.QUARANTINE_DIVERGENCE,
                            action=RecoveryAction.RESTORE_FROM_QUARANTINE,
                            draft_id=draft.id,
                            path=quarantined[0],
                            details=("active_db_row_has_only_quarantined_bundle",),
                        )
                    )
                elif len(quarantined) > 1:
                    issues.append(
                        RecoveryIssue(
                            kind=RecoveryIssueKind.QUARANTINE_DIVERGENCE,
                            action=RecoveryAction.MANUAL_REVIEW,
                            draft_id=draft.id,
                            path=quarantined[0],
                            details=("multiple_quarantined_bundles",),
                        )
                    )
                continue

            issues.append(
                RecoveryIssue(
                    kind=RecoveryIssueKind.MISSING_BUNDLE,
                    action=(
                        RecoveryAction.MANUAL_REVIEW
                        if draft.status is DraftStatus.DELETED
                        else RecoveryAction.RECREATE_BUNDLE
                    ),
                    draft_id=draft.id,
                    path=path,
                    details=("artifact_directory_missing",),
                )
            )

        for directory in self._normal_directories():
            try:
                resolved = directory.resolve()
            except OSError:
                resolved = directory
            if resolved not in expected_paths:
                issues.append(
                    RecoveryIssue(
                        kind=RecoveryIssueKind.ORPHAN_BUNDLE,
                        action=RecoveryAction.QUARANTINE_PRESERVING,
                        draft_id=(directory.name if directory.name in draft_by_id else None),
                        path=directory,
                        details=("bundle_has_no_matching_artifact_path",),
                    )
                )

        for draft_id, directories in quarantine_by_id.items():
            if draft_id not in draft_by_id:
                for directory in directories:
                    issues.append(
                        RecoveryIssue(
                            kind=RecoveryIssueKind.ORPHAN_BUNDLE,
                            action=RecoveryAction.MANUAL_REVIEW,
                            draft_id=None,
                            path=directory,
                            details=("quarantined_bundle_has_no_db_row",),
                        )
                    )

        issues.sort(
            key=lambda issue: (
                issue.kind.value,
                issue.draft_id or "",
                str(issue.path),
                issue.details,
            )
        )
        return RecoveryReport(issues=tuple(issues))

    def _recreate_bundle(self, draft: Draft) -> ProjectionResult:
        projection = self._build(draft)
        directory = self.artifacts.locate(draft.artifact_path)
        canonical = self.artifacts.draft_directory(draft.id, draft.created_at)
        if directory != canonical:
            raise DraftArtifactError("Missing bundle path is not canonical")
        required_files = draft.media_plan.get("required_files", [])
        if required_files:
            raise DraftArtifactError("Missing bundle with required media cannot be reconstructed")
        created = self.artifacts.create_bundle(
            draft_id=draft.id,
            created_at=draft.created_at,
            metadata=projection.metadata,
            content=projection.content,
            sources=projection.sources,
            generations=projection.generations,
            review=projection.review,
            media_plan=draft.media_plan,
            publication=projection.publication,
        )
        return ProjectionResult(
            draft_id=draft.id,
            directory=created,
            fingerprint=projection.fingerprint,
            files=tuple(self._expected_texts(projection)),
        )

    def reconcile_startup(self) -> RecoveryReport:
        """Repair safe stale/missing projections, preserving every other artifact."""

        initial = self.scan_recovery()
        content_divergent = {
            issue.draft_id
            for issue in initial.issues
            if issue.kind is RecoveryIssueKind.CONTENT_DIVERGENCE
        }
        attempted: set[tuple[RecoveryAction, str]] = set()
        actions: list[RecoveryActionResult] = []
        for issue in initial.issues:
            if issue.draft_id is None:
                continue
            key = (issue.action, issue.draft_id)
            if key in attempted:
                continue
            if issue.action is RecoveryAction.PROJECT:
                if issue.draft_id in content_divergent:
                    continue
                try:
                    self.project(issue.draft_id)
                except (ArtifactProjectionError, DraftArtifactError, OSError, ValueError):
                    succeeded = False
                else:
                    succeeded = True
            elif issue.action is RecoveryAction.RECREATE_BUNDLE:
                try:
                    self._recreate_bundle(self._draft(issue.draft_id))
                except (ArtifactProjectionError, DraftArtifactError, OSError, ValueError):
                    succeeded = False
                else:
                    succeeded = True
            else:
                continue
            attempted.add(key)
            if not succeeded:
                actions.append(
                    RecoveryActionResult(
                        action=issue.action,
                        draft_id=issue.draft_id,
                        path=issue.path,
                        succeeded=False,
                        detail="safe_recovery_failed",
                    )
                )
            else:
                actions.append(
                    RecoveryActionResult(
                        action=issue.action,
                        draft_id=issue.draft_id,
                        path=issue.path,
                        succeeded=True,
                        detail="projection_rebuilt",
                    )
                )

        final = self.scan_recovery()
        return RecoveryReport(issues=final.issues, actions=tuple(actions))


__all__ = [
    "ArtifactProjectionError",
    "ArtifactProjectionService",
    "ProjectionResult",
    "RecoveryAction",
    "RecoveryActionResult",
    "RecoveryIssue",
    "RecoveryIssueKind",
    "RecoveryReport",
]
