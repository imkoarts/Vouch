"""The sole deterministic gateway for crash-safe, human-confirmed X writes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Protocol

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.domain.content import content_parts, publishable_content
from app.domain.security import redact_string
from app.domain.state_machine import assert_transition_allowed
from app.models import Draft, DraftApproval, DraftVersion, Publication
from app.models.enums import (
    ApiDirection,
    ContentType,
    DraftStatus,
    PublicationKind,
    PublicationStatus,
)
from app.services.approval_service import ApprovalService
from app.services.audit import AuditService
from app.services.cost_service import CostService
from app.services.draft_service import DraftService
from app.x_api.mock import AmbiguousPublicationError, MockXClient

CONFIRMATION_HASH_PREFIX_LENGTH = 12
FaultStage = Literal[
    "after_claim_commit",
    "after_remote_acceptance",
    "after_finalize_commit",
]
PublicationFaultInjector = Callable[[FaultStage, int], None]


class XWritePort(Protocol):
    async def get_me(self) -> Mapping[str, Any]: ...

    async def upload_media(self, path: Path) -> str: ...

    async def create_post(
        self,
        *,
        text: str,
        reply_to_post_id: str | None,
        media_ids: tuple[str, ...] = (),
        quote_post_id: str | None = None,
        position: int = 0,
    ) -> Mapping[str, Any]: ...


class PublicationGateError(PermissionError):
    pass


class PublishingAccountMismatchError(PublicationGateError):
    """The configured read/publish identity differs from the authenticated writer."""

    def __init__(self, *, expected_id: str, actual_id: str, username: str) -> None:
        self.expected_id = expected_id
        self.actual_id = actual_id
        self.username = username
        writer = f"@{username}" if username else "the authenticated X writer"
        super().__init__(
            f"{writer} uses X user ID {actual_id}, but this workspace is configured for "
            f"X user ID {expected_id}. Open Settings > AI & Telegram > Credentials, set "
            f"X user ID to {actual_id}, save, and restart Vouch."
        )


class AmbiguousClaimError(PublicationGateError):
    pass


@dataclass(frozen=True, slots=True)
class PublicationPreview:
    draft_id: str
    content_hash: str
    confirmation_phrase: str
    parts: tuple[str, ...]
    weighted_lengths: tuple[int, ...]
    media_manifest: dict[str, Any]
    source: str | None
    facts: str
    estimated_cost_usd: Decimal | None
    account_id: str
    account_username: str
    action: str
    quote_post_id: str | None = None


@dataclass(frozen=True, slots=True)
class _PreparedClaim:
    claim_id: str
    text: str
    position: int
    parent_post_id: str | None
    total_parts: int
    media_files: tuple[Path, ...]
    quote_post_id: str | None = None


class PublishingService:
    """Validate gates and persist every manual write in three durable phases.

    Each thread position is prepared and committed as ``IN_FLIGHT`` before the
    writer is called.  The X writer then runs with no active SQL transaction,
    and its result is finalized in a second durable transaction.  An abrupt
    process death therefore leaves an explicit non-repeatable claim instead of
    losing the evidence that X might have accepted a post.

    A live writer is accepted only when the composition root explicitly enables
    manual live writes. Automatic publication remains architecturally forbidden.
    """

    def __init__(
        self,
        session: Session,
        drafts: DraftService,
        approvals: ApprovalService,
        writer: XWritePort,
        costs: CostService,
        *,
        publish_enabled: bool,
        expected_account_id: str | None = None,
        fault_injector: PublicationFaultInjector | None = None,
        live_writes_enabled: bool = False,
        enterprise_quote_posts_enabled: bool = False,
    ) -> None:
        if not isinstance(writer, MockXClient) and not live_writes_enabled:
            raise PublicationGateError("Live X writer requires explicit manual-write enablement")
        self.session = session
        self.drafts = drafts
        self.approvals = approvals
        self.writer = writer
        self.costs = costs
        self.publish_enabled = publish_enabled
        self.expected_account_id = expected_account_id
        self.fault_injector = fault_injector
        self.audit = AuditService(session)
        self.is_mock_writer = isinstance(writer, MockXClient)
        self.enterprise_quote_posts_enabled = enterprise_quote_posts_enabled

    @staticmethod
    def required_confirmation(draft: Draft) -> str:
        if not draft.current_content_hash:
            raise PublicationGateError("Draft has no content hash")
        prefix = draft.current_content_hash[:CONFIRMATION_HASH_PREFIX_LENGTH]
        return f"PUBLISH {draft.id} {prefix}"

    @staticmethod
    def _canonical_parts(draft: Draft, version: DraftVersion) -> tuple[str, ...]:
        raw_parts = version.content.get("parts")
        if not isinstance(raw_parts, list) or not all(isinstance(part, str) for part in raw_parts):
            raise PublicationGateError("Persisted publication content is malformed")
        return content_parts(publishable_content(raw_parts, draft.content_type))

    @staticmethod
    def _estimate_label(estimate: Decimal | None) -> str:
        return str(estimate) if estimate is not None else "not configured"

    def _estimate(self, *, units: int = 1) -> Decimal | None:
        return self.costs.estimate_optional("x_write", units=units)

    def _inject(self, stage: FaultStage, position: int) -> None:
        if self.fault_injector is not None:
            self.fault_injector(stage, position)

    def _account(self, payload: Mapping[str, Any]) -> tuple[str, str]:
        account_id = str(payload.get("id", ""))
        username = str(payload.get("username", ""))
        if not account_id:
            raise PublicationGateError("Publishing account could not be determined")
        if self.expected_account_id and account_id != self.expected_account_id:
            raise PublishingAccountMismatchError(
                expected_id=self.expected_account_id,
                actual_id=account_id,
                username=username,
            )
        return account_id, username

    def _claims_for(self, draft_id: str, content_hash: str) -> list[Publication]:
        return list(
            self.session.scalars(
                select(Publication)
                .where(
                    Publication.draft_id == draft_id,
                    Publication.content_hash == content_hash,
                )
                .order_by(Publication.thread_position)
            )
        )

    def _successful_prefix(
        self,
        draft: Draft,
        *,
        total_parts: int,
        account_id: str,
    ) -> list[Publication]:
        content_hash = draft.current_content_hash or ""
        claims = self._claims_for(draft.id, content_hash)
        unresolved = {
            PublicationStatus.CLAIMED,
            PublicationStatus.IN_FLIGHT,
            PublicationStatus.AMBIGUOUS,
        }
        if any(claim.status in unresolved for claim in claims):
            raise AmbiguousClaimError(
                "A durable publication claim has no safely repeatable outcome; "
                "manual reconciliation is required"
            )
        if any(claim.status is PublicationStatus.FAILED for claim in claims):
            raise PublicationGateError(
                "A failed publication claim requires explicit manual recovery"
            )
        if any(claim.status is not PublicationStatus.SUCCEEDED for claim in claims):
            raise PublicationGateError("Publication history contains a non-resumable state")
        if len(claims) > total_parts:
            raise PublicationGateError("Publication history has an invalid thread position")

        expected_positions = list(range(len(claims)))
        positions = [claim.thread_position for claim in claims]
        if positions != expected_positions:
            raise PublicationGateError(
                "Successful publication positions are not a contiguous prefix"
            )
        for index, claim in enumerate(claims):
            if not claim.x_post_id:
                raise AmbiguousClaimError(
                    "A successful publication claim is missing its remote post ID"
                )
            if claim.account_id != account_id:
                raise PublicationGateError("Publication history belongs to a different X account")
            if index > 0 and claim.parent_post_id != claims[index - 1].x_post_id:
                raise PublicationGateError("Thread publication parent chain is inconsistent")
        return claims

    def _validate_state(
        self,
        draft: Draft,
        *,
        successful_prefix: list[Publication],
    ) -> DraftApproval:
        if draft.status is DraftStatus.PUBLISH_PENDING_CONFIRMATION:
            if not successful_prefix:
                raise AmbiguousClaimError(
                    "A pending publication has no safely resumable successful position"
                )
            return self.approvals.validate_approval_integrity(draft)
        return self.approvals.validate_for_publication(draft)

    def _preview_from_account(
        self,
        draft: Draft,
        *,
        account_id: str,
        account_username: str,
    ) -> PublicationPreview:
        version = self.drafts.current_version(draft)
        source_context = version.generation_metadata.get("source_context", {})
        quote_post_id: str | None = None
        if isinstance(source_context, dict) and source_context.get("strategy") == "quote_post":
            raw_quote_id = str(source_context.get("source_post_id") or "")
            if not raw_quote_id.isdigit():
                raise PublicationGateError(
                    "This draft requires the original X post as quote context, but its source "
                    "post ID is missing or invalid."
                )
            if not getattr(self, "enterprise_quote_posts_enabled", False):
                raise PublicationGateError(
                    "This draft requires an X quote post. The official quote_tweet_id write "
                    "parameter is available only with X Enterprise access. Enable "
                    "publication.enterprise_quote_posts_enabled only for an eligible account, "
                    "or publish this draft manually from the X interface."
                )
            quote_post_id = raw_quote_id
        parts = self._canonical_parts(draft, version)
        if quote_post_id is not None and len(parts) != 1:
            raise PublicationGateError(
                "Quote-post publication supports exactly one commentary part"
            )
        successful = self._successful_prefix(draft, total_parts=len(parts), account_id=account_id)
        self._validate_state(draft, successful_prefix=successful)
        raw_lengths = version.content.get("weighted_lengths", [])
        if not isinstance(raw_lengths, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in raw_lengths
        ):
            raise PublicationGateError("Persisted weighted lengths are malformed")
        lengths = tuple(raw_lengths)
        if len(lengths) != len(parts):
            raise PublicationGateError("Persisted weighted lengths do not match content")
        media_manifest = self.drafts.artifacts.validate_media_manifest(
            Path(draft.artifact_path), draft.media_plan
        )
        if quote_post_id is not None and media_manifest.get("files"):
            raise PublicationGateError(
                "X quote posts cannot include uploaded media in the same request"
            )
        action = (
            "quote_post"
            if quote_post_id is not None
            else "reply"
            if draft.content_type is ContentType.REPLY
            else "thread"
            if draft.content_type is ContentType.THREAD
            else "post"
        )
        return PublicationPreview(
            draft_id=draft.id,
            content_hash=draft.current_content_hash or "",
            confirmation_phrase=self.required_confirmation(draft),
            parts=parts,
            weighted_lengths=lengths,
            media_manifest=media_manifest,
            source=str(draft.idea_id) if draft.idea_id else None,
            facts=draft.fact_check_status.value,
            estimated_cost_usd=self._estimate(units=len(parts)),
            account_id=account_id,
            account_username=account_username,
            action=action,
            quote_post_id=quote_post_id,
        )

    async def preview(self, draft_id: str) -> PublicationPreview:
        self.drafts.reconcile(draft_id, actor="operator")
        draft = self.drafts.get(draft_id)
        account_id, username = self._account(await self.writer.get_me())
        return self._preview_from_account(draft, account_id=account_id, account_username=username)

    @staticmethod
    def _payload_hash(
        *,
        content_hash: str,
        text: str,
        position: int,
        parent_id: str | None,
        account_id: str,
        media_manifest: dict[str, Any],
        quote_post_id: str | None = None,
    ) -> str:
        payload = json.dumps(
            {
                "account_id": account_id,
                "content_hash": content_hash,
                "text": text,
                "position": position,
                "parent_id": parent_id,
                "media_manifest": media_manifest,
                "quote_post_id": quote_post_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _publication_kind(content_type: ContentType) -> PublicationKind:
        if content_type is ContentType.REPLY:
            return PublicationKind.REPLY
        if content_type is ContentType.THREAD:
            return PublicationKind.THREAD_ITEM
        return PublicationKind.POST

    @staticmethod
    def _reply_parent(draft: Draft) -> str | None:
        if draft.content_type is not ContentType.REPLY:
            return None
        if draft.reply_candidate is None:
            raise PublicationGateError("Reply draft has no eligible source post")
        post_id = draft.reply_candidate.source_post.x_post_id
        if not post_id:
            raise PublicationGateError("Reply source has no X post ID")
        return post_id

    def _prepare_claim(
        self,
        preview: PublicationPreview,
        *,
        position: int,
        parent_post_id: str | None,
        confirmation_phrase: str,
        actor: str,
    ) -> _PreparedClaim:
        """Prepare and durably commit one ``IN_FLIGHT`` idempotency claim."""

        try:
            self.drafts.reconcile(preview.draft_id, actor=actor)
            draft = self.drafts.get(preview.draft_id, lock=True)
            if draft.current_content_hash != preview.content_hash:
                raise PublicationGateError("Draft hash changed before publication claim")
            if confirmation_phrase != self.required_confirmation(draft):
                raise PublicationGateError(
                    "Exact draft ID and hash-prefix confirmation is required"
                )
            version = self.drafts.current_version(draft)
            parts = self._canonical_parts(draft, version)
            if parts != preview.parts:
                raise PublicationGateError("Publishable content changed before claim")
            successful = self._successful_prefix(
                draft,
                total_parts=len(parts),
                account_id=preview.account_id,
            )
            approval = self._validate_state(draft, successful_prefix=successful)
            if position != len(successful):
                raise PublicationGateError("Publication position is not safely resumable")

            unit_estimate = self._estimate()
            self.costs.require_available(
                "x_write",
                estimated_next=unit_estimate,
                actor=actor,
            )
            request_hash = self._payload_hash(
                content_hash=preview.content_hash,
                text=parts[position],
                position=position,
                parent_id=parent_post_id,
                account_id=preview.account_id,
                media_manifest=preview.media_manifest,
                quote_post_id=(preview.quote_post_id if position == 0 else None),
            )
            old_status = draft.status
            if old_status is DraftStatus.APPROVED:
                assert_transition_allowed(
                    old_status.value,
                    DraftStatus.PUBLISH_PENDING_CONFIRMATION.value,
                )
                draft.status = DraftStatus.PUBLISH_PENDING_CONFIRMATION
                self.audit.record(
                    "publication_attempt",
                    entity_type="draft",
                    entity_id=draft.id,
                    actor=actor,
                    old_status=old_status.value,
                    new_status=draft.status.value,
                    content_hash=draft.current_content_hash,
                    metadata={
                        "account_id": preview.account_id,
                        "account_username": preview.account_username,
                        "action": preview.action,
                        "estimated_cost_usd": self._estimate_label(preview.estimated_cost_usd),
                    },
                )
                self.audit.record(
                    "publication_confirmed",
                    entity_type="draft",
                    entity_id=draft.id,
                    actor=actor,
                    old_status=draft.status.value,
                    new_status=draft.status.value,
                    content_hash=draft.current_content_hash,
                    metadata={
                        "approval_id": approval.id,
                        "account_id": preview.account_id,
                    },
                )
            else:
                self.audit.record(
                    "publication_resumed",
                    entity_type="draft",
                    entity_id=draft.id,
                    actor=actor,
                    old_status=draft.status.value,
                    new_status=draft.status.value,
                    content_hash=draft.current_content_hash,
                    metadata={"position": position, "account_id": preview.account_id},
                )

            claim = Publication(
                draft_id=draft.id,
                draft_version_id=version.id,
                content_hash=preview.content_hash,
                kind=self._publication_kind(draft.content_type),
                status=PublicationStatus.IN_FLIGHT,
                thread_position=position,
                request_payload_hash=request_hash,
                parent_post_id=parent_post_id,
                account_id=preview.account_id,
                estimated_cost_usd=unit_estimate,
                attempt_count=1,
            )
            self.session.add(claim)
            self.session.flush()
            self.audit.record(
                "publication_claimed",
                entity_type="publication",
                entity_id=claim.id,
                actor=actor,
                old_status=None,
                new_status=PublicationStatus.IN_FLIGHT.value,
                content_hash=preview.content_hash,
                metadata={
                    "draft_id": draft.id,
                    "position": position,
                    "account_id": preview.account_id,
                },
            )
            media_files = (
                self.drafts.artifacts.validated_media_files(
                    Path(draft.artifact_path), draft.media_plan
                )
                if position == 0
                else ()
            )
            prepared = _PreparedClaim(
                claim_id=claim.id,
                text=parts[position],
                position=position,
                parent_post_id=parent_post_id,
                total_parts=len(parts),
                media_files=tuple(media_files),
                quote_post_id=(preview.quote_post_id if position == 0 else None),
            )
            # This commit is the safety boundary: it must complete before any
            # create_post call is possible.
            self.session.commit()
            return prepared
        except IntegrityError as exc:
            self.session.rollback()
            raise AmbiguousClaimError("A concurrent idempotency claim already exists") from exc
        except Exception:
            self.session.rollback()
            raise

    def _finalize_failure(
        self,
        prepared: _PreparedClaim,
        *,
        status: PublicationStatus,
        error: str,
        actor: str,
    ) -> None:
        if status not in {PublicationStatus.FAILED, PublicationStatus.AMBIGUOUS}:
            raise ValueError("Only terminal failure states can be finalized here")
        try:
            claim = self.session.scalar(
                select(Publication).where(Publication.id == prepared.claim_id).with_for_update()
            )
            if claim is None or claim.status is not PublicationStatus.IN_FLIGHT:
                raise AmbiguousClaimError("Publication claim cannot be safely finalized")
            draft = self.drafts.get(claim.draft_id, lock=True)
            claim.status = status
            claim.error_details = redact_string(error)
            old_status = draft.status
            if old_status is DraftStatus.PUBLISH_PENDING_CONFIRMATION:
                assert_transition_allowed(old_status.value, DraftStatus.PUBLISH_FAILED.value)
                draft.status = DraftStatus.PUBLISH_FAILED
            self.costs.record_usage(
                provider="x",
                operation="x_write",
                direction=ApiDirection.WRITE,
                estimated_cost=claim.estimated_cost_usd,
                success=False,
            )
            action = (
                "publication_ambiguous"
                if status is PublicationStatus.AMBIGUOUS
                else "publication_failed"
            )
            self.audit.record(
                action,
                entity_type="draft",
                entity_id=draft.id,
                actor=actor,
                old_status=old_status.value,
                new_status=draft.status.value,
                content_hash=claim.content_hash,
                metadata={
                    "claim_id": claim.id,
                    "position": claim.thread_position,
                    "account_id": claim.account_id,
                },
            )
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise

    def _finalize_success(
        self,
        prepared: _PreparedClaim,
        *,
        response: Mapping[str, Any],
        post_id: str,
        actor: str,
    ) -> None:
        try:
            claim = self.session.scalar(
                select(Publication).where(Publication.id == prepared.claim_id).with_for_update()
            )
            if claim is None or claim.status is not PublicationStatus.IN_FLIGHT:
                raise AmbiguousClaimError("Publication claim cannot be safely finalized")
            draft = self.drafts.get(claim.draft_id, lock=True)
            claim.status = PublicationStatus.SUCCEEDED
            claim.x_post_id = post_id
            response_metadata: dict[str, Any] = {"mock": bool(response.get("mock"))}
            rate_limit = response.get("rate_limit")
            if isinstance(rate_limit, Mapping):
                response_metadata["rate_limit"] = {
                    str(key): str(value)
                    for key, value in rate_limit.items()
                    if str(key) in {"limit", "remaining", "reset"}
                }
            claim.response_metadata = response_metadata
            claim.published_at = datetime.now(UTC)
            self.costs.record_usage(
                provider="x",
                operation="x_write",
                direction=ApiDirection.WRITE,
                estimated_cost=claim.estimated_cost_usd,
            )
            self.audit.record(
                "publication_part_succeeded",
                entity_type="publication",
                entity_id=claim.id,
                actor=actor,
                old_status=PublicationStatus.IN_FLIGHT.value,
                new_status=PublicationStatus.SUCCEEDED.value,
                content_hash=claim.content_hash,
                metadata={
                    "draft_id": draft.id,
                    "position": claim.thread_position,
                    "account_id": claim.account_id,
                },
            )

            claims = self._claims_for(draft.id, claim.content_hash)
            all_succeeded = (
                len(claims) == prepared.total_parts
                and [item.thread_position for item in claims] == list(range(prepared.total_parts))
                and all(item.status is PublicationStatus.SUCCEEDED for item in claims)
            )
            if all_succeeded and draft.current_content_hash == claim.content_hash:
                old_status = draft.status
                assert_transition_allowed(old_status.value, DraftStatus.PUBLISHED.value)
                draft.status = DraftStatus.PUBLISHED
                self.audit.record(
                    "publication_result",
                    entity_type="draft",
                    entity_id=draft.id,
                    actor=actor,
                    old_status=old_status.value,
                    new_status=draft.status.value,
                    content_hash=claim.content_hash,
                    metadata={"mock": self.is_mock_writer, "parts": len(claims)},
                )
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise

    def _complete_from_succeeded_history(
        self,
        preview: PublicationPreview,
        *,
        actor: str,
    ) -> None:
        """Finish local state only when every remote position is already durable."""

        try:
            draft = self.drafts.get(preview.draft_id, lock=True)
            claims = self._successful_prefix(
                draft,
                total_parts=len(preview.parts),
                account_id=preview.account_id,
            )
            self.approvals.validate_approval_integrity(draft)
            if len(claims) != len(preview.parts):
                raise PublicationGateError("Publication history is incomplete")
            old_status = draft.status
            assert_transition_allowed(old_status.value, DraftStatus.PUBLISHED.value)
            draft.status = DraftStatus.PUBLISHED
            self.audit.record(
                "publication_result_recovered",
                entity_type="draft",
                entity_id=draft.id,
                actor=actor,
                old_status=old_status.value,
                new_status=draft.status.value,
                content_hash=preview.content_hash,
                metadata={"mock": self.is_mock_writer, "parts": len(claims)},
            )
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise

    async def publish(
        self,
        draft_id: str,
        *,
        confirmation_phrase: str,
        actor: str = "operator",
    ) -> list[Publication]:
        if not self.publish_enabled:
            raise PublicationGateError("PUBLISH_ENABLED is false")

        self.drafts.reconcile(draft_id, actor=actor)
        draft = self.drafts.get(draft_id, lock=True)
        if draft.status not in {
            DraftStatus.PUBLISH_PENDING_CONFIRMATION,
            DraftStatus.PUBLISH_FAILED,
        }:
            # Preserve the primary approval gate before inspecting a malformed
            # confirmation.  Resumable/failed commands are classified from their
            # durable claim history below.
            self.approvals.validate_for_publication(draft)
        if confirmation_phrase != self.required_confirmation(draft):
            raise PublicationGateError("Exact draft ID and hash-prefix confirmation is required")
        account_id, username = self._account(await self.writer.get_me())
        preview = self._preview_from_account(
            draft, account_id=account_id, account_username=username
        )
        successful = self._successful_prefix(
            draft,
            total_parts=len(preview.parts),
            account_id=preview.account_id,
        )
        successful_ids = {claim.thread_position: claim.x_post_id for claim in successful}
        parent_post_id = self._reply_parent(draft)

        for position, _text in enumerate(preview.parts):
            succeeded_id = successful_ids.get(position)
            if succeeded_id:
                parent_post_id = succeeded_id
                continue
            prepared = self._prepare_claim(
                preview,
                position=position,
                parent_post_id=parent_post_id,
                confirmation_phrase=confirmation_phrase,
                actor=actor,
            )
            self._inject("after_claim_commit", position)
            if self.session.in_transaction():
                raise RuntimeError("Publication writer cannot run inside a database transaction")
            try:
                media_ids = tuple(
                    [
                        await self.writer.upload_media(media_file)
                        for media_file in prepared.media_files
                    ]
                )
                if prepared.quote_post_id is None:
                    response = await self.writer.create_post(
                        text=prepared.text,
                        reply_to_post_id=prepared.parent_post_id,
                        media_ids=media_ids,
                        position=prepared.position,
                    )
                else:
                    response = await self.writer.create_post(
                        text=prepared.text,
                        reply_to_post_id=prepared.parent_post_id,
                        media_ids=media_ids,
                        quote_post_id=prepared.quote_post_id,
                        position=prepared.position,
                    )
            except AmbiguousPublicationError as exc:
                self._finalize_failure(
                    prepared,
                    status=PublicationStatus.AMBIGUOUS,
                    error=str(exc),
                    actor=actor,
                )
                raise AmbiguousClaimError(
                    "X publication outcome is ambiguous; automatic retry is prohibited"
                ) from exc
            except Exception as exc:
                self._finalize_failure(
                    prepared,
                    status=PublicationStatus.FAILED,
                    error=str(exc),
                    actor=actor,
                )
                raise

            post_id = str(response.get("id", ""))
            if not post_id:
                self._finalize_failure(
                    prepared,
                    status=PublicationStatus.AMBIGUOUS,
                    error="X writer response did not contain a post ID",
                    actor=actor,
                )
                raise AmbiguousClaimError(
                    "X writer response is incomplete; automatic retry is prohibited"
                )
            self._inject("after_remote_acceptance", position)
            self._finalize_success(
                prepared,
                response=response,
                post_id=post_id,
                actor=actor,
            )
            parent_post_id = post_id
            successful_ids[position] = post_id
            self._inject("after_finalize_commit", position)

        if successful_ids and len(successful_ids) == len(preview.parts):
            refreshed = self.drafts.get(draft_id)
            if refreshed.status is DraftStatus.PUBLISH_PENDING_CONFIRMATION:
                self._complete_from_succeeded_history(preview, actor=actor)

        return self._claims_for(draft_id, preview.content_hash)
