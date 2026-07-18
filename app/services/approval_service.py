"""Hash- and version-scoped human approval gates."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.content import publishable_content
from app.domain.security import require_fact_check_for_approval, require_fact_check_for_publication
from app.domain.state_machine import assert_transition_allowed
from app.models import Draft, DraftApproval
from app.models.enums import ApprovalAction, DraftStatus, FactCheckStatus
from app.services.audit import AuditService
from app.services.draft_files import DraftArtifactStore
from app.services.draft_service import DraftService, DraftValidationError


class ApprovalGateError(PermissionError):
    pass


class ApprovalService:
    def __init__(
        self,
        session: Session,
        drafts: DraftService,
        artifacts: DraftArtifactStore,
    ) -> None:
        self.session = session
        self.drafts = drafts
        self.artifacts = artifacts
        self.audit = AuditService(session)

    def active_approval(self, draft: Draft) -> DraftApproval | None:
        approvals = self.session.scalars(
            select(DraftApproval)
            .where(
                DraftApproval.draft_id == draft.id,
                DraftApproval.action == ApprovalAction.APPROVED,
            )
            .order_by(DraftApproval.created_at.desc())
        ).all()
        for approval in approvals:
            revocation = self.session.scalar(
                select(DraftApproval.id).where(
                    DraftApproval.action == ApprovalAction.REVOKED,
                    DraftApproval.supersedes_id == approval.id,
                )
            )
            if revocation is None:
                return approval
        return None

    def revoke_active(self, draft: Draft, *, actor: str, reason: str) -> bool:
        approval = self.active_approval(draft)
        if approval is None:
            return False
        self.session.add(
            DraftApproval(
                draft_id=draft.id,
                draft_version_id=approval.draft_version_id,
                content_hash=approval.content_hash,
                action=ApprovalAction.REVOKED,
                actor=actor,
                reason=reason,
                supersedes_id=approval.id,
            )
        )
        self.audit.record(
            "approval_revoked",
            entity_type="draft",
            entity_id=draft.id,
            actor=actor,
            old_status=draft.status.value,
            new_status=DraftStatus.NEEDS_REVIEW.value,
            content_hash=draft.current_content_hash,
            metadata={"reason": reason},
        )
        self.session.flush()
        return True

    def verify_facts(
        self,
        draft_id: str,
        *,
        actor: str = "operator",
        expected_hash_prefix: str | None = None,
    ) -> Draft:
        """Persist fact verification independently from approval and safety gates."""

        self.drafts.reconcile(draft_id, actor=actor)
        draft = self.drafts.get(draft_id, lock=True)
        current_hash = str(draft.current_content_hash or "")
        if expected_hash_prefix and not current_hash.startswith(expected_hash_prefix):
            raise ApprovalGateError(
                "The draft changed after the fact-check prompt. Review the current version."
            )
        if draft.fact_check_status is FactCheckStatus.FAILED:
            raise ApprovalGateError("Fact checking failed for this draft")
        if draft.fact_check_status is FactCheckStatus.REQUIRED:
            draft.fact_check_status = FactCheckStatus.VERIFIED
            self.audit.record(
                "fact_check_verified",
                entity_type="draft",
                entity_id=draft.id,
                actor=actor,
                old_status=draft.status.value,
                new_status=draft.status.value,
                content_hash=current_hash or None,
                metadata={"fact_check_status": draft.fact_check_status.value},
            )
            self.session.flush()
        return draft

    def approve(
        self,
        draft_id: str,
        *,
        actor: str = "operator",
        facts_confirmed: bool = False,
    ) -> DraftApproval:
        self.drafts.reconcile(draft_id, actor=actor)
        draft = self.drafts.get(draft_id, lock=True)
        if draft.status is not DraftStatus.NEEDS_REVIEW:
            raise ApprovalGateError("Only a needs_review draft can be approved")
        if facts_confirmed and draft.fact_check_status is FactCheckStatus.REQUIRED:
            draft.fact_check_status = FactCheckStatus.VERIFIED
        require_fact_check_for_approval(draft.fact_check_status.value)
        if draft.blocking_safety_flags:
            rendered = ", ".join(draft.blocking_safety_flags)
            raise ApprovalGateError(
                "Blocking safety flags must be resolved before approval: " + rendered
            )
        version = self.drafts.current_version(draft)
        parts = tuple(version.content["parts"])
        canonical_content = publishable_content(parts, draft.content_type)
        current_hash = self.artifacts.compute_approval_hash(
            Path(draft.artifact_path),
            canonical_content,
            draft.media_plan,
        )
        if current_hash != draft.current_content_hash or current_hash != version.content_hash:
            raise ApprovalGateError("Draft content or media changed during approval")
        self.artifacts.validate_media_manifest(Path(draft.artifact_path), draft.media_plan)
        assert_transition_allowed(draft.status.value, DraftStatus.APPROVED.value)
        approval = DraftApproval(
            draft_id=draft.id,
            draft_version_id=version.id,
            content_hash=current_hash,
            action=ApprovalAction.APPROVED,
            actor=actor,
        )
        self.session.add(approval)
        draft.status = DraftStatus.APPROVED
        self.session.flush()
        self.audit.record(
            "approval",
            entity_type="draft",
            entity_id=draft.id,
            actor=actor,
            old_status=DraftStatus.NEEDS_REVIEW.value,
            new_status=draft.status.value,
            content_hash=current_hash,
            metadata={
                "version": version.version_number,
                "facts_confirmed": facts_confirmed,
                "fact_check_status": draft.fact_check_status.value,
            },
        )
        return approval

    def validate_for_publication(self, draft: Draft) -> DraftApproval:
        if draft.status is not DraftStatus.APPROVED:
            raise ApprovalGateError("Draft status is not approved")
        return self.validate_approval_integrity(draft)

    def validate_approval_integrity(self, draft: Draft) -> DraftApproval:
        """Revalidate a durable approval without weakening the initial status gate.

        A publication command moves an approved draft to
        ``publish_pending_confirmation`` before its first external call.  A process can
        therefore restart between successfully finalized thread positions.  Resuming that
        already-confirmed command must re-check every approval, content, fact, safety, and
        media invariant, while the initial call still goes through
        :meth:`validate_for_publication` and requires the exact ``approved`` status.
        """

        require_fact_check_for_publication(draft.fact_check_status.value)
        if draft.blocking_safety_flags:
            raise ApprovalGateError("Draft has blocking safety flags")
        approval = self.active_approval(draft)
        if approval is None:
            raise ApprovalGateError("No active approval exists")
        version = self.drafts.current_version(draft)
        if approval.draft_version_id != version.id:
            raise ApprovalGateError("Approval belongs to a different draft version")
        if not draft.current_content_hash:
            raise DraftValidationError("Draft has no current content hash")
        parts = tuple(version.content["parts"])
        canonical_content = publishable_content(parts, draft.content_type)
        current_hash = self.artifacts.compute_approval_hash(
            Path(draft.artifact_path), canonical_content, draft.media_plan
        )
        if not (
            current_hash
            == approval.content_hash
            == version.content_hash
            == draft.current_content_hash
        ):
            raise ApprovalGateError("Approval hash does not match current content and media")
        self.artifacts.validate_media_manifest(Path(draft.artifact_path), draft.media_plan)
        return approval
