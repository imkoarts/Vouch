"""Offline integration coverage for claim extraction, support policy, and edit gates."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.container import build_mock_services
from app.database import build_engine
from app.domain.enums import GenerationMode
from app.models import AuditEvent, Base, DraftVersion
from app.schemas.content import EvidenceItem, EvidencePacket, GenerationVariant
from app.services.claim_extraction import DeterministicClaimExtractor
from app.services.claim_structure import compare_claim_sets, extract_claims
from app.services.factual_inspection import inspect_factual_integrity
from app.services.idea_collector import IdeaCollector
from app.services.semantic_classifier import DeterministicSemanticInspector

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _LiveClaimExtractor:
    def __init__(self) -> None:
        self.delegate = DeterministicClaimExtractor()

    async def extract_claims(self, text: str, *, context):
        return await self.delegate.extract_claims(text, context=context)


class _LiveSemanticInspector:
    def __init__(self) -> None:
        self.delegate = DeterministicSemanticInspector()

    async def inspect_source(self, packet):
        return await self.delegate.inspect_source(packet)

    async def inspect_contributions(self, packet, contributions, source_map):
        return await self.delegate.inspect_contributions(packet, contributions, source_map)

    async def inspect_draft(self, packet, contribution, text, source_map, *, commentary_mode):
        return await self.delegate.inspect_draft(
            packet, contribution, text, source_map, commentary_mode=commentary_mode
        )


def _packet(text: str) -> EvidencePacket:
    return EvidencePacket(
        mode=GenerationMode.SOURCE_POST,
        topic="claim hardening",
        language="en",
        items=(EvidenceItem(evidence_id="e1", source_type="x_post", text=text),),
    )


def _variant(text: str) -> GenerationVariant:
    return GenerationVariant(
        label="selected",
        text=text,
        similarity_risk=0.0,
        angle_id="a1",
        angle_type="plain_update",
        evidence_ids=("e1",),
        confidence="high",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
    )


def test_multi_claim_source_supports_one_accurately_selected_numeric_claim() -> None:
    report = inspect_factual_integrity(
        _variant("Revenue increased to 10 million."),
        packet=_packet("Revenue increased to 10 million. Users increased to 20 million."),
        factual_inspection_required=True,
    )

    assert report.status == "pass"
    assert report.claims[0].support == "supported"


def test_unknown_verb_material_claim_fails_closed_in_commentary() -> None:
    report = inspect_factual_integrity(
        _variant("Alice brokered the Acme sale."),
        packet=_packet("Alice discussed Acme."),
        factual_inspection_required=False,
    )

    assert report.status == "block"
    assert report.claims[0].support == "extraction_uncertain"
    assert "CLAIM_EXTRACTION_UNCERTAIN" in report.claims[0].issue_codes


def test_attributed_source_cannot_become_unattributed_assertion() -> None:
    report = inspect_factual_integrity(
        _variant("Alice signed the bill."),
        packet=_packet("According to Bob, Alice signed the bill."),
        factual_inspection_required=True,
    )

    assert report.status == "block"
    assert report.human_verification_required is True
    assert "ATTRIBUTION_LOST" in report.claims[0].issue_codes


def test_edit_diff_and_current_evidence_check_block_modality_change() -> None:
    previous = extract_claims("Alice may sign the bill.")
    current = extract_claims("Alice signed the bill.")
    changes = compare_claim_sets(previous, current)
    report = inspect_factual_integrity(
        _variant("Alice signed the bill."),
        packet=_packet("Alice may sign the bill."),
        factual_inspection_required=True,
        previous_claims=previous,
        extracted_claims=current,
    )

    assert any(
        change.change_type == "changed" and "modality" in change.fields_changed
        for change in changes
    )
    assert report.status == "block"
    assert report.approval_eligible is False


def test_mock_offline_unknown_material_claim_is_not_treated_as_opinion() -> None:
    claim = extract_claims("Acme brokered the arrangement.")[0]
    report = inspect_factual_integrity(
        _variant(claim.exact_text),
        packet=_packet("Acme announced an arrangement."),
        extracted_claims=(claim,),
    )

    assert claim.kind == "material_claim_unknown"
    assert report.status == "block"
    assert report.human_verification_required is True


@pytest.mark.asyncio
async def test_live_mode_edit_requires_completed_semantic_reinspection(tmp_path: Path) -> None:
    engine = build_engine(f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    settings = Settings(
        app_env="test",
        mock_mode=True,
        publish_enabled=True,
        database_url=str(engine.url),
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        config_dir=PROJECT_ROOT / "config",
    )
    services = build_mock_services(session, settings)
    try:
        idea_id = IdeaCollector(
            session, source_configuration=services.configuration.sources
        ).collect_mock()[0]
        draft = await services.drafts.generate_from_idea(idea_id)
        assert draft is not None
        session.commit()
        services.approvals.approve(draft.id)

        # Reconciliation is synchronous. In a live deployment it must fail closed until the
        # configured structured semantic inspector has completed on the edited version.
        services.drafts.semantic_inspector = _LiveSemanticInspector()
        services.drafts.claim_extractor = _LiveClaimExtractor()
        markdown = Path(draft.artifact_path) / "draft.md"
        markdown.write_text(
            markdown.read_text(encoding="utf-8") + "\nManual clarification.\n",
            encoding="utf-8",
        )

        assert services.drafts.reconcile(draft.id) is True
        current = session.scalar(
            select(DraftVersion)
            .where(DraftVersion.draft_id == draft.id)
            .order_by(DraftVersion.version_number.desc())
        )
        assert current is not None
        factual = current.generation_metadata["factual_inspection"]
        assert factual["semantic_reinspection_required"] is True
        assert factual["approval_eligible"] is False
        assert (
            "SEMANTIC_REINSPECTION_REQUIRED" in services.drafts.get(draft.id).blocking_safety_flags
        )
        assert services.approvals.active_approval(services.drafts.get(draft.id)) is None

        version_number = current.version_number
        content_hash = services.drafts.get(draft.id).current_content_hash
        reinspected = await services.drafts.reinspect_current_version(draft.id, actor="test")
        session.flush()
        refreshed = services.drafts.current_version(reinspected)
        completed = refreshed.generation_metadata["factual_inspection"]
        assert refreshed.version_number == version_number + 1
        assert refreshed.origin == "semantic_reinspection"
        assert refreshed.content_hash == content_hash
        assert reinspected.current_content_hash == content_hash
        assert completed["semantic_reinspection_required"] is False
        assert "SEMANTIC_REINSPECTION_REQUIRED" not in reinspected.blocking_safety_flags
        assert services.approvals.active_approval(reinspected) is None
        assert (
            session.scalar(
                select(AuditEvent).where(
                    AuditEvent.entity_id == draft.id,
                    AuditEvent.action == "semantic_reinspection_completed",
                )
            )
            is not None
        )
        same = await services.drafts.reinspect_current_version(draft.id, actor="test")
        assert same.current_version_number == refreshed.version_number
    finally:
        session.close()
