from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.config import Settings
from app.container import build_services
from app.database import build_engine
from app.models import Base
from app.schemas.content import (
    EditorialIssue,
    EditorialReport,
    FactualClaimInspection,
    FactualInspectionReport,
)
from app.services.draft_service import (
    _reviewable_editorial_codes,
    _reviewable_factual_codes,
)
from app.services.publishing_service import PublishingAccountMismatchError, PublishingService


@pytest.mark.parametrize(
    "code",
    ("IMPLIED_POINT_RESTATEMENT", "MANUFACTURED_CONTRAST"),
)
def test_current_canonical_style_issues_are_reviewable(code: str) -> None:
    report = EditorialReport(
        status="block",
        source_overlap="partial",
        contribution_support="supported",
        information_gain="weak",
        internal_redundancy="none",
        source_stronger_than_draft=False,
        contribution_delivered=True,
        issues=(
            EditorialIssue(
                code=code,
                severity="block",
                evidence_spans=("A reviewable sentence.",),
                explanation="This is a style-level issue for human review.",
            ),
        ),
    )

    assert _reviewable_editorial_codes(report) == (code,)


def test_operator_factual_uncertainty_can_be_deferred_to_human_fact_check() -> None:
    report = FactualInspectionReport(
        applicable=True,
        status="block",
        claims=(
            FactualClaimInspection(
                claim="A compact opinion with a claim-parser boundary.",
                kind="material_claim_unknown",
                support="extraction_uncertain",
                issue_codes=("CLAIM_EXTRACTION_UNCERTAIN",),
                human_verification_required=True,
            ),
        ),
        approval_eligible=False,
        human_verification_required=True,
    )

    assert _reviewable_factual_codes(report) == ("CLAIM_EXTRACTION_UNCERTAIN",)


def test_invalid_evidence_reference_still_fails_closed() -> None:
    report = FactualInspectionReport(
        applicable=True,
        status="block",
        claims=(),
        invalid_evidence_ids=("missing-evidence",),
        approval_eligible=False,
    )

    assert _reviewable_factual_codes(report) is None


def test_publishing_account_mismatch_identifies_the_authenticated_writer() -> None:
    service = object.__new__(PublishingService)
    service.expected_account_id = "expected-123"

    with pytest.raises(PublishingAccountMismatchError) as captured:
        service._account({"id": "actual-456", "username": "writer"})

    assert captured.value.expected_id == "expected-123"
    assert captured.value.actual_id == "actual-456"
    assert captured.value.username == "writer"
    assert "@writer" in str(captured.value)
    assert "set X user ID to actual-456" in str(captured.value)


def test_home_timeline_reader_id_does_not_pin_the_oauth_writer(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        mock_mode=True,
        x_user_id="home-timeline-reader",
        database_url=f"sqlite:///{(tmp_path / 'workspace.db').as_posix()}",
        config_dir=Path("config"),
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        logs_dir=tmp_path / "logs",
    )
    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as session:
            services = build_services(session, settings)
            assert services.publishing.expected_account_id is None
    finally:
        engine.dispose()
