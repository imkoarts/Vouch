from __future__ import annotations

from sqlalchemy.exc import OperationalError

from app.domain.enums import PostLengthMode
from app.schemas.content import QualityReport
from app.services.draft_service import PostLengthMismatchError
from app.services.post_length import resolve_post_length
from app.utils.errors import operator_message, safe_exception_summary


def _operational(message: str) -> OperationalError:
    return OperationalError("SELECT 1", {}, RuntimeError(message))


def test_missing_schema_error_points_to_one_click_launcher() -> None:
    rendered = operator_message(_operational("no such table: ideas"), "ERR123")

    assert "ERR123" in rendered
    assert "START_VOUCH.bat" in rendered
    assert "no such table" in rendered


def test_locked_database_error_explains_duplicate_process() -> None:
    rendered = operator_message(_operational("database is locked"), "LOCK123")

    assert "second bot instance" in rendered
    assert "logs/vouch.log" not in rendered


def test_error_summary_redacts_embedded_token() -> None:
    token = "1234567890:" + "abcdefghijklmnopqrstuvwxyz_ABCDE"
    rendered = safe_exception_summary(RuntimeError(f"failed {token}"))

    assert token not in rendered
    assert "REDACTED" in rendered


def test_insufficient_quota_error_explains_that_delay_will_not_help() -> None:
    from app.providers.responses_provider import ProviderInsufficientQuotaError

    rendered = operator_message(
        ProviderInsufficientQuotaError("openai", cooldown_seconds=3600),
        "QUOTA123",
    )

    assert "QUOTA123" in rendered
    assert "billing" in rendered
    assert "Request delays" in rendered
    assert "paid X reads" in rendered


def test_structured_output_error_explains_ai_only_retry() -> None:
    from app.providers.responses_provider import ProviderStructuredOutputError

    rendered = operator_message(
        ProviderStructuredOutputError("openai", attempts=2),
        "STRUCT123",
    )

    assert "STRUCT123" in rendered
    assert "without reading X again" in rendered
    assert "2" in rendered


def test_length_mismatch_error_explains_bounded_ai_only_revision() -> None:
    report = QualityReport(
        passed=True,
        thesis_score=1.0,
        specificity_score=1.0,
        evidence_score=1.0,
        naturalness_score=1.0,
        plainness_score=1.0,
        rhetorical_density_score=0.0,
        reply_fit_score=1.0,
        distinctiveness_score=1.0,
    )
    rendered = operator_message(
        PostLengthMismatchError(
            selection=resolve_post_length(PostLengthMode.MEDIUM),
            lengths=(210, 225, 240),
            reports=(report, report, report),
        ),
        "LENGTH123",
    )

    assert "LENGTH123" in rendered
    assert "bounded revision" in rendered
    assert "X was not read again" in rendered


def test_fact_check_block_is_rendered_as_operator_action_not_processing_error() -> None:
    from app.domain.enums import FactCheckStatus
    from app.domain.security import FactCheckBlockedError

    rendered = operator_message(
        FactCheckBlockedError(FactCheckStatus.REQUIRED, "approval"),
        "FACT123",
    )

    assert "FACT123" in rendered
    assert "verification button" in rendered
    assert "nothing was published" in rendered
    assert "Processing error" not in rendered


def test_x_permission_error_is_rendered_as_actionable_publish_failure() -> None:
    from app.x_api.live import XPermissionError

    error = XPermissionError(
        "X denied manual publication with HTTP 403. Set the app to Read and write "
        "and regenerate the token.",
        status_code=403,
        request_id="req-403",
    )
    rendered = operator_message(error, "XWRITE403")

    assert "XWRITE403" in rendered
    assert "No post was published" in rendered
    assert "CONFIGURE_VOUCH.bat" in rendered
    assert "req-403" in rendered
    assert "Processing error" not in rendered
