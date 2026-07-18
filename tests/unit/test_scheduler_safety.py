"""Architecture-level proof that background jobs have no publication capability."""

from pathlib import Path

import pytest

from app.services.scheduler import ForbiddenSchedulerJobError, SchedulerService


def test_scheduler_module_does_not_import_publishing_service() -> None:
    source = Path(__file__).resolve().parents[2] / "app" / "services" / "scheduler.py"
    module_text = source.read_text(encoding="utf-8").casefold()
    assert "publishing_service" not in module_text
    assert "create_post" not in module_text


def test_scheduler_rejects_publish_job_even_if_configured() -> None:
    scheduler = SchedulerService({})
    with pytest.raises(ForbiddenSchedulerJobError):
        scheduler.add_interval_job("publish", minutes=1)
