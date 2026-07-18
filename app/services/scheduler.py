"""Allowlisted background jobs with no publication dependency or capability."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler


class SafeJob(StrEnum):
    SYNC_HOME = "sync_home"
    RECENT_SEARCH = "recent_search"
    UPDATE_METRICS = "update_metrics"
    SCAN_MENTIONS = "scan_mentions"
    CREATE_IDEAS = "create_ideas"
    GENERATE_DRAFTS = "generate_drafts"
    CHECK_VIDEO_JOBS = "check_video_jobs"


class ForbiddenSchedulerJobError(ValueError):
    pass


class SchedulerService:
    """Register only explicit read/generation jobs; arbitrary callables are rejected."""

    def __init__(self, handlers: Mapping[SafeJob, Callable[[], Any]]) -> None:
        unknown = set(handlers) - set(SafeJob)
        if unknown:
            raise ForbiddenSchedulerJobError("Scheduler handler is outside the safe allowlist")
        self.handlers = dict(handlers)
        self.scheduler = BackgroundScheduler(timezone="UTC")

    def add_interval_job(self, job: SafeJob | str, *, minutes: int) -> None:
        try:
            safe_job = SafeJob(job)
        except ValueError as exc:
            raise ForbiddenSchedulerJobError(
                "Scheduler job is forbidden; approval and writes are never background tasks"
            ) from exc
        handler = self.handlers.get(safe_job)
        if handler is None:
            raise ForbiddenSchedulerJobError("No safe handler is registered for this job")
        self.scheduler.add_job(
            handler,
            trigger="interval",
            minutes=minutes,
            id=safe_job.value,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

    def start(self) -> None:
        self.scheduler.start()

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
