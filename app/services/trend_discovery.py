"""Automatic, budget-bounded discovery from official X sources.

The private Creator Inspiration web UI is never scraped. The runtime combines
official trends, the authenticated home timeline, and explicitly selected account
timelines. Recent Search is intentionally excluded. Every run is capped before any request.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.container import build_services
from app.database import session_scope
from app.domain.enums import ApiDirection, ContentSourceKind, ContentType, IdeaStatus
from app.models import (
    AuditEvent,
    ContentSource,
    Idea,
    IdeaSource,
    SourcePost,
    SourcePostOccurrence,
)
from app.schemas.configuration import RequestPacingConfiguration
from app.services.audit import AuditService
from app.services.configuration import ConfigurationService
from app.services.cost_service import BudgetExceededError, CostService
from app.services.draft_service import (
    DraftValidationError,
    EditorialQualityError,
    MediaContextUnavailableError,
    PostLengthMismatchError,
)
from app.services.generation_pipeline import InsufficientContextError
from app.services.idea_collector import IdeaCollector
from app.services.reference_selection import assess_reference_eligibility
from app.utils.errors import (
    new_error_id,
    operator_message,
    safe_exception_summary,
    safe_traceback,
)
from app.utils.request_pacing import quota_cooldown_remaining, serialized_operation
from app.x_api.read import (
    XReadAuthenticationError,
    XReadClient,
    XReadError,
    XReadPaymentRequiredError,
    XReadRateLimitError,
    XReadTemporaryError,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DiscoveryOutcome:
    status: str
    draft_id: str | None
    idea_id: str | None
    fetched_posts: int
    home_posts: int
    search_posts: int
    trends: tuple[str, ...]
    estimated_x_cost_usd: Decimal
    message: str
    operator_notices: tuple[str, ...] = ()


def _secret(value: object) -> str:
    getter = getattr(value, "get_secret_value", None)
    return str(getter()) if callable(getter) else ""


def build_x_read_client(
    settings: Settings,
    *,
    request_pacing: RequestPacingConfiguration | None = None,
) -> XReadClient:
    pacing = request_pacing
    return XReadClient(
        base_url=settings.x_api_base_url,
        auth_mode=settings.x_auth_mode,
        bearer_token=_secret(settings.x_bearer_token),
        access_token=_secret(settings.x_access_token),
        consumer_key=_secret(settings.x_consumer_key),
        consumer_secret=_secret(settings.x_consumer_secret),
        access_token_secret=_secret(settings.x_access_token_secret),
        timeout_seconds=settings.request_timeout_seconds,
        proxy_url=settings.outbound_proxy_url,
        temporary_error_max_retries=int(getattr(pacing, "x_temporary_error_max_retries", 2)),
        temporary_error_initial_backoff_seconds=float(
            getattr(pacing, "x_temporary_error_initial_backoff_seconds", 2.0)
        ),
        temporary_error_max_backoff_seconds=float(
            getattr(pacing, "x_temporary_error_max_backoff_seconds", 10.0)
        ),
    )


def _read_diagnostic(scope: str, error: XReadError) -> str:
    if isinstance(error, XReadPaymentRequiredError):
        return f"{scope}: add X API credits in console.x.com"
    if isinstance(error, XReadAuthenticationError):
        return f"{scope}: check X OAuth credentials and app permissions"
    if isinstance(error, XReadRateLimitError):
        return f"{scope}: X API rate limit reached"
    if isinstance(error, XReadTemporaryError):
        status = f"HTTP {error.status_code}" if error.status_code is not None else "transport error"
        return f"{scope}: X API temporarily unavailable after {error.attempts} attempts ({status})"
    return f"{scope}: {type(error).__name__}"


def _trend_query(name: str) -> str:
    normalized = " ".join(name.strip().split())
    if not normalized:
        raise ValueError("Trend name is empty")
    if re.fullmatch(r"#[\w_]{1,100}", normalized, flags=re.UNICODE):
        term = normalized
    else:
        escaped = normalized.replace("\\", "\\\\").replace('"', '\\"')
        term = f'"{escaped[:180]}"'
    return f"{term} -is:retweet -is:reply"


def _generation_rejection_diagnostic(
    error: (
        DraftValidationError
        | EditorialQualityError
        | PostLengthMismatchError
        | InsufficientContextError
    ),
) -> str:
    if isinstance(error, EditorialQualityError):
        codes = sorted({issue.code for report in error.assessments for issue in report.issues})
        return "editorial gate: " + (", ".join(codes) or "no acceptable variant")
    if isinstance(error, PostLengthMismatchError):
        actual = ", ".join(str(length) for length in error.lengths)
        return (
            f"length gate: required {error.selection.minimum}-{error.selection.maximum}; "
            f"actual {actual}"
        )
    if isinstance(error, DraftValidationError):
        if "factual evidence validation" in str(error).casefold():
            return "factual gate: deterministic evidence validation failed"
        return "draft gate: deterministic validation failed"
    return f"context gate: {error}"


class AutomaticDiscoveryService:
    """Fetch a tiny source sample, rank it, and generate one review draft."""

    def __init__(self, settings: Settings, *, reader: XReadClient | None = None) -> None:
        self.settings = settings
        self.configuration = ConfigurationService(settings.config_dir).load()
        self.discovery = self.configuration.sources.automatic_discovery
        self.request_pacing = self.configuration.runtime.request_pacing
        self.reader = reader or build_x_read_client(settings, request_pacing=self.request_pacing)

    @staticmethod
    def _create_editorial_ideas(
        session: Session,
        candidates: list[Idea],
        *,
        trends: tuple[str, ...],
        max_sources: int = 5,
    ) -> tuple[str, ...]:
        """Create bounded single-anchor editorial ideas in ranked source order.

        A rejected generation must not pin every later discovery run to the same source. The
        caller may try the next already-read candidate without repeating X reads.
        """

        candidate_ids = [candidate.id for candidate in candidates]
        raw_rows = session.execute(
            select(IdeaSource, SourcePost)
            .join(SourcePost, SourcePost.id == IdeaSource.source_post_id)
            .where(IdeaSource.idea_id.in_(candidate_ids))
        ).all()
        rows: list[tuple[IdeaSource, SourcePost, float, str]] = []
        for source_link, post in raw_rows:
            eligibility = assess_reference_eligibility(post.text)
            if not eligibility.eligible:
                continue
            selection_score = (
                0.65 * float(source_link.contribution) + 0.35 * eligibility.utility_score
            )
            rows.append((source_link, post, selection_score, eligibility.editorial_intent))
        rows.sort(key=lambda item: (-item[2], item[1].id))
        created: list[str] = []
        seen_sources: set[str] = set()
        trend_label = ", ".join(trends[:3]) or "home timeline"
        for source_link, post, selection_score, editorial_intent in rows[:max_sources]:
            source_key = post.x_post_id or post.id
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            dedupe_key = hashlib.sha256(f"editorial-anchor:{source_key}".encode()).hexdigest()
            existing = session.scalar(select(Idea).where(Idea.dedupe_key == dedupe_key))
            if existing is not None:
                if existing.status is not IdeaStatus.ARCHIVED and not existing.drafts:
                    created.append(existing.id)
                continue

            has_media = bool(
                post.media_metadata.get("items") or post.media_metadata.get("media_keys")
            )
            summary = (
                "Write one personal X post from this single source anchor. Use the source as "
                "internal evidence, not as the narrator of the post. Do not mention the source "
                "account, describe what the account posted, or combine it with another feed item. "
                "Develop one thesis from the anchor itself. If source media supplies the context, "
                "preserve that context through an attached source-media asset or quote-post plan "
                "rather than explaining the source."
            )
            idea = Idea(
                status=IdeaStatus.RANKED,
                title=(post.text.splitlines()[0] or "Single-source editorial anchor")[:500],
                summary=summary,
                recommended_format=ContentType.SHORT_POST,
                score=max(0.0, selection_score),
                score_components={
                    "single_source_anchor": 1.0,
                    "source_count": 1.0,
                    "anchor_score": float(source_link.contribution),
                    "reference_utility_score": selection_score,
                },
                explanation={
                    "origin": "automatic_discovery_single_source",
                    "editorial_intent": editorial_intent,
                    "why_selected": "highest-ranked individual source signal",
                    "differentiation": (
                        "develop one source-specific thought; never collage feed items"
                    ),
                    "trends": list(trends[:3]),
                    "source_count": 1,
                    "anchor_source_post_id": post.id,
                    "anchor_x_post_id": post.x_post_id,
                    "anchor_url": post.url,
                    "anchor_author_username": post.username,
                    "anchor_has_media": has_media,
                    "discovery_context": trend_label,
                },
                dedupe_key=dedupe_key,
            )
            session.add(idea)
            session.flush()
            session.add(
                IdeaSource(
                    idea_id=idea.id,
                    source_post_id=post.id,
                    source_priority=source_link.source_priority,
                    contribution=source_link.contribution,
                    position=0,
                    link_metadata={
                        "origin": "automatic_discovery_single_source",
                        "source_labels": post.source_labels,
                        "public_metrics": post.public_metrics,
                        "username": post.username,
                        "url": post.url,
                        "has_media": has_media,
                    },
                )
            )
            session.flush()
            created.append(idea.id)
        return tuple(created)

    @staticmethod
    def _create_editorial_idea(
        session: Session,
        candidates: list[Idea],
        *,
        trends: tuple[str, ...],
        max_sources: int = 5,
    ) -> str | None:
        """Compatibility wrapper returning the first bounded editorial idea."""

        ids = AutomaticDiscoveryService._create_editorial_ideas(
            session, candidates, trends=trends, max_sources=max_sources
        )
        return ids[0] if ids else None

    def _runs_today(self) -> int:
        start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        completed_status = AuditEvent.event_metadata["status"].as_string()
        with session_scope() as session:
            value = session.scalar(
                select(func.count(AuditEvent.id)).where(
                    AuditEvent.action == "automatic_discovery_completed",
                    AuditEvent.created_at >= start,
                    completed_status.in_(("draft_created", "no_candidate")),
                )
            )
            return int(value or 0)

    def _costs(self) -> tuple[Decimal, Decimal, Decimal]:
        costs = self.configuration.costs.x
        if costs.read_usd is None or costs.trends_request_usd is None:
            raise RuntimeError("X read/trends costs must be configured before automatic discovery")
        sources = self.configuration.sources.sources
        post_units = 0
        if sources.home_timeline.enabled:
            post_units += sources.home_timeline.max_posts
        if sources.selected_accounts.enabled:
            active_accounts = sum(
                1 for account in sources.selected_accounts.tracked_accounts if account.enabled
            )
            post_units += active_accounts * sources.selected_accounts.max_posts_per_account
        maximum = costs.read_usd * post_units
        trend_attempts = self.request_pacing.x_temporary_error_max_retries + 1
        maximum += costs.trends_request_usd * trend_attempts
        return costs.read_usd, costs.trends_request_usd, maximum

    def _preflight_budget(self, maximum: Decimal) -> None:
        if self.settings.daily_x_read_limit_usd is None:
            raise RuntimeError("DAILY_X_READ_LIMIT_USD is required for automatic discovery")
        with session_scope() as session:
            CostService(
                session,
                limits={"x_read": Decimal(str(self.settings.daily_x_read_limit_usd))},
            ).require_available(
                "x_read",
                estimated_next=maximum,
                background=True,
                actor="automatic_discovery",
            )

    def _record_started(self, *, actor: str, maximum_cost: Decimal) -> None:
        with session_scope() as session:
            AuditService(session).record(
                "automatic_discovery_started",
                entity_type="automation",
                entity_id=None,
                actor=actor,
                metadata={
                    "maximum_estimated_x_cost_usd": str(maximum_cost),
                    "lookback_hours": self.discovery.lookback_hours,
                },
            )

    def _record_usage(
        self,
        *,
        operation: str,
        units: int,
        unit_cost: Decimal,
        success: bool,
        post_ids: tuple[str, ...] | None = None,
    ) -> None:
        with session_scope() as session:
            costs = CostService(session)
            if post_ids is not None and success:
                costs.record_x_post_reads(
                    operation=operation,
                    post_ids=post_ids,
                    unit_cost=unit_cost,
                    success=True,
                )
                return
            costs.record_usage(
                provider="x",
                operation=operation,
                direction=ApiDirection.READ,
                units=units,
                estimated_cost=unit_cost * units,
                success=success,
            )

    @staticmethod
    def _latest_x_post_id(
        session: Session,
        *,
        source_kind: ContentSourceKind,
        source_identifier: str,
    ) -> str | None:
        """Return the durable high-water mark for one already-ingested timeline.

        X post IDs are monotonically increasing integers. Deriving the cursor from persisted
        occurrences avoids a second cursor store that could drift away from the source rows.
        """

        values = session.scalars(
            select(SourcePost.x_post_id)
            .join(
                SourcePostOccurrence,
                SourcePostOccurrence.source_post_id == SourcePost.id,
            )
            .join(
                ContentSource,
                ContentSource.id == SourcePostOccurrence.content_source_id,
            )
            .where(
                ContentSource.kind == source_kind,
                ContentSource.identifier == source_identifier,
                SourcePost.x_post_id.is_not(None),
            )
        ).all()
        numeric = [value for value in values if value is not None and value.isdecimal()]
        return max(numeric, key=int) if numeric else None

    def _timeline_cursor(
        self,
        *,
        source_kind: ContentSourceKind,
        source_identifier: str,
    ) -> str | None:
        with session_scope() as session:
            return self._latest_x_post_id(
                session,
                source_kind=source_kind,
                source_identifier=source_identifier,
            )

    async def run_once(self, *, actor: str = "automatic_discovery") -> DiscoveryOutcome:
        async with serialized_operation("automatic_discovery"):
            return await self._run_once_serialized(actor=actor)

    async def _run_once_serialized(self, *, actor: str = "automatic_discovery") -> DiscoveryOutcome:
        if not self.discovery.enabled:
            return DiscoveryOutcome(
                "disabled",
                None,
                None,
                0,
                0,
                0,
                (),
                Decimal("0"),
                "Automatic discovery is disabled in config/content_sources.yml.",
            )
        if self.settings.mock_mode:
            return await self._run_mock(actor=actor)

        provider_name = self.configuration.runtime.generation.provider
        provider_cooldown = quota_cooldown_remaining(provider_name)
        if provider_cooldown > 0:
            minutes = max(1, int((provider_cooldown + 59) // 60))
            return DiscoveryOutcome(
                "provider_quota_cooldown",
                None,
                None,
                0,
                0,
                0,
                (),
                Decimal("0"),
                (
                    f"Automatic discovery skipped paid X reads because {provider_name} "
                    "quota is unavailable. Check provider billing or credits and try later; "
                    "the circuit breaker remains active for "
                    f"about {minutes} more minutes."
                ),
            )

        post_cost, trends_cost, maximum_cost = self._costs()
        sources = self.configuration.sources.sources
        home_ready = (
            sources.home_timeline.enabled
            and bool(self.settings.x_user_id)
            and not self.settings.missing_x_home_timeline_credentials()
        )
        active_accounts = tuple(
            account for account in sources.selected_accounts.tracked_accounts if account.enabled
        )
        selected_ready = (
            sources.selected_accounts.enabled
            and bool(active_accounts)
            and not self.settings.missing_x_read_credentials()
        )
        if not home_ready and not selected_ready:
            missing_home = sources.home_timeline.enabled and (
                not self.settings.x_user_id
                or bool(self.settings.missing_x_home_timeline_credentials())
            )
            missing_selected = (
                sources.selected_accounts.enabled
                and bool(active_accounts)
                and bool(self.settings.missing_x_read_credentials())
            )
            status = "credentials_missing" if missing_home or missing_selected else "no_sources"
            message = (
                "Automatic discovery has no readable tweet source. Configure the home timeline "
                "or enable at least one Selected account. Recent Search is intentionally disabled."
            )
            if status == "credentials_missing":
                message = (
                    "Automatic discovery is not ready: home timeline needs X_USER_ID plus "
                    "user-context OAuth, while Selected accounts need X read credentials. "
                    "Recent Search is intentionally disabled."
                )
            return DiscoveryOutcome(
                status,
                None,
                None,
                0,
                0,
                0,
                (),
                maximum_cost,
                message,
            )
        runs_today = self._runs_today()
        if runs_today >= self.discovery.max_runs_per_utc_day:
            return DiscoveryOutcome(
                "daily_limit",
                None,
                None,
                0,
                0,
                0,
                (),
                Decimal("0"),
                (
                    "The daily automatic-discovery run limit has been reached "
                    f"({runs_today}/{self.discovery.max_runs_per_utc_day} UTC today). "
                    "Change automatic_discovery.max_runs_per_utc_day in "
                    "config/content_sources.yml or CONFIGURE_VOUCH.bat. "
                    "DAILY_X_READ_LIMIT_USD and DAILY_X_WRITE_LIMIT_USD are spend budgets, "
                    "not run-count limits."
                ),
            )

        try:
            self._preflight_budget(maximum_cost)
        except BudgetExceededError as exc:
            next_cost = exc.estimated_next or maximum_cost
            return DiscoveryOutcome(
                "budget_blocked",
                None,
                None,
                0,
                0,
                0,
                (),
                Decimal("0"),
                (
                    "The bot's local X-read safety budget blocked this run: "
                    f"estimated ${exc.spent:.3f} used today, local daily limit "
                    f"${exc.limit:.3f}, and the next bounded run can reserve up to "
                    f"${next_cost:.3f}. This limit is separate from the Manage Spend Cap "
                    "in X Console. Open CONFIGURE_VOUCH.bat and increase 'Local daily X "
                    "read budget, USD', or wait until 00:00 UTC."
                ),
            )
        self._record_started(actor=actor, maximum_cost=maximum_cost)
        since = datetime.now(UTC) - timedelta(hours=self.discovery.lookback_hours)
        home_posts: tuple[Mapping[str, object], ...] = ()
        account_batches: list[tuple[str, tuple[Mapping[str, object], ...]]] = []
        trends: tuple[str, ...] = ()
        diagnostics: list[str] = []
        source_errors: list[XReadError] = []
        source_paths_attempted = 0
        source_paths_succeeded = 0
        x_request_count = 0

        async def before_x_request() -> None:
            nonlocal x_request_count
            if x_request_count > 0 and self.request_pacing.x_request_delay_seconds > 0:
                await asyncio.sleep(self.request_pacing.x_request_delay_seconds)
            x_request_count += 1

        if sources.home_timeline.enabled:
            if not self.settings.x_user_id:
                diagnostics.append("home timeline skipped: X_USER_ID is missing")
            elif self.settings.missing_x_home_timeline_credentials():
                diagnostics.append("home timeline skipped: user-context OAuth is missing")
            else:
                source_paths_attempted += 1
                try:
                    _LOGGER.info(
                        "Reading X home timeline: max_posts=%s lookback_hours=%s",
                        sources.home_timeline.max_posts,
                        self.discovery.lookback_hours,
                    )
                    await before_x_request()
                    home_identifier = f"home:{self.discovery.lookback_hours}h"
                    home_posts = await self.reader.get_home_timeline(
                        self.settings.x_user_id,
                        max_results=sources.home_timeline.max_posts,
                        start_time=since,
                        since_id=self._timeline_cursor(
                            source_kind=ContentSourceKind.HOME_TIMELINE,
                            source_identifier=home_identifier,
                        ),
                        exclude_replies=sources.home_timeline.exclude_replies,
                        exclude_retweets=sources.home_timeline.exclude_retweets,
                    )
                except XReadError as exc:
                    source_errors.append(exc)
                    diagnostics.append(_read_diagnostic("home timeline", exc))
                    self._record_usage(
                        operation="home_timeline",
                        units=0,
                        unit_cost=post_cost,
                        success=False,
                    )
                else:
                    source_paths_succeeded += 1
                    _LOGGER.info("X home timeline read completed: posts=%s", len(home_posts))
                    self._record_usage(
                        operation="home_timeline",
                        units=len(home_posts),
                        unit_cost=post_cost,
                        success=True,
                        post_ids=tuple(
                            str(post.get("id", "")).strip()
                            for post in home_posts
                            if str(post.get("id", "")).strip()
                        ),
                    )

        if self.settings.missing_x_read_credentials():
            diagnostics.append("trends skipped: X read credentials are missing")
        else:
            try:
                await before_x_request()
                trends = await self.reader.get_trends_by_woeid(
                    self.discovery.trends_woeid,
                    max_trends=self.discovery.max_trends,
                )
            except XReadError as exc:
                diagnostics.append(_read_diagnostic("trends", exc))
                self._record_usage(
                    operation="trends",
                    units=(exc.attempts if isinstance(exc, XReadTemporaryError) else 1),
                    unit_cost=trends_cost,
                    success=False,
                )
            else:
                self._record_usage(
                    operation="trends",
                    units=max(1, int(getattr(self.reader, "last_attempt_count", 1))),
                    unit_cost=trends_cost,
                    success=True,
                )

        if sources.selected_accounts.enabled:
            if not active_accounts:
                diagnostics.append("selected accounts skipped: no enabled accounts configured")
            elif self.settings.missing_x_read_credentials():
                diagnostics.append("selected accounts skipped: X read credentials are missing")
            else:
                for account in active_accounts:
                    source_paths_attempted += 1
                    try:
                        user_id = account.user_id
                        if not user_id:
                            await before_x_request()
                            user = await self.reader.get_user_by_username(account.username)
                            user_id = str(user.get("id", "")).strip()
                            if not user_id:
                                raise XReadError(f"X account @{account.username} has no user ID")
                        await before_x_request()
                        account_identifier = f"account:{account.username}"
                        posts = await self.reader.get_user_posts(
                            user_id,
                            max_results=sources.selected_accounts.max_posts_per_account,
                            start_time=since,
                            since_id=self._timeline_cursor(
                                source_kind=ContentSourceKind.SELECTED_ACCOUNT,
                                source_identifier=account_identifier,
                            ),
                            exclude_replies=sources.home_timeline.exclude_replies,
                            exclude_retweets=sources.home_timeline.exclude_retweets,
                        )
                    except XReadError as exc:
                        source_errors.append(exc)
                        diagnostics.append(
                            _read_diagnostic(f"selected account @{account.username}", exc)
                        )
                        self._record_usage(
                            operation="selected_account_timeline",
                            units=0,
                            unit_cost=post_cost,
                            success=False,
                        )
                    else:
                        source_paths_succeeded += 1
                        account_batches.append((account.username, posts))
                        self._record_usage(
                            operation="selected_account_timeline",
                            units=len(posts),
                            unit_cost=post_cost,
                            success=True,
                            post_ids=tuple(
                                str(post.get("id", "")).strip()
                                for post in posts
                                if str(post.get("id", "")).strip()
                            ),
                        )

        fetched = len(home_posts) + sum(len(posts) for _, posts in account_batches)
        if fetched > self.discovery.max_total_posts:
            remaining = max(0, self.discovery.max_total_posts - len(home_posts))
            limited: list[tuple[str, tuple[Mapping[str, object], ...]]] = []
            for username, posts in account_batches:
                if remaining <= 0:
                    break
                selected = posts[:remaining]
                limited.append((username, selected))
                remaining -= len(selected)
            account_batches = limited
            fetched = len(home_posts) + sum(len(posts) for _, posts in account_batches)

        idea_ids: list[str] = []
        with session_scope() as session:
            collector = IdeaCollector(
                session,
                source_configuration=self.configuration.sources,
                similarity_threshold=self.settings.similarity_threshold,
            )
            if home_posts:
                idea_ids.extend(
                    collector.ingest(
                        home_posts,
                        source_kind=ContentSourceKind.HOME_TIMELINE,
                        source_name="Home timeline (24h bounded)",
                        source_identifier=f"home:{self.discovery.lookback_hours}h",
                    )
                )
            for username, posts in account_batches:
                idea_ids.extend(
                    collector.ingest(
                        posts,
                        source_kind=ContentSourceKind.SELECTED_ACCOUNT,
                        source_name=f"Selected account: @{username}",
                        source_identifier=f"account:{username}",
                    )
                )

        unique_ids = tuple(dict.fromkeys(idea_ids))
        selected_idea_ids: tuple[str, ...] = ()
        if unique_ids:
            with session_scope() as session:
                candidates = list(
                    session.scalars(
                        select(Idea)
                        .where(
                            Idea.id.in_(unique_ids),
                            Idea.status != IdeaStatus.ARCHIVED,
                            ~Idea.drafts.any(),
                        )
                        .order_by(Idea.score.desc(), Idea.created_at.desc())
                        .limit(self.discovery.final_candidates)
                    ).all()
                )
                if candidates:
                    selected_idea_ids = self._create_editorial_ideas(
                        session,
                        candidates,
                        trends=trends,
                        max_sources=self.discovery.final_candidates,
                    )[: self.discovery.generation_candidates_per_run]

        if not selected_idea_ids:
            source_unavailable = source_paths_attempted > 0 and source_paths_succeeded == 0
            status = "source_unavailable" if source_unavailable else "no_candidate"
            with session_scope() as session:
                AuditService(session).record(
                    "automatic_discovery_completed",
                    entity_type="automation",
                    entity_id=None,
                    actor=actor,
                    metadata={
                        "status": status,
                        "fetched_posts": fetched,
                        "source_paths_attempted": source_paths_attempted,
                        "source_paths_succeeded": source_paths_succeeded,
                        "diagnostics": diagnostics,
                    },
                )
            if source_unavailable:
                temporary_only = bool(source_errors) and all(
                    isinstance(error, XReadTemporaryError) for error in source_errors
                )
                if temporary_only:
                    message = (
                        "X API was temporarily unavailable after bounded retries, so the bot "
                        "could not read a source sample. This does not mean your timeline has "
                        "no new posts. Retry /discover in a few minutes."
                    )
                else:
                    message = (
                        "X source reads failed, so the bot could not determine whether new "
                        "posts exist. Review the diagnostics and retry /discover."
                    )
            else:
                message = (
                    "No new editorial candidate was found in the source sample that responded."
                )
            if diagnostics:
                message += " " + "; ".join(diagnostics)
            return DiscoveryOutcome(
                status,
                None,
                None,
                fetched,
                len(home_posts),
                sum(len(posts) for _, posts in account_batches),
                trends,
                maximum_cost,
                message,
            )

        draft_id: str | None = None
        selected_idea_id: str | None = None
        rejection_diagnostics: list[str] = []
        operator_notices: list[str] = []
        for candidate_idea_id in selected_idea_ids:
            try:
                with session_scope() as session:
                    services = build_services(session, self.settings)
                    draft = await services.drafts.generate_from_idea(
                        candidate_idea_id,
                        actor=actor,
                    )
                    if draft is None:
                        rejection_diagnostics.append(
                            f"{candidate_idea_id}: no_post_needed (no grounded contribution)"
                        )
                        rejected = session.scalar(select(Idea).where(Idea.id == candidate_idea_id))
                        if rejected is not None and not rejected.drafts:
                            rejected.status = IdeaStatus.ARCHIVED
                        AuditService(session).record(
                            "automatic_discovery_candidate_rejected",
                            entity_type="idea",
                            entity_id=candidate_idea_id,
                            actor=actor,
                            metadata={
                                "status": "no_post_needed",
                                "diagnostic": "no grounded contribution",
                            },
                        )
                        continue
                    draft_id = draft.id
                    selected_idea_id = candidate_idea_id
                break
            except MediaContextUnavailableError as exc:
                source_url = exc.source_url or "the original X post"
                diagnostic = f"media context unavailable: {exc.detail}"
                rejection_diagnostics.append(diagnostic)
                operator_notices.append(
                    "The original X media could not be downloaded after bounded retries, so "
                    "no text-only draft was saved. Open the source and quote/retweet it "
                    f"manually: {source_url}"
                )
                _LOGGER.warning(
                    "Automatic discovery media context unavailable idea=%s source=%s: %s",
                    candidate_idea_id,
                    exc.source_post_id,
                    exc.detail,
                )
                with session_scope() as session:
                    rejected = session.scalar(select(Idea).where(Idea.id == candidate_idea_id))
                    if rejected is not None and not rejected.drafts:
                        rejected.status = IdeaStatus.ARCHIVED
                    AuditService(session).record(
                        "automatic_discovery_candidate_rejected",
                        entity_type="idea",
                        entity_id=candidate_idea_id,
                        actor=actor,
                        metadata={
                            "status": "media_context_unavailable",
                            "diagnostic": diagnostic,
                            "source_post_id": exc.source_post_id,
                            "source_url": exc.source_url,
                        },
                    )
            except (
                DraftValidationError,
                EditorialQualityError,
                PostLengthMismatchError,
                InsufficientContextError,
            ) as exc:
                diagnostic = _generation_rejection_diagnostic(exc)
                rejection_diagnostics.append(diagnostic)
                _LOGGER.warning(
                    "Automatic discovery generation rejected idea=%s: %s",
                    candidate_idea_id,
                    diagnostic,
                )
                with session_scope() as session:
                    rejected = session.scalar(select(Idea).where(Idea.id == candidate_idea_id))
                    if rejected is not None and not rejected.drafts:
                        rejected.status = IdeaStatus.ARCHIVED
                    AuditService(session).record(
                        "automatic_discovery_candidate_rejected",
                        entity_type="idea",
                        entity_id=candidate_idea_id,
                        actor=actor,
                        metadata={
                            "status": "generation_rejected",
                            "diagnostic": diagnostic,
                        },
                    )

        if draft_id is None or selected_idea_id is None:
            diagnostic = "; next candidate: ".join(rejection_diagnostics)
            with session_scope() as session:
                AuditService(session).record(
                    "automatic_discovery_completed",
                    entity_type="automation",
                    entity_id=None,
                    actor=actor,
                    metadata={
                        "status": "generation_rejected",
                        "fetched_posts": fetched,
                        "home_posts": len(home_posts),
                        "search_posts": sum(len(posts) for _, posts in account_batches),
                        "trends": list(trends),
                        "attempted_idea_ids": list(selected_idea_ids),
                        "diagnostics": rejection_diagnostics,
                    },
                )
            return DiscoveryOutcome(
                "generation_rejected",
                None,
                selected_idea_ids[0] if selected_idea_ids else None,
                fetched,
                len(home_posts),
                sum(len(posts) for _, posts in account_batches),
                trends,
                maximum_cost,
                (
                    "X sources were read, but the bounded generation attempts did not produce "
                    "a reviewable draft. Rejected source anchors were archived so the next run "
                    "will not repeat them. This run did not consume the daily successful-run "
                    "allowance. Reason: " + diagnostic
                ),
                tuple(operator_notices),
            )

        with session_scope() as session:
            AuditService(session).record(
                "automatic_discovery_completed",
                entity_type="draft",
                entity_id=draft_id,
                actor=actor,
                metadata={
                    "status": "draft_created",
                    "idea_id": selected_idea_id,
                    "fetched_posts": fetched,
                    "home_posts": len(home_posts),
                    "search_posts": sum(len(posts) for _, posts in account_batches),
                    "trends": list(trends),
                    "maximum_estimated_x_cost_usd": str(maximum_cost),
                    "diagnostics": diagnostics,
                },
            )
        _LOGGER.info(
            "Automatic discovery succeeded: draft=%s home=%s selected_accounts=%s",
            draft_id,
            len(home_posts),
            sum(len(posts) for _, posts in account_batches),
        )
        return DiscoveryOutcome(
            "draft_created",
            draft_id,
            selected_idea_id,
            fetched,
            len(home_posts),
            sum(len(posts) for _, posts in account_batches),
            trends,
            maximum_cost,
            "A new editorial angle was selected from X signals and sent for review.",
            tuple(operator_notices),
        )

    async def _run_mock(self, *, actor: str) -> DiscoveryOutcome:
        with session_scope() as session:
            ids = IdeaCollector(
                session,
                source_configuration=self.configuration.sources,
                similarity_threshold=self.settings.similarity_threshold,
            ).collect_mock()
            candidate_ids = tuple(
                idea.id
                for idea in session.scalars(
                    select(Idea)
                    .where(
                        Idea.id.in_(ids),
                        Idea.status != IdeaStatus.ARCHIVED,
                        ~Idea.drafts.any(),
                    )
                    .order_by(Idea.score.desc(), Idea.created_at.desc())
                ).all()
            )
        if not candidate_ids:
            return DiscoveryOutcome(
                "no_candidate",
                None,
                None,
                0,
                0,
                0,
                (),
                Decimal("0"),
                (
                    "Mock sources did not produce a new candidate; existing drafts were not "
                    "regenerated."
                ),
            )
        diagnostics: list[str] = []
        for candidate_id in candidate_ids:
            try:
                with session_scope() as session:
                    services = build_services(session, self.settings)
                    draft = await services.drafts.generate_from_idea(candidate_id, actor=actor)
                    if draft is not None:
                        return DiscoveryOutcome(
                            "draft_created",
                            draft.id,
                            candidate_id,
                            2,
                            1,
                            1,
                            ("mock",),
                            Decimal("0"),
                            "Mock candidate created.",
                        )
                    diagnostics.append(f"{candidate_id}: no grounded contribution")
            except (
                EditorialQualityError,
                PostLengthMismatchError,
                InsufficientContextError,
            ) as exc:
                diagnostics.append(_generation_rejection_diagnostic(exc))
            with session_scope() as session:
                rejected = session.get(Idea, candidate_id)
                if rejected is not None and not rejected.drafts:
                    rejected.status = IdeaStatus.ARCHIVED

        return DiscoveryOutcome(
            "generation_rejected",
            None,
            candidate_ids[0],
            2,
            1,
            1,
            ("mock",),
            Decimal("0"),
            "Mock candidates were exhausted safely. Reason: " + "; ".join(diagnostics),
        )


async def automatic_discovery_loop(
    settings: Settings,
    *,
    stop_event: asyncio.Event,
    on_outcome: object,
) -> None:
    """Run one bounded discovery immediately, then continue on the configured schedule."""

    configuration = ConfigurationService(settings.config_dir).load_sources().automatic_discovery
    if not configuration.enabled:
        return
    first_run = True
    while not stop_event.is_set():
        if first_run:
            first_run = False
            _LOGGER.info("Automatic discovery startup run started")
        else:
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=configuration.effective_interval_minutes * 60,
                )
                continue
            except TimeoutError:
                pass
        try:
            outcome = await AutomaticDiscoveryService(settings).run_once()
        except Exception as exc:
            error_id = new_error_id()
            _LOGGER.error(
                "Automatic discovery failed [%s]: %s: %s\n%s",
                error_id,
                type(exc).__name__,
                safe_exception_summary(exc),
                safe_traceback(exc),
            )
            outcome = DiscoveryOutcome(
                "error",
                None,
                None,
                0,
                0,
                0,
                (),
                Decimal("0"),
                operator_message(exc, error_id),
            )
        callback = on_outcome
        if callable(callback):
            result = callback(outcome)
            if hasattr(result, "__await__"):
                await result
        _LOGGER.info(
            "Automatic discovery run finished: status=%s fetched=%s home=%s selected_accounts=%s",
            outcome.status,
            outcome.fetched_posts,
            outcome.home_posts,
            outcome.search_posts,
        )
