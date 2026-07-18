from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest

from app.config import Settings
from app.services.trend_discovery import (
    AutomaticDiscoveryService,
    DiscoveryOutcome,
    _generation_rejection_diagnostic,
    _trend_query,
    automatic_discovery_loop,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _settings(tmp_path: Path, **updates: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "config_dir": PROJECT_ROOT / "config",
        "data_dir": tmp_path / "data",
        "drafts_dir": tmp_path / "drafts",
        "logs_dir": tmp_path / "logs",
        "database_url": f"sqlite:///{(tmp_path / 'app.db').as_posix()}",
        "mock_mode": False,
        "x_bearer_token": "synthetic-bearer",
        "daily_x_read_limit_usd": 0.25,
    }
    values.update(updates)
    return Settings(**values)


def test_discovery_cost_includes_bounded_temporary_trend_retries(tmp_path: Path) -> None:
    service = AutomaticDiscoveryService(_settings(tmp_path))

    post_cost, trends_cost, maximum = service._costs()

    assert post_cost == Decimal("0.005")
    assert trends_cost == Decimal("0.010")
    assert maximum == Decimal("0.055")
    assert service.discovery.max_total_posts == 15
    assert service.discovery.max_runs_per_utc_day == 5
    assert service.discovery.generation_candidates_per_run == 5


def test_discovery_requires_explicit_daily_budget_before_database_access(tmp_path: Path) -> None:
    service = AutomaticDiscoveryService(_settings(tmp_path, daily_x_read_limit_usd=None))

    with pytest.raises(RuntimeError, match="DAILY_X_READ_LIMIT_USD"):
        service._preflight_budget(Decimal("0.085"))


def test_trend_query_quotes_phrases_and_rejects_empty_value() -> None:
    assert _trend_query("World Cup") == '"World Cup" -is:retweet -is:reply'
    assert _trend_query("#WorldCup") == "#WorldCup -is:retweet -is:reply"
    with pytest.raises(ValueError, match="empty"):
        _trend_query("   ")


@pytest.mark.asyncio
async def test_automatic_discovery_loop_runs_immediately_on_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stop_event = asyncio.Event()
    outcomes: list[DiscoveryOutcome] = []
    calls = 0

    class FakeDiscoveryService:
        def __init__(self, _settings: Settings) -> None:
            pass

        async def run_once(self) -> DiscoveryOutcome:
            nonlocal calls
            calls += 1
            stop_event.set()
            return DiscoveryOutcome(
                "no_candidate",
                None,
                None,
                0,
                0,
                0,
                (),
                Decimal("0"),
                "Startup scan completed.",
            )

    async def capture(outcome: DiscoveryOutcome) -> None:
        outcomes.append(outcome)

    monkeypatch.setattr(
        "app.services.trend_discovery.AutomaticDiscoveryService",
        FakeDiscoveryService,
    )

    await asyncio.wait_for(
        automatic_discovery_loop(
            _settings(tmp_path),
            stop_event=stop_event,
            on_outcome=capture,
        ),
        timeout=1,
    )

    assert calls == 1
    assert [item.message for item in outcomes] == ["Startup scan completed."]


@pytest.mark.asyncio
async def test_provider_quota_cooldown_skips_paid_x_reads(tmp_path: Path) -> None:
    from app.utils.request_pacing import mark_quota_cooldown, reset_request_pacing_for_tests

    reset_request_pacing_for_tests()
    mark_quota_cooldown("openai", cooldown_seconds=300)
    service = AutomaticDiscoveryService(_settings(tmp_path))

    outcome = await service.run_once()

    assert outcome.status == "provider_quota_cooldown"
    assert outcome.fetched_posts == 0
    assert outcome.estimated_x_cost_usd == Decimal("0")
    assert "skipped paid X reads" in outcome.message
    reset_request_pacing_for_tests()


def test_automatic_discovery_selects_one_editorial_anchor() -> None:
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from app.domain.enums import ContentType, IdeaStatus
    from app.models import Base, Idea, IdeaSource, SourcePost

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        first_post = SourcePost(
            x_post_id="1001",
            text="Prediction market volume rose after a new product announcement.",
            text_hash="a" * 64,
            username="market_watch",
            public_metrics={"like_count": 120, "reply_count": 14},
            source_labels=["home_timeline"],
        )
        second_post = SourcePost(
            x_post_id="1002",
            text="Users are debating whether eligibility rules should be published early.",
            text_hash="b" * 64,
            username="product_notes",
            public_metrics={"like_count": 90, "quote_count": 11},
            source_labels=["recent_search"],
        )
        session.add_all((first_post, second_post))
        session.flush()
        first = Idea(
            status=IdeaStatus.RANKED,
            title="First signal",
            summary="First source summary",
            recommended_format=ContentType.SHORT_POST,
            score=0.92,
            score_components={},
            explanation={},
            dedupe_key="c" * 64,
        )
        second = Idea(
            status=IdeaStatus.RANKED,
            title="Second signal",
            summary="Second source summary",
            recommended_format=ContentType.SHORT_POST,
            score=0.81,
            score_components={},
            explanation={},
            dedupe_key="d" * 64,
        )
        session.add_all((first, second))
        session.flush()
        session.add_all(
            (
                IdeaSource(
                    idea_id=first.id,
                    source_post_id=first_post.id,
                    source_priority=0.5,
                    contribution=0.92,
                    position=0,
                ),
                IdeaSource(
                    idea_id=second.id,
                    source_post_id=second_post.id,
                    source_priority=0.25,
                    contribution=0.81,
                    position=0,
                ),
            )
        )
        session.flush()

        editorial_id = AutomaticDiscoveryService._create_editorial_idea(
            session,
            [first, second],
            trends=("Prediction markets",),
            max_sources=5,
        )
        assert editorial_id is not None
        editorial = session.get(Idea, editorial_id)
        assert editorial is not None
        links = session.scalars(
            select(IdeaSource)
            .where(IdeaSource.idea_id == editorial_id)
            .order_by(IdeaSource.position)
        ).all()

        assert editorial.explanation["origin"] == "automatic_discovery_single_source"
        assert editorial.explanation["source_count"] == 1
        assert "single source anchor" in editorial.summary.casefold()
        assert "do not mention the source account" in editorial.summary.casefold()
        assert "never collage" in editorial.explanation["differentiation"]
        assert [link.source_post_id for link in links] == [first_post.id]


@pytest.mark.asyncio
async def test_all_temporary_x_failures_are_not_reported_as_no_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.models import Base
    from app.x_api.read import XReadTemporaryError

    class FailingReader:
        async def get_home_timeline(self, *_args: object, **_kwargs: object) -> tuple[object, ...]:
            raise XReadTemporaryError(
                "X API is temporarily unavailable",
                attempts=3,
                status_code=503,
            )

        async def get_trends_by_woeid(self, *_args: object, **_kwargs: object) -> tuple[str, ...]:
            raise XReadTemporaryError(
                "X API is temporarily unavailable",
                attempts=3,
                status_code=503,
            )

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    @contextmanager
    def fake_session_scope():
        with Session(engine) as session:
            yield session
            session.commit()

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("app.services.trend_discovery.session_scope", fake_session_scope)
    monkeypatch.setattr("app.services.trend_discovery.asyncio.sleep", no_sleep)

    settings = _settings(
        tmp_path,
        x_user_id="450",
        x_auth_mode="oauth2",
        x_access_token="synthetic-user-token",
    )
    service = AutomaticDiscoveryService(settings, reader=FailingReader())
    monkeypatch.setattr(service, "_runs_today", lambda: 0)
    monkeypatch.setattr(service, "_preflight_budget", lambda _maximum: None)
    monkeypatch.setattr(service, "_record_started", lambda **_kwargs: None)
    monkeypatch.setattr(service, "_record_usage", lambda **_kwargs: None)

    outcome = await service.run_once()

    assert outcome.status == "source_unavailable"
    assert outcome.fetched_posts == 0
    assert "does not mean your timeline has no new posts" in outcome.message
    assert "Retry /discover" in outcome.message
    assert "HTTP 503" in outcome.message
    assert "No new editorial candidate" not in outcome.message


def test_temporary_source_failure_does_not_consume_daily_run_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager
    from datetime import UTC, datetime

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.models import AuditEvent, Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for status in ("source_unavailable", "no_candidate", "draft_created"):
            session.add(
                AuditEvent(
                    action="automatic_discovery_completed",
                    entity_type="automation",
                    entity_id=None,
                    actor="test",
                    event_metadata={"status": status},
                    created_at=datetime.now(UTC),
                )
            )
        session.add(
            AuditEvent(
                action="automatic_discovery_started",
                entity_type="automation",
                entity_id=None,
                actor="test",
                event_metadata={},
                created_at=datetime.now(UTC),
            )
        )
        session.commit()

    @contextmanager
    def fake_session_scope():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr("app.services.trend_discovery.session_scope", fake_session_scope)

    service = AutomaticDiscoveryService(_settings(tmp_path))

    assert service._runs_today() == 2


@pytest.mark.asyncio
async def test_one_successful_source_path_keeps_no_candidate_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.models import Base
    from app.x_api.read import XReadTemporaryError

    class PartialReader:
        async def get_home_timeline(self, *_args: object, **_kwargs: object) -> tuple[object, ...]:
            return ()

        async def get_trends_by_woeid(self, *_args: object, **_kwargs: object) -> tuple[str, ...]:
            raise XReadTemporaryError(
                "X API is temporarily unavailable",
                attempts=3,
                status_code=503,
            )

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    @contextmanager
    def fake_session_scope():
        with Session(engine) as session:
            yield session
            session.commit()

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("app.services.trend_discovery.session_scope", fake_session_scope)
    monkeypatch.setattr("app.services.trend_discovery.asyncio.sleep", no_sleep)

    settings = _settings(
        tmp_path,
        x_user_id="450",
        x_auth_mode="oauth2",
        x_access_token="synthetic-user-token",
    )
    service = AutomaticDiscoveryService(settings, reader=PartialReader())
    monkeypatch.setattr(service, "_runs_today", lambda: 0)
    monkeypatch.setattr(service, "_preflight_budget", lambda _maximum: None)
    monkeypatch.setattr(service, "_record_started", lambda **_kwargs: None)
    monkeypatch.setattr(service, "_record_usage", lambda **_kwargs: None)

    outcome = await service.run_once()

    assert outcome.status == "no_candidate"
    assert "source sample that responded" in outcome.message
    assert "trends: X API temporarily unavailable" in outcome.message


def test_generation_rejection_diagnostic_distinguishes_quality_length_and_context() -> None:
    from app.domain.enums import PostLengthMode
    from app.schemas.content import QualityIssue, QualityReport
    from app.services.draft_service import (
        DraftValidationError,
        EditorialQualityError,
        PostLengthMismatchError,
    )
    from app.services.generation_pipeline import InsufficientContextError
    from app.services.post_length import resolve_post_length

    report = QualityReport(
        passed=False,
        specificity_score=0.0,
        evidence_score=1.0,
        naturalness_score=0.0,
        distinctiveness_score=1.0,
        issues=(
            QualityIssue(
                code="EMPTY_THESIS",
                severity="error",
                message="No publishable observation",
                suggested_action="regenerate_angle",
            ),
        ),
    )
    quality = _generation_rejection_diagnostic(EditorialQualityError((report,)))
    length = _generation_rejection_diagnostic(
        PostLengthMismatchError(
            selection=resolve_post_length(PostLengthMode.MEDIUM),
            lengths=(21, 21, 21),
            reports=(report,),
        )
    )
    context = _generation_rejection_diagnostic(InsufficientContextError("No usable evidence"))
    factual = _generation_rejection_diagnostic(
        DraftValidationError(
            "The selected variant failed deterministic factual evidence validation."
        )
    )

    assert quality == "editorial gate: EMPTY_THESIS"
    assert "required 300-500" in length
    assert "actual 21, 21, 21" in length
    assert context == "context gate: No usable evidence"
    assert factual == "factual gate: deterministic evidence validation failed"


def test_discovery_builds_multiple_single_source_candidates_and_skips_archived() -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.domain.enums import ContentType, IdeaStatus
    from app.models import Base, Idea, IdeaSource, SourcePost

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        posts = [
            SourcePost(
                x_post_id=f"20{index}",
                text=f"A company announced source event {index} with enough concrete context.",
                text_hash=str(index) * 64,
                source_labels=["home_timeline"],
            )
            for index in (1, 2)
        ]
        session.add_all(posts)
        session.flush()
        ideas = []
        for index, post in enumerate(posts, start=1):
            idea = Idea(
                status=IdeaStatus.RANKED,
                title=f"Candidate {index}",
                summary="source",
                recommended_format=ContentType.SHORT_POST,
                score=1 - index / 10,
                score_components={},
                explanation={},
                dedupe_key=(str(index + 2) * 64),
            )
            session.add(idea)
            session.flush()
            session.add(
                IdeaSource(
                    idea_id=idea.id,
                    source_post_id=post.id,
                    source_priority=0.5,
                    contribution=1 - index / 10,
                    position=0,
                )
            )
            ideas.append(idea)
        session.flush()

        editorial_ids = AutomaticDiscoveryService._create_editorial_ideas(
            session,
            ideas,
            trends=("test",),
            max_sources=2,
        )
        assert len(editorial_ids) == 2

        first = session.get(Idea, editorial_ids[0])
        assert first is not None
        first.status = IdeaStatus.ARCHIVED
        session.flush()

        remaining = AutomaticDiscoveryService._create_editorial_ideas(
            session,
            ideas,
            trends=("test",),
            max_sources=2,
        )
        assert remaining == (editorial_ids[1],)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_failure", ["media", "editorial", "factual"])
async def test_generation_failure_tries_next_candidate_and_reports_media_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_failure: str
) -> None:
    from contextlib import contextmanager
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.models import Base
    from app.schemas.content import QualityIssue, QualityReport
    from app.services.draft_service import (
        DraftValidationError,
        EditorialQualityError,
        MediaContextUnavailableError,
    )

    now = datetime.now(UTC).isoformat()

    class Reader:
        async def get_home_timeline(self, *_args: object, **_kwargs: object):
            return (
                {
                    "id": "900000000000000101",
                    "text": "A company launched a product and showed the result in a source video.",
                    "author_id": "700000000000000101",
                    "username": "source_one",
                    "lang": "en",
                    "created_at": now,
                    "conversation_id": "900000000000000101",
                    "url": "https://x.com/source_one/status/900000000000000101",
                    "public_metrics": {"like_count": 100},
                    "media": {
                        "items": [
                            {
                                "media_key": "7_101",
                                "type": "video",
                                "variants": [
                                    {
                                        "content_type": "video/mp4",
                                        "bit_rate": 2176000,
                                        "url": "https://video.twimg.com/one.mp4",
                                    }
                                ],
                            }
                        ]
                    },
                },
                {
                    "id": "900000000000000102",
                    "text": (
                        "A second company announced an independent product event with "
                        "enough concrete context."
                    ),
                    "author_id": "700000000000000102",
                    "username": "source_two",
                    "lang": "en",
                    "created_at": now,
                    "conversation_id": "900000000000000102",
                    "url": "https://x.com/source_two/status/900000000000000102",
                    "public_metrics": {"like_count": 90},
                },
            )

        async def get_trends_by_woeid(self, *_args: object, **_kwargs: object):
            return ()

    engine = create_engine(f"sqlite:///{(tmp_path / 'discovery.db').as_posix()}")
    Base.metadata.create_all(engine)

    @contextmanager
    def fake_session_scope():
        with Session(engine) as session:
            yield session
            session.commit()

    calls: list[str] = []

    class Drafts:
        async def generate_from_idea(self, idea_id: str, *, actor: str):
            calls.append(idea_id)
            if len(calls) == 1:
                if first_failure == "editorial":
                    raise EditorialQualityError(
                        (
                            QualityReport(
                                passed=False,
                                specificity_score=0.0,
                                evidence_score=1.0,
                                naturalness_score=0.0,
                                distinctiveness_score=1.0,
                                issues=(
                                    QualityIssue(
                                        code="FORMAT_MISMATCH",
                                        severity="error",
                                        message="Needs a bounded revision",
                                        suggested_action="shorten",
                                    ),
                                ),
                            ),
                        )
                    )
                if first_failure == "factual":
                    raise DraftValidationError(
                        "The selected variant failed deterministic factual evidence validation."
                    )
                raise MediaContextUnavailableError(
                    source_post_id="900000000000000101",
                    source_url="https://x.com/source_one/status/900000000000000101",
                    detail="X source media transport failed",
                )
            return SimpleNamespace(id="draft-from-second-candidate")

    fake_services = SimpleNamespace(drafts=Drafts())
    monkeypatch.setattr("app.services.trend_discovery.session_scope", fake_session_scope)
    monkeypatch.setattr(
        "app.services.trend_discovery.build_services",
        lambda _session, _settings: fake_services,
    )

    settings = _settings(
        tmp_path,
        x_user_id="450",
        x_auth_mode="oauth2",
        x_access_token="synthetic-user-token",
    )
    service = AutomaticDiscoveryService(settings, reader=Reader())
    monkeypatch.setattr(service, "_runs_today", lambda: 0)
    monkeypatch.setattr(service, "_preflight_budget", lambda _maximum: None)
    monkeypatch.setattr(service, "_record_started", lambda **_kwargs: None)
    monkeypatch.setattr(service, "_record_usage", lambda **_kwargs: None)

    outcome = await service.run_once()

    assert outcome.status == "draft_created"
    assert outcome.draft_id == "draft-from-second-candidate"
    assert len(calls) == 2
    if first_failure == "media":
        assert len(outcome.operator_notices) == 1
        assert "no text-only draft was saved" in outcome.operator_notices[0]
        assert "https://x.com/source_one/status/900000000000000101" in outcome.operator_notices[0]
    else:
        assert outcome.operator_notices == ()


@pytest.mark.asyncio
async def test_local_x_read_budget_returns_actionable_outcome_instead_of_processing_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services.cost_service import BudgetExceededError

    service = AutomaticDiscoveryService(
        _settings(
            tmp_path,
            daily_x_read_limit_usd=0.25,
            x_user_id="450",
            x_auth_mode="oauth2",
            x_access_token="synthetic-user-token",
        )
    )
    monkeypatch.setattr(service, "_runs_today", lambda: 0)

    def blocked(_maximum: Decimal) -> None:
        raise BudgetExceededError(
            "x_read",
            Decimal("0.240"),
            Decimal("0.250"),
            Decimal("0.055"),
        )

    monkeypatch.setattr(service, "_preflight_budget", blocked)

    outcome = await service.run_once()

    assert outcome.status == "budget_blocked"
    assert outcome.fetched_posts == 0
    assert "local X-read safety budget" in outcome.message
    assert "separate from the Manage Spend Cap" in outcome.message
    assert "CONFIGURE_VOUCH.bat" in outcome.message


@pytest.mark.asyncio
async def test_automatic_discovery_never_calls_recent_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.models import Base

    class Reader:
        async def get_home_timeline(self, *_args: object, **_kwargs: object) -> tuple[object, ...]:
            return ()

        async def get_trends_by_woeid(self, *_args: object, **_kwargs: object) -> tuple[str, ...]:
            return ("Synthetic trend",)

        async def recent_search(self, *_args: object, **_kwargs: object) -> tuple[object, ...]:
            raise AssertionError("Recent Search must not be called by automatic discovery")

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    @contextmanager
    def fake_session_scope():
        with Session(engine) as session:
            yield session
            session.commit()

    monkeypatch.setattr("app.services.trend_discovery.session_scope", fake_session_scope)
    settings = _settings(
        tmp_path,
        x_user_id="450",
        x_auth_mode="oauth2",
        x_access_token="synthetic-user-token",
    )
    service = AutomaticDiscoveryService(settings, reader=Reader())
    monkeypatch.setattr(service, "_runs_today", lambda: 0)
    monkeypatch.setattr(service, "_preflight_budget", lambda _maximum: None)
    monkeypatch.setattr(service, "_record_started", lambda **_kwargs: None)
    monkeypatch.setattr(service, "_record_usage", lambda **_kwargs: None)

    outcome = await service.run_once()

    assert outcome.status == "no_candidate"
    assert outcome.trends == ("Synthetic trend",)
