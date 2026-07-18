"""Unknown estimates remain distinct from explicitly configured zero prices."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.database import build_engine
from app.models import Base
from app.models.enums import ApiDirection
from app.schemas.configuration import XCostEstimates
from app.services.cost_service import CostEstimateRequiredError, CostService


def test_unknown_known_and_explicit_zero_estimates_remain_distinct(
    tmp_path: Path,
) -> None:
    engine = build_engine(f"sqlite:///{(tmp_path / 'costs.db').as_posix()}")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        unknown = CostService(session)
        assert unknown.estimate("x_write") is None
        unknown_usage = unknown.record_usage(
            provider="x",
            operation="x_write",
            direction=ApiDirection.WRITE,
        )
        assert unknown_usage.estimated_cost_usd is None

        configured = CostService(
            session,
            estimates={
                "free_operation": Decimal("0"),
                "known_operation": Decimal("0.0125"),
            },
        )
        assert configured.estimate("free_operation") == Decimal("0")
        assert configured.estimate("known_operation", units=3) == Decimal("0.0375")
        free_usage = configured.record_usage(
            provider="x",
            operation="free_operation",
            direction=ApiDirection.READ,
        )
        assert free_usage.estimated_cost_usd == Decimal("0")
        explicitly_unknown_usage = configured.record_usage(
            provider="x",
            operation="known_operation",
            direction=ApiDirection.WRITE,
            estimated_cost=None,
        )
        assert explicitly_unknown_usage.estimated_cost_usd is None

    explicit_zero = XCostEstimates(read_usd=Decimal("0"), write_usd=Decimal("0"))
    assert explicit_zero.write_usd == Decimal("0")


def test_live_budget_rejects_unknown_estimate_but_mock_policy_can_allow_it(
    tmp_path: Path,
) -> None:
    engine = build_engine(f"sqlite:///{(tmp_path / 'budget.db').as_posix()}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        live_style = CostService(
            session,
            limits={"x_write": Decimal("1.00")},
        )
        with pytest.raises(CostEstimateRequiredError, match="unconfigured"):
            live_style.require_available("x_write", estimated_next=None)

        mock_style = CostService(
            session,
            limits={"x_write": Decimal("1.00")},
            allow_unknown_estimates=True,
        )
        mock_style.require_available("x_write", estimated_next=None)


def test_x_read_estimator_deduplicates_same_post_across_endpoints_per_utc_day(
    tmp_path: Path,
) -> None:
    engine = build_engine(f"sqlite:///{(tmp_path / 'dedupe.db').as_posix()}")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        costs = CostService(session)
        first = costs.record_x_post_reads(
            operation="home_timeline",
            post_ids=("100", "101", "101"),
            unit_cost=Decimal("0.005"),
        )
        second = costs.record_x_post_reads(
            operation="recent_search",
            post_ids=("101", "102"),
            unit_cost=Decimal("0.005"),
        )
        session.commit()

        assert first.units == 2
        assert first.estimated_cost_usd == Decimal("0.010")
        assert second.units == 1
        assert second.estimated_cost_usd == Decimal("0.005")
        assert second.usage_metadata["daily_deduplicated_count"] == 1
        assert costs.spent_today("x_read") == Decimal("0.015000")
