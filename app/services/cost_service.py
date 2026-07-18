"""Configurable cost estimates and hard daily background-operation budgets."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import ApiUsage
from app.models.enums import ApiDirection
from app.services.audit import AuditService

BudgetKind = Literal["x_read", "x_write", "openai", "xai", "heygen"]


class _UnspecifiedEstimate:
    __slots__ = ()


_UNSPECIFIED_ESTIMATE = _UnspecifiedEstimate()


class BudgetExceededError(RuntimeError):
    def __init__(
        self,
        kind: BudgetKind,
        spent: Decimal,
        limit: Decimal,
        estimated_next: Decimal | None = None,
    ) -> None:
        self.kind = kind
        self.spent = spent
        self.limit = limit
        self.estimated_next = estimated_next
        super().__init__(f"Daily {kind} estimate budget is exhausted")


class CostEstimateRequiredError(RuntimeError):
    def __init__(self, kind: BudgetKind) -> None:
        self.kind = kind
        super().__init__(
            f"Daily {kind} budget cannot authorize an operation with an unconfigured cost estimate"
        )


class CostService:
    def __init__(
        self,
        session: Session,
        *,
        limits: dict[BudgetKind, Decimal | None] | None = None,
        estimates: dict[str, Decimal] | None = None,
        allow_unknown_estimates: bool = False,
    ) -> None:
        self.session = session
        self.limits = limits or {}
        self.estimates = estimates or {}
        self.allow_unknown_estimates = allow_unknown_estimates

    @staticmethod
    def _provider_and_direction(kind: BudgetKind) -> tuple[str, ApiDirection | None]:
        if kind == "x_read":
            return "x", ApiDirection.READ
        if kind == "x_write":
            return "x", ApiDirection.WRITE
        return kind, None

    def spent_today(self, kind: BudgetKind, *, now: datetime | None = None) -> Decimal:
        now = now or datetime.now(UTC)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        provider, direction = self._provider_and_direction(kind)
        conditions = [ApiUsage.provider == provider, ApiUsage.created_at >= start]
        if direction is not None:
            conditions.append(ApiUsage.direction == direction)
        value = self.session.scalar(
            select(func.coalesce(func.sum(ApiUsage.estimated_cost_usd), 0)).where(*conditions)
        )
        return Decimal(str(value or 0))

    def require_available(
        self,
        kind: BudgetKind,
        *,
        estimated_next: Decimal | None = None,
        background: bool = False,
        actor: str = "system",
    ) -> None:
        limit = self.limits.get(kind)
        if limit is None:
            return
        if estimated_next is None:
            if self.allow_unknown_estimates:
                return
            if background:
                AuditService(self.session).record(
                    "budget_stop",
                    entity_type="budget",
                    entity_id=kind,
                    actor=actor,
                    metadata={
                        "limit_usd": str(limit),
                        "reason": "cost_estimate_not_configured",
                    },
                )
            raise CostEstimateRequiredError(kind)
        spent = self.spent_today(kind)
        if spent + estimated_next <= limit:
            return
        if background:
            AuditService(self.session).record(
                "budget_stop",
                entity_type="budget",
                entity_id=kind,
                actor=actor,
                metadata={"spent_usd": str(spent), "limit_usd": str(limit)},
            )
        raise BudgetExceededError(kind, spent, limit, estimated_next)

    def x_post_ids_billed_today(self, *, now: datetime | None = None) -> set[str]:
        """Return post IDs already counted by the local X-read estimator today.

        X bills the same post only once within a UTC day even when multiple
        endpoints return it. Usage rows created before metadata tracking remain
        conservative and cannot be reconstructed.
        """

        now = now or datetime.now(UTC)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        rows = self.session.scalars(
            select(ApiUsage).where(
                ApiUsage.provider == "x",
                ApiUsage.direction == ApiDirection.READ,
                ApiUsage.created_at >= start,
                ApiUsage.success.is_(True),
            )
        ).all()
        result: set[str] = set()
        for row in rows:
            raw_ids = (row.usage_metadata or {}).get("post_ids", [])
            if isinstance(raw_ids, list):
                result.update(str(value) for value in raw_ids if str(value).strip())
        return result

    def record_x_post_reads(
        self,
        *,
        operation: str,
        post_ids: list[str] | tuple[str, ...],
        unit_cost: Decimal,
        success: bool = True,
        request_id: str | None = None,
    ) -> ApiUsage:
        """Record only newly billable post IDs under X's daily deduplication rule."""

        normalized = list(
            dict.fromkeys(str(value).strip() for value in post_ids if str(value).strip())
        )
        billed = self.x_post_ids_billed_today() if success else set()
        new_ids = [value for value in normalized if value not in billed]
        return self.record_usage(
            provider="x",
            operation=operation,
            direction=ApiDirection.READ,
            units=len(new_ids),
            estimated_cost=unit_cost * len(new_ids),
            success=success,
            request_id=request_id,
            usage_metadata={
                "post_ids": new_ids,
                "returned_post_count": len(normalized),
                "daily_deduplicated_count": len(normalized) - len(new_ids),
            },
        )

    def estimate(self, operation: str, *, units: int = 1) -> Decimal | None:
        """Return the configured estimate without inventing a zero price."""

        return self.estimate_optional(operation, units=units)

    def estimate_optional(self, operation: str, *, units: int = 1) -> Decimal | None:
        """Return ``None`` when the operator has not configured an estimate."""

        value = self.estimates.get(operation)
        return value * units if value is not None else None

    def record_usage(
        self,
        *,
        provider: str,
        operation: str,
        direction: ApiDirection,
        units: int = 1,
        estimated_cost: Decimal | None | _UnspecifiedEstimate = _UNSPECIFIED_ESTIMATE,
        success: bool = True,
        request_id: str | None = None,
        usage_metadata: dict[str, object] | None = None,
    ) -> ApiUsage:
        stored_estimate = (
            self.estimate_optional(operation, units=units)
            if isinstance(estimated_cost, _UnspecifiedEstimate)
            else estimated_cost
        )
        usage = ApiUsage(
            provider=provider,
            operation=operation,
            direction=direction,
            units=units,
            estimated_cost_usd=stored_estimate,
            request_id=request_id,
            success=success,
            usage_metadata=usage_metadata or {},
        )
        self.session.add(usage)
        self.session.flush()
        return usage
