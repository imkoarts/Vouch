"""Shared SQLAlchemy primitives.

The application deliberately uses string UUIDs and SQLAlchemy's portable JSON
type so the same model metadata works on both SQLite and PostgreSQL.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(UTC)


def new_uuid() -> str:
    """Return a database-portable UUID value."""

    return str(uuid4())


class UTCDateTime(TypeDecorator[datetime]):
    """Persist aware UTC datetimes and restore tzinfo on SQLite reads."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        del dialect
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("UTCDateTime values must be timezone-aware")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        del dialect
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class Base(DeclarativeBase):
    """Declarative base with deterministic constraint names."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    """Created/updated columns for mutable aggregate roots."""

    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        default=utc_now,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        default=utc_now,
        onupdate=utc_now,
        server_default=func.now(),
        nullable=False,
    )


class AppendOnlyTimestampMixin:
    """Creation timestamp for immutable history rows."""

    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        default=utc_now,
        server_default=func.now(),
        nullable=False,
    )


JsonDict = dict[str, Any]
