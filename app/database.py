"""Synchronous SQLAlchemy engine and unit-of-work helpers."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker


def ensure_sqlite_directory(database_url: str, *, base_dir: Path | None = None) -> Path | None:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database:
        return None
    if url.database in {":memory:", ""} or url.database.startswith("file:"):
        return None
    database_path = Path(url.database).expanduser()
    if not database_path.is_absolute() and base_dir is not None:
        database_path = Path(base_dir) / database_path
    database_path = database_path.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return database_path


def build_engine(database_url: str, **kwargs: Any) -> Engine:
    """Create a configured sync engine for SQLite or PostgreSQL."""

    ensure_sqlite_directory(database_url)
    url = make_url(database_url)
    options: dict[str, Any] = {"pool_pre_ping": True}
    options.update(kwargs)
    if url.get_backend_name() == "sqlite":
        connect_args = dict(options.pop("connect_args", {}))
        connect_args.setdefault("check_same_thread", False)
        options["connect_args"] = connect_args

    engine = create_engine(database_url, **options)
    if url.get_backend_name() == "sqlite":

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection: Any, connection_record: Any) -> None:
            del connection_record
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=15000")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
            finally:
                cursor.close()

    return engine


def _settings_database_url() -> str:
    # Import lazily so Alembic/tests can provide an explicit URL without loading
    # unrelated application settings.
    from app.config import get_settings

    return str(get_settings().database_url)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide engine configured by Pydantic Settings."""

    return build_engine(_settings_database_url())


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Return a non-expiring SQLAlchemy session factory."""

    return sessionmaker(
        bind=get_engine(),
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a transaction-neutral session."""

    session = _active_session_factory()()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Commit on success and roll back on failure."""

    session = _active_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _active_session_factory() -> sessionmaker[Session]:
    """Resolve tenant ContextVars for nested services and background request tasks."""

    from app.services.tenant_context import bound_engine

    engine = bound_engine()
    if engine is None:
        return get_session_factory()
    return sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )


def reset_database_state() -> None:
    """Dispose cached resources (primarily for isolated tests)."""

    if get_engine.cache_info().currsize:
        get_engine().dispose()
    get_session_factory.cache_clear()
    get_engine.cache_clear()
