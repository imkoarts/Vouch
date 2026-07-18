"""Alembic migration environment."""

from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from alembic import context
from app.database import ensure_sqlite_directory
from app.models import Base  # imports the complete model registry

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    explicit_url = config.get_main_option("sqlalchemy.url").strip()
    if explicit_url:
        return explicit_url

    from app.config import get_settings

    return str(get_settings().database_url)


def run_migrations_offline() -> None:
    """Run migrations without creating an Engine."""

    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations using a short-lived synchronous connection."""

    url = _database_url()
    ensure_sqlite_directory(url)
    connect_args: dict[str, object] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    connectable = create_engine(
        url,
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    # SQLite reports non-transactional DDL to Alembic, but the version-table
    # INSERT/DELETE still participates in SQLAlchemy's implicit transaction.
    # Engine.begin() guarantees that revision state is committed before the
    # short-lived connection is returned to the pool.
    with connectable.begin() as connection:
        if connection.dialect.name == "sqlite":
            connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()
        if connection.dialect.name == "sqlite":
            # Alembic normally leaves an empty version table after a complete
            # downgrade to base.  Remove it only when it is actually empty so
            # informational commands such as ``alembic current`` remain valid.
            version_table_exists = connection.exec_driver_sql(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'alembic_version'"
            ).first()
            if version_table_exists is not None:
                revision_count = connection.exec_driver_sql(
                    "SELECT COUNT(*) FROM alembic_version"
                ).scalar_one()
                if revision_count == 0:
                    connection.exec_driver_sql("DROP TABLE alembic_version")

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
