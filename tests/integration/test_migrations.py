"""The initial Alembic revision must bootstrap and tear down a fresh DB."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine.reflection import Inspector

from alembic import command
from app.models import Base

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _alembic_config(database_path: Path) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")
    return config


def _unique_names(inspector: Inspector, table_name: str) -> set[str]:
    constraints = inspector.get_unique_constraints(table_name)
    return {item["name"] for item in constraints if item["name"] is not None}


def _unique_index_names(inspector: Inspector, table_name: str) -> set[str]:
    indexes = inspector.get_indexes(table_name)
    return {item["name"] for item in indexes if item["name"] is not None and item.get("unique")}


def test_initial_migration_matches_model_registry_and_downgrades(tmp_path: Path) -> None:
    database_path = tmp_path / "migration.db"
    config = _alembic_config(database_path)

    command.upgrade(config, "head")
    # Informational Alembic commands also execute env.py and must not assume a
    # destination revision is present in the command context.
    command.current(config)

    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    inspector = inspect(engine)
    expected_tables = {
        "api_usage",
        "app_settings",
        "auth_sessions",
        "audit_events",
        "content_sources",
        "dashboard_chat_messages",
        "draft_approvals",
        "draft_versions",
        "drafts",
        "idea_sources",
        "ideas",
        "llm_runs",
        "local_otp_challenges",
        "opt_out_users",
        "publications",
        "reply_candidates",
        "source_posts",
        "source_post_occurrences",
        "topics",
        "user_accounts",
        "video_jobs",
        "video_scripts",
        "voice_profiles",
        "x_activity_events",
    }

    actual_tables = set(inspector.get_table_names()) - {"alembic_version"}
    assert actual_tables == expected_tables
    assert actual_tables == set(Base.metadata.tables)
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
            "0008_local_passwordless"
        )

    assert "uq_draft_versions_draft_version" in _unique_names(inspector, "draft_versions")
    assert "uq_draft_approvals_supersedes" in _unique_names(inspector, "draft_approvals")
    assert "uq_publications_draft_hash_position" in _unique_names(inspector, "publications")
    assert "uq_publications_request_position" in _unique_names(inspector, "publications")
    assert "uq_reply_candidates_source_post_id" in _unique_names(inspector, "reply_candidates")
    assert "uq_source_posts_text_hash" not in _unique_names(inspector, "source_posts")
    assert "ux_source_posts_import_fingerprint" in _unique_index_names(inspector, "source_posts")
    assert "uq_source_post_occurrences_post_source" in _unique_names(
        inspector, "source_post_occurrences"
    )

    publication_columns = {column["name"] for column in inspector.get_columns("publications")}
    assert {
        "draft_id",
        "x_post_id",
        "parent_post_id",
        "thread_position",
        "request_payload_hash",
        "response_metadata",
        "created_at",
        "deletion_timestamp",
        "error_details",
    } <= publication_columns
    assert "import_fingerprint" in {
        column["name"] for column in inspector.get_columns("source_posts")
    }
    for table_name in ("llm_runs", "api_usage", "publications"):
        cost_column = next(
            column
            for column in inspector.get_columns(table_name)
            if column["name"] == "estimated_cost_usd"
        )
        assert cost_column["nullable"] is True

    engine.dispose()
    command.downgrade(config, "base")
    command.current(config)

    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    try:
        assert inspect(engine).get_table_names() == []
    finally:
        engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    try:
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
                "0008_local_passwordless"
            )
    finally:
        engine.dispose()


def test_alembic_creates_missing_sqlite_parent_directory(tmp_path: Path) -> None:
    database_path = tmp_path / "folder with spaces" / "nested" / "app.db"
    assert not database_path.parent.exists()

    config = _alembic_config(database_path)
    command.upgrade(config, "head")

    assert database_path.is_file()
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    try:
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
                "0008_local_passwordless"
            )
    finally:
        engine.dispose()


def test_occurrence_migration_preserves_existing_source_association(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "existing.db"
    config = _alembic_config(database_path)
    command.upgrade(config, "0001_initial")

    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO content_sources "
                "(id, kind, name, identifier, category, weight, enabled, "
                "configuration, created_at, updated_at) VALUES "
                "(:id, :kind, :name, :identifier, NULL, :weight, 1, '{}', "
                ":created_at, :updated_at)"
            ),
            {
                "id": "source-1",
                "kind": "recent_search",
                "name": "Existing search",
                "identifier": "query",
                "weight": 0.25,
                "created_at": "2026-07-11 00:00:00",
                "updated_at": "2026-07-11 00:00:00",
            },
        )
        connection.execute(
            text(
                "INSERT INTO source_posts "
                "(id, x_post_id, content_source_id, url, author_id, username, "
                "text, language, published_at, public_metrics, conversation_id, "
                "media_metadata, source_labels, fetched_at, raw_json, text_hash, "
                "created_at, updated_at) VALUES "
                "(:id, :x_post_id, :content_source_id, NULL, NULL, NULL, :body, "
                "'en', NULL, '{}', :conversation_id, '{}', '[]', :fetched_at, "
                "'{}', :text_hash, :created_at, :updated_at)"
            ),
            {
                "id": "post-1",
                "x_post_id": "301",
                "content_source_id": "source-1",
                "body": "Synthetic existing row",
                "conversation_id": "301",
                "fetched_at": "2026-07-11 00:00:00",
                "text_hash": "a" * 64,
                "created_at": "2026-07-11 00:00:00",
                "updated_at": "2026-07-11 00:00:00",
            },
        )
    engine.dispose()

    command.upgrade(config, "head")

    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    try:
        inspector = inspect(engine)
        assert "content_source_id" not in {
            column["name"] for column in inspector.get_columns("source_posts")
        }
        with engine.connect() as connection:
            occurrence = connection.execute(
                text(
                    "SELECT source_post_id, content_source_id, source_weight "
                    "FROM source_post_occurrences"
                )
            ).one()
        assert tuple(occurrence) == ("post-1", "source-1", 0.25)
    finally:
        engine.dispose()

    command.downgrade(config, "0001_initial")
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    try:
        inspector = inspect(engine)
        assert "content_source_id" in {
            column["name"] for column in inspector.get_columns("source_posts")
        }
        assert "uq_source_posts_text_hash" in _unique_names(inspector, "source_posts")
        with engine.connect() as connection:
            assert (
                connection.scalar(
                    text("SELECT content_source_id FROM source_posts WHERE id = 'post-1'")
                )
                == "source-1"
            )
            assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
                "0001_initial"
            )
    finally:
        engine.dispose()


def test_occurrence_downgrade_refuses_duplicate_text_hash_before_any_change(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "duplicate-text.db"
    config = _alembic_config(database_path)
    command.upgrade(config, "0002_source_occurrences")

    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    with engine.begin() as connection:
        for post_id, x_post_id in (("post-1", "401"), ("post-2", "402")):
            connection.execute(
                text(
                    "INSERT INTO source_posts "
                    "(id, x_post_id, url, author_id, username, text, language, "
                    "published_at, public_metrics, conversation_id, media_metadata, "
                    "source_labels, fetched_at, raw_json, text_hash, created_at, "
                    "updated_at) VALUES "
                    "(:id, :x_post_id, NULL, NULL, NULL, :body, 'en', NULL, '{}', "
                    ":conversation_id, '{}', '[]', :fetched_at, '{}', :text_hash, "
                    ":created_at, :updated_at)"
                ),
                {
                    "id": post_id,
                    "x_post_id": x_post_id,
                    "body": f"Synthetic duplicate {post_id}",
                    "conversation_id": x_post_id,
                    "fetched_at": "2026-07-11 00:00:00",
                    "text_hash": "d" * 64,
                    "created_at": "2026-07-11 00:00:00",
                    "updated_at": "2026-07-11 00:00:00",
                },
            )
    with engine.connect() as connection:
        schema_before = connection.execute(
            text(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            )
        ).all()
        rows_before = connection.execute(
            text("SELECT id, x_post_id, text, text_hash FROM source_posts ORDER BY id")
        ).all()
    engine.dispose()

    with pytest.raises(RuntimeError, match=r"Downgrade.*blocked"):
        command.downgrade(config, "0001_initial")

    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    try:
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
                "0002_source_occurrences"
            )
            assert (
                connection.execute(
                    text(
                        "SELECT type, name, sql FROM sqlite_master "
                        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
                    )
                ).all()
                == schema_before
            )
            assert (
                connection.execute(
                    text("SELECT id, x_post_id, text, text_hash FROM source_posts ORDER BY id")
                ).all()
                == rows_before
            )
        assert "_source_post_occurrence_stage" not in inspect(engine).get_table_names()
    finally:
        engine.dispose()
