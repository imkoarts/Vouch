from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select, text

from app.config import Settings
from app.database import session_scope
from app.models import DashboardChatMessage
from app.services.environment_config import update_environment_file
from app.services.tenant_context import bind_tenant, reset_tenant
from app.services.tenant_storage import TenantWorkspaceRegistry


def test_accounts_receive_physically_separate_persistent_workspaces(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "marker.yml").write_text("safe: true\n", encoding="utf-8")
    settings = Settings(
        app_env="test",
        database_url=f"sqlite:///{(tmp_path / 'master.db').as_posix()}",
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        logs_dir=tmp_path / "logs",
        config_dir=config_dir,
        openai_api_key="global-secret-must-not-leak",
        x_user_id="global-user-id-must-not-leak",
    )
    registry = TenantWorkspaceRegistry(settings)
    first = registry.get("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    second = registry.get("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    try:
        assert first.root != second.root
        assert first.settings.database_url != second.settings.database_url
        assert first.settings.openai_api_key is None
        assert first.settings.x_user_id is None
        with first.engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO dashboard_chat_messages "
                    "(id, role, text, command, metadata, created_at, updated_at) "
                    "VALUES ('one', 'user', 'private', '/help', '{}', "
                    "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )
        with second.engine.connect() as connection:
            count = connection.scalar(text("SELECT COUNT(*) FROM dashboard_chat_messages"))
        assert count == 0
        assert registry.get("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa") is first
    finally:
        registry.dispose()

    update_environment_file(
        first.root / ".env",
        values={"openai_api_key": "tenant-one-secret", "x_user_id": "12345"},
    )
    reloaded_registry = TenantWorkspaceRegistry(settings)
    reloaded = reloaded_registry.get("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    isolated = reloaded_registry.get("cccccccc-cccc-cccc-cccc-cccccccccccc")
    try:
        assert reloaded.settings.openai_api_key is not None
        assert reloaded.settings.openai_api_key.get_secret_value() == "tenant-one-secret"
        assert reloaded.settings.x_user_id == "12345"
        assert isolated.settings.openai_api_key is None
        assert isolated.settings.x_user_id is None
    finally:
        reloaded_registry.dispose()


def test_nested_session_scope_uses_the_bound_account_engine(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "marker.yml").write_text("safe: true\n", encoding="utf-8")
    settings = Settings(
        app_env="test",
        database_url=f"sqlite:///{(tmp_path / 'master.db').as_posix()}",
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
        logs_dir=tmp_path / "logs",
        config_dir=config_dir,
    )
    registry = TenantWorkspaceRegistry(settings)
    workspace = registry.get("dddddddd-dddd-dddd-dddd-dddddddddddd")
    tokens = bind_tenant(workspace.engine, workspace.settings)
    try:
        with session_scope() as session:
            session.add(
                DashboardChatMessage(
                    id="tenant-message",
                    role="user",
                    text="isolated",
                    command="/help",
                    message_metadata={},
                )
            )
        with workspace.engine.connect() as connection:
            count = connection.scalar(select(func.count()).select_from(DashboardChatMessage))
        assert count == 1
    finally:
        reset_tenant(tokens)
        registry.dispose()
