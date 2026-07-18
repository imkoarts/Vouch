"""Contained, physically separate workspaces for authenticated accounts."""

from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from alembic.config import Config
from pydantic import SecretStr
from sqlalchemy import Engine

from alembic import command
from app.config import Settings
from app.database import build_engine
from app.resources import resolve_alembic_config_path
from app.services.environment_config import EDITABLE_ENV_FIELDS

_PLAIN_CREDENTIAL_FIELDS = {"x_client_id", "x_user_id"}


def _account_credentials(path: Path) -> dict[str, str | SecretStr | None]:
    """Load only allowlisted account credentials without inheriting global secrets."""

    result: dict[str, str | SecretStr | None] = {field: None for field in EDITABLE_ENV_FIELDS}
    if not path.exists():
        return result
    by_environment_name = {name: field for field, name in EDITABLE_ENV_FIELDS.items()}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, raw_value = line.split("=", 1)
        field = by_environment_name.get(name.strip())
        if field is None:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if not value:
            continue
        result[field] = value if field in _PLAIN_CREDENTIAL_FIELDS else SecretStr(value)
    return result


@dataclass(frozen=True)
class TenantWorkspace:
    root: Path
    engine: Engine
    settings: Settings


class TenantWorkspaceRegistry:
    """Provision and cache one local workspace per opaque account storage key."""

    def __init__(self, settings: Settings) -> None:
        self._base = (settings.data_dir / "users").resolve()
        self._base.mkdir(parents=True, exist_ok=True)
        self._source_config = settings.config_dir.resolve()
        self._settings = settings
        self._items: dict[str, TenantWorkspace] = {}
        self._lock = threading.Lock()

    def get(self, storage_key: str) -> TenantWorkspace:
        if not storage_key or any(
            character not in "0123456789abcdef-" for character in storage_key
        ):
            raise ValueError("Invalid account storage key")
        with self._lock:
            existing = self._items.get(storage_key)
            if existing is not None:
                return existing
            root = (self._base / storage_key).resolve()
            if self._base not in root.parents:
                raise ValueError("Account workspace escaped the configured root")
            config_dir = root / "config"
            if not config_dir.exists():
                shutil.copytree(self._source_config, config_dir)
            for directory in (root / "data", root / "drafts", root / "logs"):
                directory.mkdir(parents=True, exist_ok=True)
            database_path = (root / "data" / "workspace.db").resolve()
            database_url = f"sqlite:///{database_path.as_posix()}"
            alembic = Config(str(resolve_alembic_config_path()))
            alembic.set_main_option("sqlalchemy.url", database_url)
            command.upgrade(alembic, "head")
            engine = build_engine(database_url)
            tenant_settings = self._settings.model_copy(
                update={
                    "database_url": database_url,
                    "data_dir": root / "data",
                    "drafts_dir": root / "drafts",
                    "logs_dir": root / "logs",
                    "config_dir": config_dir,
                    **_account_credentials(root / ".env"),
                }
            )
            workspace = TenantWorkspace(root=root, engine=engine, settings=tenant_settings)
            self._items[storage_key] = workspace
            return workspace

    def dispose(self) -> None:
        with self._lock:
            for workspace in self._items.values():
                workspace.engine.dispose()
            self._items.clear()
