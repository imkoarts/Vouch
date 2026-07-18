"""Native desktop shell for the shared local Vouch dashboard."""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import shutil
import socket
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

_LOGGER = logging.getLogger("xbot.desktop")
_LOCAL_OPENER = build_opener(ProxyHandler({}))


def application_root() -> Path:
    """Return the shared writable project root or the standalone EXE directory.

    A locally built one-folder EXE lives in ``project/dist/Vouch``. In that
    layout it deliberately reuses the source project's ``.env``, configuration,
    database, drafts, and logs so the browser and EXE views stay synchronized.
    Once the built folder is copied elsewhere, the EXE becomes self-contained and
    uses files beside the executable instead.
    """

    if getattr(sys, "frozen", False):
        executable_root = Path(sys.executable).resolve().parent
        project_root = executable_root.parent.parent
        if (
            executable_root.parent.name.casefold() == "dist"
            and (project_root / "launcher.py").is_file()
            and (project_root / "config").is_dir()
        ):
            return project_root
        return executable_root
    return Path(__file__).resolve().parent


def bundle_root() -> Path:
    """Return the source or PyInstaller bundle resource root."""

    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return Path(str(bundled)).resolve()
    return Path(__file__).resolve().parent


def prepare_environment() -> Path:
    """Prepare writable runtime paths before application settings are imported."""

    root = application_root()
    os.chdir(root)
    config_target = root / "config"
    if not config_target.is_dir():
        candidates = (
            bundle_root() / "config",
            bundle_root() / "app" / "_bundled" / "config",
        )
        source = next((candidate for candidate in candidates if candidate.is_dir()), None)
        if source is None:
            raise RuntimeError("Bundled default configuration is missing")
        shutil.copytree(source, config_target)

    os.environ.setdefault("CONFIG_DIR", str(config_target))
    os.environ.setdefault("DATA_DIR", str(root / "data"))
    os.environ.setdefault("DRAFTS_DIR", str(root / "drafts"))
    os.environ.setdefault("LOGS_DIR", str(root / "logs"))
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{(root / 'data' / 'app.db').as_posix()}")
    return root


def apply_migrations() -> None:
    """Upgrade the local database before the dashboard server starts."""

    from alembic.config import Config

    from alembic import command
    from app.config import get_settings
    from app.resources import resolve_alembic_config_path

    get_settings.cache_clear()
    settings = get_settings()
    settings.ensure_directories()
    configuration = Config(str(resolve_alembic_config_path()))
    configuration.set_main_option("sqlalchemy.url", str(settings.database_url))
    command.upgrade(configuration, "head")


def configure_desktop_logging(root: Path) -> Path:
    """Write desktop and embedded-server diagnostics to a durable local file."""

    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "desktop.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    _LOGGER.info("Desktop launcher starting; executable=%s", sys.executable)
    return log_path


def _url_host(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def server_is_live(base_url: str, timeout: float = 0.75) -> bool:
    """Check only process liveness and always bypass system/outbound proxies.

    The desktop shell previously polled ``/health/ready``. That endpoint intentionally
    performs a SQLite write probe and can return 503 while a legitimate background job
    briefly owns the database write lock. The HTTP server is already usable in that state,
    so desktop startup must use the liveness contract instead.
    """

    try:
        request = f"{base_url}/health/live"
        with _LOCAL_OPENER.open(request, timeout=timeout) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return payload == {"status": "alive"}
    except (OSError, URLError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def port_is_open(host: str, port: int, timeout: float = 0.35) -> bool:
    """Return whether another process currently accepts TCP connections on the port."""

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class EmbeddedServer:
    """Run Uvicorn in-process so closing the desktop window cannot leave a child process."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.thread: threading.Thread | None = None
        self.server: Any = None
        self.owned = False
        self.startup_error: BaseException | None = None

    @property
    def base_url(self) -> str:
        return f"http://{_url_host(self.host)}:{self.port}"

    def start(self) -> None:
        if server_is_live(self.base_url):
            _LOGGER.info("Using an already-running Vouch server at %s", self.base_url)
            self.owned = False
            return
        if port_is_open(self.host, self.port):
            raise RuntimeError(
                f"Port {self.port} is already used by another application. "
                "Close that application or change APP_PORT in .env."
            )

        apply_migrations()
        import uvicorn

        from app.config import get_settings
        from app.main import create_app

        get_settings.cache_clear()
        settings = get_settings()
        config = uvicorn.Config(
            create_app(settings),
            host=self.host,
            port=self.port,
            reload=False,
            log_config=None,
            access_log=False,
        )
        self.server = uvicorn.Server(config)

        def run_server() -> None:
            try:
                self.server.run()
            except BaseException as exc:  # Includes Uvicorn's bind-failure SystemExit.
                self.startup_error = exc
                _LOGGER.exception("Embedded dashboard server crashed during startup")

        self.thread = threading.Thread(
            target=run_server,
            name="xbot-dashboard-server",
            daemon=True,
        )
        self.thread.start()
        self.owned = True

        deadline = time.monotonic() + 35.0
        while time.monotonic() < deadline:
            if server_is_live(self.base_url):
                _LOGGER.info("Embedded dashboard server is live at %s", self.base_url)
                return
            if self.thread is not None and not self.thread.is_alive():
                break
            time.sleep(0.15)

        startup_error = self.startup_error
        self.stop()
        if startup_error is not None:
            raise RuntimeError(
                "The local dashboard server crashed: "
                f"{type(startup_error).__name__}: {startup_error}"
            ) from startup_error
        raise RuntimeError(
            "The local dashboard HTTP server did not answer its liveness check within 35 seconds. "
            "See logs/desktop.log for details."
        )

    def stop(self) -> None:
        if not self.owned:
            return
        server = self.server
        if server is not None:
            server.should_exit = True
        thread = self.thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=8.0)
        if thread is not None and thread.is_alive() and server is not None:
            server.force_exit = True
            thread.join(timeout=2.0)
        self.owned = False


def run_desktop() -> int:
    root = prepare_environment()
    configure_desktop_logging(root)

    try:
        import webview
    except ImportError as exc:
        raise RuntimeError(
            "Desktop support is not installed. Run START_DESKTOP.bat or BUILD_EXE.bat."
        ) from exc

    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    server = EmbeddedServer(settings.app_host, settings.app_port)
    server.start()
    atexit.register(server.stop)

    window = webview.create_window(
        "Vouch",
        f"{server.base_url}/?desktop=1",
        width=1380,
        height=840,
        min_size=(1080, 700),
        resizable=True,
        fullscreen=False,
        maximized=False,
        frameless=False,
        confirm_close=False,
        background_color="#f4f6f8",
    )
    window.events.closed += server.stop
    try:
        webview.start(debug=False)
    finally:
        server.stop()
    return 0


def main() -> int:
    try:
        return run_desktop()
    except Exception as exc:
        with contextlib.suppress(Exception):
            root = application_root()
            log_dir = root / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / "desktop.log").open("a", encoding="utf-8") as handle:
                handle.write("\nFATAL DESKTOP STARTUP ERROR\n")
                traceback.print_exc(file=handle)
        _LOGGER.exception("Could not start the desktop app")
        try:
            import tkinter as tk
            from tkinter import messagebox

            tk_root = tk.Tk()
            tk_root.withdraw()
            messagebox.showerror(
                "Vouch",
                f"Could not start the desktop app.\n\n{exc}\n\nDetails: logs\\desktop.log",
            )
            tk_root.destroy()
        except Exception:
            print(f"Could not start the desktop app: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
