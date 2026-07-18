"""Structured logging with mandatory recursive secret redaction."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import MutableMapping
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import structlog

from app.domain.security import redact_secrets


class ColorConsoleFormatter(logging.Formatter):
    """Readable ANSI console colors while file logs remain plain text."""

    RESET = "\x1b[0m"
    RED = "\x1b[31m"
    YELLOW = "\x1b[33m"
    GREEN = "\x1b[32m"
    GRAY = "\x1b[90m"

    _SYSTEM_LOGGERS = ("uvicorn", "httpx", "httpcore", "asyncio")

    def __init__(self, *args: Any, use_color: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.use_color = use_color and os.environ.get("NO_COLOR") is None

    def _color(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.ERROR:
            return self.RED
        if record.levelno >= logging.WARNING:
            return self.YELLOW
        if record.name.startswith(self._SYSTEM_LOGGERS):
            return self.GRAY
        if record.levelno >= logging.INFO:
            return self.GREEN
        return self.GRAY

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        if not self.use_color:
            return rendered
        return f"{self._color(record)}{rendered}{self.RESET}"


def _enable_windows_ansi() -> None:
    """Enable virtual-terminal colors in classic Windows consoles when possible."""

    if os.name != "nt":
        return
    try:
        import ctypes

        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return
        kernel32 = windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except (AttributeError, OSError):
        return


class XBotConsoleHandler(logging.StreamHandler):
    """Console handler owned by this module, distinct from pytest/Click capture handlers."""


class RecursiveRedactionFilter(logging.Filter):
    """Apply the same fail-closed redaction to standard-library log records."""

    _SPECIAL_FIELDS = frozenset({"args", "exc_info", "exc_text", "msg", "stack_info"})

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered_message = record.getMessage()
        except (TypeError, ValueError):
            rendered_message = str(record.msg)
        safe_message = redact_secrets(rendered_message)
        record.msg = safe_message if isinstance(safe_message, str) else "log_redaction_failed"
        record.args = ()

        for name, value in tuple(vars(record).items()):
            if name in self._SPECIAL_FIELDS:
                continue
            redacted = redact_secrets({name: value})
            if isinstance(redacted, dict):
                setattr(record, name, redacted[name])

        if record.exc_info is not None:
            exception_type = record.exc_info[0]
            type_name = getattr(exception_type, "__name__", "Exception")
            record.exc_info = None
            record.exc_text = f"{type_name}: exception details redacted"
        elif record.exc_text:
            record.exc_text = "Exception: exception details redacted"
        if record.stack_info:
            redacted_stack = redact_secrets(record.stack_info)
            record.stack_info = (
                redacted_stack if isinstance(redacted_stack, str) else "stack redacted"
            )
        return True


def redact_event(
    logger: Any,
    method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Structlog processor that redacts credentials before rendering."""

    del logger, method_name
    redacted = redact_secrets(dict(event_dict))
    if not isinstance(redacted, dict):
        return {"event": "log_redaction_failed"}
    event_dict.clear()
    event_dict.update(redacted)
    return event_dict


def configure_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    *,
    color: bool = True,
) -> None:
    """Configure redacted console and rotating local-file logs."""

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _enable_windows_ansi()
    console_formatter = ColorConsoleFormatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        use_color=color,
    )
    stream_handlers = [
        handler for handler in root_logger.handlers if isinstance(handler, logging.StreamHandler)
    ]
    if not stream_handlers:
        console = XBotConsoleHandler(sys.stderr)
        console.setFormatter(console_formatter)
        root_logger.addHandler(console)
    else:
        # Never reconfigure pytest/Click capture handlers. Their streams are temporary and may be
        # closed after a CLI invocation. Only handlers created by this module are authoritative.
        for handler in stream_handlers:
            if isinstance(handler, XBotConsoleHandler):
                handler.setFormatter(console_formatter)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        target = (log_dir / "vouch.log").resolve()
        has_target = any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == target
            for handler in root_logger.handlers
        )
        if not has_target:
            file_handler = RotatingFileHandler(
                target, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    for root_handler in root_logger.handlers:
        if not any(isinstance(item, RecursiveRedactionFilter) for item in root_handler.filters):
            root_handler.addFilter(RecursiveRedactionFilter())
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_event,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "vouch") -> Any:
    return structlog.get_logger(name)
