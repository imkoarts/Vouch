from __future__ import annotations

import logging
import sys
import unittest

from app.domain.security import redact_secrets
from app.utils.logging import RecursiveRedactionFilter


class StandardLoggingRedactionTests(unittest.TestCase):
    def test_filter_redacts_format_args_extras_and_exception_messages(self) -> None:
        private_exception_text = "private draft source body"
        try:
            raise ValueError(private_exception_text)
        except ValueError:
            exception_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="API_KEY=%s",
            args=("format-secret",),
            exc_info=exception_info,
        )
        record.api_token = "extra-secret"

        self.assertTrue(RecursiveRedactionFilter().filter(record))
        rendered = logging.Formatter("%(message)s %(api_token)s").format(record)

        self.assertNotIn("format-secret", rendered)
        self.assertNotIn("extra-secret", rendered)
        self.assertNotIn(private_exception_text, rendered)
        self.assertIn("exception details redacted", rendered)

    def test_raw_telegram_token_is_redacted_inside_request_urls(self) -> None:
        token = "1234567890:" + "abcdefghijklmnopqrstuvwxyz_ABCDE"
        rendered = redact_secrets(f"POST https://api.telegram.org/bot{token}/getMe")

        self.assertIsInstance(rendered, str)
        self.assertNotIn(token, rendered)
        self.assertIn("***REDACTED***", rendered)


if __name__ == "__main__":
    unittest.main()


class ConsoleColorTests(unittest.TestCase):
    def test_console_formatter_uses_requested_level_and_system_colors(self) -> None:
        from app.utils.logging import ColorConsoleFormatter

        formatter = ColorConsoleFormatter("%(message)s", use_color=True)
        cases = (
            (logging.ERROR, "app.service", ColorConsoleFormatter.RED),
            (logging.WARNING, "app.service", ColorConsoleFormatter.YELLOW),
            (logging.INFO, "app.service", ColorConsoleFormatter.GREEN),
            (logging.INFO, "httpx", ColorConsoleFormatter.GRAY),
        )
        for level, name, expected in cases:
            record = logging.LogRecord(name, level, __file__, 1, "message", (), None)
            self.assertTrue(formatter.format(record).startswith(expected))


def test_console_logging_uses_stderr_so_cli_stdout_remains_machine_readable(tmp_path) -> None:
    from app.utils.logging import configure_logging

    root = logging.getLogger()
    previous = list(root.handlers)
    try:
        root.handlers.clear()
        configure_logging("INFO", tmp_path, color=False)
        stream_handlers = [
            handler
            for handler in root.handlers
            if isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
        ]
        assert stream_handlers
        assert all(handler.stream is sys.stderr for handler in stream_handlers)
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers[:] = previous
