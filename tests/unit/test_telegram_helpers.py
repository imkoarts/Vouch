from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.config import Settings
from app.telegram.api import TelegramApiError
from app.telegram.bot import TelegramReviewBot, _button, _split_text
from app.telegram.state import TelegramStateStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_split_text_preserves_content_and_respects_limit() -> None:
    text = "First paragraph.\n" + ("word " * 40) + "\nFinal."

    chunks = _split_text(text, limit=70)

    assert all(len(chunk) <= 70 for chunk in chunks)
    assert " ".join(" ".join(chunks).split()) == " ".join(text.split())


def test_callback_button_enforces_telegram_byte_limit() -> None:
    assert _button("OK", "save|draft") == {"text": "OK", "callback_data": "save|draft"}
    with pytest.raises(ValueError, match="64 bytes"):
        _button("Too long", "x" * 65)


def test_telegram_state_round_trip_is_atomic(tmp_path: Path) -> None:
    store = TelegramStateStore(tmp_path / "state.json")

    store.set_offset(12)
    store.set_pending(123456789, action="new_topic", draft_id="draft-1")

    assert store.offset() == 12
    assert store.pop_pending(123456789) == {
        "action": "new_topic",
        "draft_id": "draft-1",
    }
    assert store.pop_pending(123456789) is None
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.asyncio
async def test_polling_retries_after_initial_telegram_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = PROJECT_ROOT
    stop = asyncio.Event()

    class RetryApi:
        def __init__(self) -> None:
            self.get_me_calls = 0
            self.delete_calls = 0

        async def get_me(self) -> dict[str, object]:
            self.get_me_calls += 1
            if self.get_me_calls == 1:
                raise TelegramApiError("synthetic startup failure")
            return {"id": 1}

        async def delete_webhook(self, *, drop_pending_updates: bool = False) -> None:
            del drop_pending_updates
            self.delete_calls += 1

        async def get_updates(self, *, offset: int | None, poll_timeout: int) -> tuple[()]:
            del offset, poll_timeout
            stop.set()
            return ()

    api = RetryApi()
    settings = Settings(
        _env_file=None,
        config_dir=project_root / "config",
        data_dir=tmp_path / "data",
        drafts_dir=tmp_path / "drafts",
    )
    bot = TelegramReviewBot(settings, api)  # type: ignore[arg-type]

    async def no_wait(_stop_event: asyncio.Event | None, _seconds: float) -> None:
        return None

    monkeypatch.setattr(bot, "_sleep_or_stop", no_wait)
    await bot.run_forever(stop)

    assert api.get_me_calls == 2
    assert api.delete_calls == 1


@pytest.mark.asyncio
async def test_telegram_transport_error_never_exposes_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from app.telegram.api import TelegramApiError, TelegramBotApi

    token = "synthetic-secret-token"
    api = TelegramBotApi(token)

    async def fail_post(*_args: object, **_kwargs: object) -> object:
        request = httpx.Request("POST", f"https://api.telegram.org/bot{token}/getMe")
        raise httpx.ConnectError("failed", request=request)

    monkeypatch.setattr(api._client, "post", fail_post)
    try:
        with pytest.raises(TelegramApiError) as raised:
            await api.get_me()
    finally:
        await api.close()

    assert token not in str(raised.value)
    assert str(raised.value) == "Telegram transport request failed"
