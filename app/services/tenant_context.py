"""Request-local workspace routing for authenticated dashboard users."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.schemas.configuration import VoiceProfileConfiguration

_engine: ContextVar[Engine | None] = ContextVar("vouch_tenant_engine", default=None)
_settings: ContextVar[Settings | None] = ContextVar("vouch_tenant_settings", default=None)
_voice: ContextVar[VoiceProfileConfiguration | None] = ContextVar(
    "vouch_tenant_voice", default=None
)


def bind_tenant(
    engine: Engine,
    settings: Settings,
    voice_profile: VoiceProfileConfiguration | None = None,
) -> tuple[
    Token[Engine | None],
    Token[Settings | None],
    Token[VoiceProfileConfiguration | None],
]:
    """Bind a tenant workspace for the current request/task context."""

    return _engine.set(engine), _settings.set(settings), _voice.set(voice_profile)


def reset_tenant(
    tokens: tuple[
        Token[Engine | None],
        Token[Settings | None],
        Token[VoiceProfileConfiguration | None],
    ],
) -> None:
    _engine.reset(tokens[0])
    _settings.reset(tokens[1])
    _voice.reset(tokens[2])


def active_voice_profile(
    fallback: VoiceProfileConfiguration,
) -> VoiceProfileConfiguration:
    return _voice.get() or fallback


def active_engine(fallback: Engine) -> Engine:
    return _engine.get() or fallback


def bound_engine() -> Engine | None:
    """Return the request-bound engine without constructing a global fallback."""

    return _engine.get()


def active_settings(fallback: Settings) -> Settings:
    return _settings.get() or fallback


class ContextSessionFactory:
    """A sessionmaker-shaped adapter that resolves the active engine per call."""

    def __init__(self, fallback: Engine) -> None:
        self._fallback = fallback

    def _factory(self) -> sessionmaker[Session]:
        return sessionmaker(
            bind=active_engine(self._fallback),
            class_=Session,
            autoflush=False,
            expire_on_commit=False,
        )

    def __call__(self) -> Session:
        return self._factory()()

    def begin(self) -> Any:
        return self._factory().begin()


class ContextSettings:
    """Attribute proxy for code that was written against one Settings instance."""

    def __init__(self, fallback: Settings) -> None:
        self._fallback = fallback

    def __getattr__(self, name: str) -> Any:
        return getattr(active_settings(self._fallback), name)
