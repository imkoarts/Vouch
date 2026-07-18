"""Process-local pacing and quota circuit breakers for outbound AI requests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

SleepCallable = Callable[[float], Awaitable[None]]


@dataclass(slots=True)
class _LoopState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    operation_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    last_started: dict[str, float] = field(default_factory=dict)
    quota_blocked_until: dict[str, float] = field(default_factory=dict)


_STATES: WeakKeyDictionary[asyncio.AbstractEventLoop, _LoopState] = WeakKeyDictionary()


def _state() -> tuple[asyncio.AbstractEventLoop, _LoopState]:
    loop = asyncio.get_running_loop()
    state = _STATES.get(loop)
    if state is None:
        state = _LoopState()
        _STATES[loop] = state
    return loop, state


async def wait_for_request_slot(
    key: str,
    *,
    minimum_interval_seconds: float,
    pre_request_delay_seconds: float,
    sleep: SleepCallable = asyncio.sleep,
) -> float:
    """Serialize request starts and return the total delay applied."""

    loop, state = _state()
    waited = 0.0
    async with state.lock:
        previous = state.last_started.get(key)
        if previous is not None:
            remaining = minimum_interval_seconds - (loop.time() - previous)
            if remaining > 0:
                await sleep(remaining)
                waited += remaining
        if pre_request_delay_seconds > 0:
            await sleep(pre_request_delay_seconds)
            waited += pre_request_delay_seconds
        state.last_started[key] = loop.time()
    return waited


@asynccontextmanager
async def serialized_operation(key: str) -> AsyncIterator[None]:
    """Prevent overlapping paid workflows with the same process-local key."""

    _loop, state = _state()
    lock = state.operation_locks.setdefault(key, asyncio.Lock())
    async with lock:
        yield


def mark_quota_cooldown(key: str, *, cooldown_seconds: float) -> None:
    """Block repeated provider attempts in the current process for a fixed window."""

    loop, state = _state()
    state.quota_blocked_until[key] = max(
        state.quota_blocked_until.get(key, 0.0),
        loop.time() + max(cooldown_seconds, 0.0),
    )


def quota_cooldown_remaining(key: str) -> float:
    """Return remaining cooldown seconds, or zero when the provider is available."""

    loop, state = _state()
    remaining = state.quota_blocked_until.get(key, 0.0) - loop.time()
    if remaining <= 0:
        state.quota_blocked_until.pop(key, None)
        return 0.0
    return remaining


def reset_request_pacing_for_tests() -> None:
    """Clear process-local state for deterministic test isolation."""

    _STATES.clear()
