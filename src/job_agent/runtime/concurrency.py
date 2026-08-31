from __future__ import annotations

import threading
from collections.abc import Callable
from time import perf_counter
from typing import TypeVar


T = TypeVar("T")


class ExecutionCoordinator:
    """Instance-local cross-session semaphore, per-session lock, and idempotency cache."""

    def __init__(self, max_parallel_sessions: int = 4) -> None:
        if max_parallel_sessions <= 0:
            raise ValueError("max_parallel_sessions must be positive")
        self._semaphore = threading.BoundedSemaphore(max_parallel_sessions)
        self._guard = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}
        self._cache: dict[str, T] = {}

    def execute(self, *, session_id: str, idempotency_key: str, operation: Callable[[], T]) -> tuple[T, int, bool]:
        with self._guard:
            if idempotency_key in self._cache:
                return self._cache[idempotency_key], 0, True
            session_lock = self._session_locks.setdefault(session_id, threading.Lock())
        started = perf_counter()
        with self._semaphore, session_lock:
            queue_wait_ms = max(int((perf_counter() - started) * 1000), 0)
            with self._guard:
                if idempotency_key in self._cache:
                    return self._cache[idempotency_key], queue_wait_ms, True
            value = operation()
            with self._guard:
                self._cache[idempotency_key] = value
            return value, queue_wait_ms, False
