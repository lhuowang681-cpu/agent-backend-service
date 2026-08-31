from __future__ import annotations

from queue import Empty, Queue
from threading import Lock

from job_agent.backend_service.contracts import RunDispatchMessage


class InMemoryDispatchQueue:
    """Process-local wake-up queue used only by the tracer bullet."""

    _STOP = object()

    def __init__(self) -> None:
        self._queue: Queue[RunDispatchMessage | object] = Queue()
        self._lock = Lock()
        self._closed = False

    def publish(self, message: RunDispatchMessage) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("dispatch_queue_closed")
            self._queue.put(message)

    def receive(self, *, timeout: float) -> RunDispatchMessage | None:
        try:
            item = self._queue.get(timeout=timeout)
        except Empty:
            return None
        if item is self._STOP:
            self._queue.task_done()
            return None
        return item  # type: ignore[return-value]

    def task_done(self) -> None:
        self._queue.task_done()

    def depth(self) -> int:
        return self._queue.qsize()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(self._STOP)
