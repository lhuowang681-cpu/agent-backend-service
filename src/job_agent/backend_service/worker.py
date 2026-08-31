from __future__ import annotations

from threading import Event, Lock, Thread

from job_agent.backend_service.execution import AgentExecutionAdapter, AgentExecutionFailedError
from job_agent.backend_service.queueing import InMemoryDispatchQueue
from job_agent.backend_service.repository import (
    InMemoryRunRepository,
    QueueOwnershipError,
)


class WorkerEngine:
    """Single process-local worker used to prove the asynchronous service seam."""

    def __init__(
        self,
        *,
        repository: InMemoryRunRepository,
        dispatch_queue: InMemoryDispatchQueue,
        execution_adapter: AgentExecutionAdapter,
    ) -> None:
        self.repository = repository
        self.dispatch_queue = dispatch_queue
        self.execution_adapter = execution_adapter
        self._lifecycle_lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None
        self.rejected_message_count = 0

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = Thread(
                target=self._run,
                name="job-agent-tracer-worker",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                return
            self._stop.set()
            self.dispatch_queue.close()
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise RuntimeError("worker_stop_timeout")
        with self._lifecycle_lock:
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            message = self.dispatch_queue.receive(timeout=0.1)
            if message is None:
                continue
            try:
                self._handle(message)
            finally:
                self.dispatch_queue.task_done()

    def _handle(self, message) -> None:
        try:
            record = self.repository.claim(message)
        except QueueOwnershipError:
            self.rejected_message_count += 1
            return
        if record is None:
            return
        try:
            outcome = self.execution_adapter.execute(
                record,
                progress_callback=lambda stage: self.repository.record_progress(
                    user_id=record.user_id,
                    run_id=record.run_id,
                    stage=stage,
                ),
            )
        except AgentExecutionFailedError as exc:
            self.repository.fail(
                user_id=record.user_id,
                run_id=record.run_id,
                error_code=exc.error_code,
            )
            return
        self.repository.succeed(
            user_id=record.user_id,
            run_id=record.run_id,
            result=outcome.result,
        )
