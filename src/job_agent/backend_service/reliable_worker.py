from __future__ import annotations

from threading import Event, Thread
from typing import Callable

import psycopg
from psycopg_pool import PoolTimeout

from job_agent.backend_service.contracts import (
    AgentApprovalPause,
    ClaimedRun,
    StreamDelivery,
)
from job_agent.backend_service.execution import AgentExecutionAdapter, AgentExecutionFailedError
from job_agent.backend_service.postgres_repository import (
    PostgresRunRepository,
    RunDispatchDeferredError,
    StaleAttemptError,
)
from job_agent.backend_service.redis_streams import RedisStreamDispatchQueue
from job_agent.backend_service.repository import QueueOwnershipError
from job_agent.backend_service.failures import FailureClassifier, FailureDisposition
from job_agent.backend_service.provider_admission import (
    ProviderAdmissionRejected,
)
from job_agent.backend_service.metrics import MetricsRecorder, NullMetricsRecorder, Timer


class _LeaseHeartbeat:
    def __init__(
        self,
        *,
        repository: PostgresRunRepository,
        claim: ClaimedRun,
        lease_seconds: float,
    ) -> None:
        self.repository = repository
        self.claim = claim
        self.lease_seconds = lease_seconds
        self.stop_event = Event()
        self.lease_lost = Event()
        self.thread = Thread(target=self._run, name="run-lease-heartbeat", daemon=True)

    def __enter__(self) -> "_LeaseHeartbeat":
        self.thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(1.0, self.lease_seconds))

    def _run(self) -> None:
        interval = max(0.1, self.lease_seconds / 3)
        while not self.stop_event.wait(interval):
            try:
                renewed = self.repository.heartbeat(
                    self.claim,
                    lease_seconds=self.lease_seconds,
                )
            except (psycopg.Error, PoolTimeout):
                self.lease_lost.set()
                return
            if not renewed:
                self.lease_lost.set()
                return


class ReliableWorker:
    """One Redis delivery at a time with DB claim, lease heartbeat and fencing."""

    def __init__(
        self,
        *,
        repository: PostgresRunRepository,
        dispatch_queue: RedisStreamDispatchQueue,
        execution_adapter: AgentExecutionAdapter,
        worker_id: str,
        lease_seconds: float = 30.0,
        failure_classifier: FailureClassifier | None = None,
        failpoint: Callable[[str], None] | None = None,
        metrics_recorder: MetricsRecorder | None = None,
    ) -> None:
        self.repository = repository
        self.dispatch_queue = dispatch_queue
        self.execution_adapter = execution_adapter
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.failure_classifier = failure_classifier or FailureClassifier()
        self.failpoint = failpoint
        self.metrics_recorder = metrics_recorder or NullMetricsRecorder()
        self.rejected_message_count = 0

    def process(self, delivery: StreamDelivery) -> None:
        try:
            claim = self.repository.claim(
                delivery.message,
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
            )
        except QueueOwnershipError:
            self.rejected_message_count += 1
            self.dispatch_queue.ack(delivery.stream_message_id)
            return
        except RunDispatchDeferredError:
            return
        if claim is None:
            self.dispatch_queue.ack(delivery.stream_message_id)
            return

        task_timer = Timer()
        self.repository.mark_agent_active(claim)
        metric_outcome = "crashed"
        try:
            with _LeaseHeartbeat(
                repository=self.repository,
                claim=claim,
                lease_seconds=self.lease_seconds,
            ) as heartbeat:
                outcome = self.execution_adapter.execute(
                    claim.run,
                    progress_callback=lambda stage: self.repository.record_progress(
                        claim,
                        stage=stage,
                    ),
                    claim=claim,
                )
                if heartbeat.lease_lost.is_set():
                    raise StaleAttemptError("lease_lost_during_execution")
                if self.failpoint is not None:
                    self.failpoint("after_agent_before_commit")
                if isinstance(outcome, AgentApprovalPause):
                    self.repository.wait_for_approval(claim, pause=outcome)
                    metric_outcome = "waiting_approval"
                else:
                    self.repository.succeed(
                        claim,
                        result=outcome.result,
                        provider_call_count=outcome.provider_call_count,
                    )
                    metric_outcome = "succeeded"
        except ProviderAdmissionRejected as exc:
            self.repository.defer_for_backpressure(
                claim,
                error_code=exc.error_code,
                delay_seconds=exc.retry_after_seconds,
            )
            metric_outcome = "backpressured"
        except AgentExecutionFailedError as exc:
            decision = self.failure_classifier.classify(
                error_code=exc.error_code,
                attempt_no=claim.attempt_no,
            )
            if decision.disposition == FailureDisposition.RETRY:
                assert decision.retry_delay_seconds is not None
                self.repository.retry(
                    claim,
                    error_code=decision.error_code,
                    delay_seconds=decision.retry_delay_seconds,
                )
                metric_outcome = "retry"
            elif decision.disposition == FailureDisposition.UNCERTAIN:
                self.repository.mark_uncertain(claim, error_code=decision.error_code)
                metric_outcome = "uncertain"
            else:
                self.repository.fail(claim, error_code=decision.error_code)
                metric_outcome = "failed"
        except StaleAttemptError:
            metric_outcome = "stale"
            self.dispatch_queue.ack(delivery.stream_message_id)
            return
        finally:
            self.metrics_recorder.record_worker_task(
                outcome=metric_outcome,
                duration_seconds=task_timer.elapsed(),
            )
        self.dispatch_queue.ack(delivery.stream_message_id)

    def receive_and_process(self, *, block_ms: int = 1000) -> bool:
        delivery = self.dispatch_queue.receive(
            consumer_name=self.worker_id,
            block_ms=block_ms,
        )
        if delivery is None:
            return False
        self.process(delivery)
        return True

    def reclaim_and_process(self, *, min_idle_ms: int) -> int:
        deliveries = self.dispatch_queue.reclaim(
            consumer_name=self.worker_id,
            min_idle_ms=min_idle_ms,
        )
        for delivery in deliveries:
            self.process(delivery)
        return len(deliveries)
