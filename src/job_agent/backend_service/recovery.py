from __future__ import annotations

from dataclasses import dataclass

from job_agent.backend_service.outbox import OutboxPublisher
from job_agent.backend_service.postgres_repository import PostgresRunRepository
from job_agent.backend_service.redis_streams import RedisStreamDispatchQueue


@dataclass(frozen=True)
class RecoverySummary:
    published: int
    expired_attempts_requeued: int
    dispatches_reconciled: int


class RecoveryCoordinator:
    """Turns PostgreSQL evidence into bounded, auditable recovery actions."""

    def __init__(
        self,
        *,
        repository: PostgresRunRepository,
        dispatch_queue: RedisStreamDispatchQueue,
        publisher: OutboxPublisher,
        max_attempts: int,
        dispatch_reconcile_grace_seconds: float = 30.0,
    ) -> None:
        self.repository = repository
        self.dispatch_queue = dispatch_queue
        self.publisher = publisher
        self.max_attempts = max_attempts
        self.dispatch_reconcile_grace_seconds = dispatch_reconcile_grace_seconds

    def run_once(self, *, limit: int = 100) -> RecoverySummary:
        self.dispatch_queue.ensure_group()
        published = self.publisher.publish_once(limit=limit)
        recovered = self.repository.recover_expired_attempts(
            limit=limit,
            max_attempts=self.max_attempts,
        )
        reconciled = self.repository.reconcile_ready_queued_runs(
            limit=limit,
            grace_seconds=self.dispatch_reconcile_grace_seconds,
        )
        published += self.publisher.publish_once(limit=limit)
        return RecoverySummary(
            published=published,
            expired_attempts_requeued=recovered,
            dispatches_reconciled=reconciled,
        )
