from __future__ import annotations

from redis.exceptions import RedisError

from job_agent.backend_service.postgres_repository import PostgresRunRepository
from job_agent.backend_service.redis_streams import RedisStreamDispatchQueue


class OutboxPublisher:
    def __init__(
        self,
        *,
        repository: PostgresRunRepository,
        dispatch_queue: RedisStreamDispatchQueue,
    ) -> None:
        self.repository = repository
        self.dispatch_queue = dispatch_queue

    def publish_once(self, *, limit: int = 100) -> int:
        published = 0
        for message in self.repository.pending_dispatches(limit=limit):
            try:
                self.dispatch_queue.publish(message)
            except RedisError:
                self.repository.mark_dispatch_failed(
                    message_id=message.message_id,
                    error_code="redis_unavailable",
                )
                break
            self.repository.mark_dispatch_published(message_id=message.message_id)
            published += 1
        return published
