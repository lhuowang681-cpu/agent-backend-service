from __future__ import annotations

from datetime import UTC, datetime

from redis import Redis
from redis.exceptions import ResponseError

from job_agent.backend_service.contracts import (
    RunDispatchMessage,
    StreamDelivery,
)


class RedisStreamDispatchQueue:
    """At-least-once wake-up transport; PostgreSQL remains authoritative."""

    def __init__(
        self,
        redis_url: str,
        *,
        stream_name: str = "job-agent:runs",
        group_name: str = "agent-workers",
    ) -> None:
        self._redis = Redis.from_url(redis_url, decode_responses=True)
        self.stream_name = stream_name
        self.group_name = group_name

    def close(self) -> None:
        self._redis.close()

    def ping(self) -> bool:
        return bool(self._redis.ping())

    def ensure_group(self) -> None:
        try:
            self._redis.xgroup_create(
                self.stream_name,
                self.group_name,
                id="0-0",
                mkstream=True,
            )
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def publish(self, message: RunDispatchMessage) -> str:
        return self._redis.xadd(
            self.stream_name,
            {
                "message_id": message.message_id,
                "user_id": message.user_id,
                "run_id": message.run_id,
                "dispatch_generation": str(message.dispatch_generation),
                "enqueued_at": message.enqueued_at.astimezone(UTC).isoformat(),
            },
        )

    def receive(
        self,
        *,
        consumer_name: str,
        block_ms: int = 1000,
    ) -> StreamDelivery | None:
        response = self._redis.xreadgroup(
            self.group_name,
            consumer_name,
            {self.stream_name: ">"},
            count=1,
            block=block_ms,
        )
        if not response:
            return None
        _, entries = response[0]
        stream_message_id, fields = entries[0]
        return self._delivery(stream_message_id, fields)

    def reclaim(
        self,
        *,
        consumer_name: str,
        min_idle_ms: int,
        count: int = 10,
    ) -> list[StreamDelivery]:
        response = self._redis.xautoclaim(
            self.stream_name,
            self.group_name,
            consumer_name,
            min_idle_ms,
            "0-0",
            count=count,
        )
        entries = response[1]
        return [self._delivery(stream_id, fields) for stream_id, fields in entries]

    def ack(self, stream_message_id: str) -> int:
        return int(
            self._redis.xack(self.stream_name, self.group_name, stream_message_id)
        )

    def depth(self) -> int:
        """Consumer-group lag: entries not delivered to any consumer yet."""
        try:
            groups = self._redis.xinfo_groups(self.stream_name)
        except ResponseError:
            return 0
        for group in groups:
            if group["name"] == self.group_name:
                return int(group.get("lag") or 0)
        return 0

    def retained_entries(self) -> int:
        """Physical Stream length; ACK does not reduce this retention metric."""
        return int(self._redis.xlen(self.stream_name))

    def pending(self) -> int:
        summary = self._redis.xpending(self.stream_name, self.group_name)
        return int(summary["pending"])

    @staticmethod
    def _delivery(stream_message_id: str, fields: dict[str, str]) -> StreamDelivery:
        return StreamDelivery(
            stream_message_id=stream_message_id,
            message=RunDispatchMessage(
                message_id=fields["message_id"],
                user_id=fields["user_id"],
                run_id=fields["run_id"],
                dispatch_generation=int(fields["dispatch_generation"]),
                enqueued_at=datetime.fromisoformat(fields["enqueued_at"]),
            ),
        )
