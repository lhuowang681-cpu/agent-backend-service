from __future__ import annotations

from datetime import UTC, datetime
from threading import RLock
from uuid import uuid4

from job_agent.backend_service.contracts import (
    CreateRunRequest,
    DebugEventView,
    ExecutionManifest,
    EventPage,
    RunDebugView,
    RunDispatchMessage,
    RunEventView,
    RunRecord,
    RunStatus,
)


class RunNotFoundError(LookupError):
    pass


class IdempotencyConflictError(ValueError):
    pass


class QueueCapacityExceededError(RuntimeError):
    def __init__(self, scope: str, *, retry_after_seconds: float = 1.0) -> None:
        super().__init__(f"queued_run_capacity_exceeded:{scope}")
        self.scope = scope
        self.retry_after_seconds = retry_after_seconds


class QueueOwnershipError(ValueError):
    pass


class InvalidRunTransitionError(RuntimeError):
    pass


class InMemoryRunRepository:
    """Thread-safe tracer repository with user-scoped access at every public seam."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._runs: dict[str, RunRecord] = {}
        self._idempotency: dict[tuple[str, str], str] = {}
        self._events: dict[str, list[RunEventView]] = {}

    def create_or_get(
        self,
        *,
        user_id: str,
        request_id: str,
        idempotency_key: str,
        request_hash: str,
        request: CreateRunRequest,
        execution_manifest: ExecutionManifest | None = None,
        max_queued_global: int | None = None,
        max_queued_per_user: int | None = None,
    ) -> tuple[RunRecord, bool]:
        identity = (user_id, idempotency_key)
        with self._lock:
            existing_id = self._idempotency.get(identity)
            if existing_id is not None:
                existing = self._runs[existing_id]
                if existing.request_hash != request_hash:
                    raise IdempotencyConflictError("idempotency_key_payload_mismatch")
                return existing.model_copy(deep=True), False
            queued_global = sum(
                record.status == RunStatus.QUEUED for record in self._runs.values()
            )
            queued_user = sum(
                record.status == RunStatus.QUEUED and record.user_id == user_id
                for record in self._runs.values()
            )
            if max_queued_global is not None and queued_global >= max_queued_global:
                raise QueueCapacityExceededError("global")
            if max_queued_per_user is not None and queued_user >= max_queued_per_user:
                raise QueueCapacityExceededError("user")

            now = datetime.now(UTC)
            run_id = str(uuid4())
            record = RunRecord(
                run_id=run_id,
                user_id=user_id,
                session_id=request.session_id,
                task_type=request.task_type,
                status=RunStatus.QUEUED,
                status_version=0,
                dispatch_generation=1,
                created_request_id=request_id,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                request=request,
                execution_manifest=execution_manifest,
                created_at=now,
                updated_at=now,
            )
            self._runs[run_id] = record
            self._idempotency[identity] = run_id
            self._events[run_id] = []
            self._append_event_locked(record, "run.queued", {"status": RunStatus.QUEUED.value})
            return record.model_copy(deep=True), True

    def get_for_user(self, *, user_id: str, run_id: str) -> RunRecord:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None or record.user_id != user_id:
                raise RunNotFoundError("resource_not_found")
            return record.model_copy(deep=True)

    def claim(self, message: RunDispatchMessage) -> RunRecord | None:
        with self._lock:
            record = self._runs.get(message.run_id)
            if record is None:
                return None
            if record.user_id != message.user_id:
                raise QueueOwnershipError("queue_owner_mismatch")
            if (
                record.status != RunStatus.QUEUED
                or record.dispatch_generation != message.dispatch_generation
            ):
                return None
            now = datetime.now(UTC)
            updated = record.model_copy(
                update={
                    "status": RunStatus.RUNNING,
                    "status_version": record.status_version + 1,
                    "current_stage": "starting",
                    "updated_at": now,
                }
            )
            self._runs[record.run_id] = updated
            self._append_event_locked(updated, "run.running", {"status": RunStatus.RUNNING.value})
            return updated.model_copy(deep=True)

    def record_progress(self, *, user_id: str, run_id: str, stage: str) -> None:
        with self._lock:
            record = self._owned_running_locked(user_id=user_id, run_id=run_id)
            updated = record.model_copy(
                update={"current_stage": stage, "updated_at": datetime.now(UTC)}
            )
            self._runs[run_id] = updated
            self._append_event_locked(updated, "run.progress", {"stage": stage})

    def succeed(self, *, user_id: str, run_id: str, result: dict[str, object]) -> RunRecord:
        with self._lock:
            record = self._owned_running_locked(user_id=user_id, run_id=run_id)
            now = datetime.now(UTC)
            updated = record.model_copy(
                update={
                    "status": RunStatus.SUCCEEDED,
                    "status_version": record.status_version + 1,
                    "current_stage": "completed",
                    "result": result,
                    "updated_at": now,
                    "finished_at": now,
                }
            )
            self._runs[run_id] = updated
            self._append_event_locked(updated, "run.succeeded", {"status": RunStatus.SUCCEEDED.value})
            return updated.model_copy(deep=True)

    def fail(self, *, user_id: str, run_id: str, error_code: str) -> RunRecord:
        with self._lock:
            record = self._owned_running_locked(user_id=user_id, run_id=run_id)
            now = datetime.now(UTC)
            updated = record.model_copy(
                update={
                    "status": RunStatus.FAILED,
                    "status_version": record.status_version + 1,
                    "current_stage": "failed",
                    "error_code": error_code,
                    "updated_at": now,
                    "finished_at": now,
                }
            )
            self._runs[run_id] = updated
            self._append_event_locked(
                updated,
                "run.failed",
                {"status": RunStatus.FAILED.value, "error_code": error_code},
            )
            return updated.model_copy(deep=True)

    def list_events(
        self,
        *,
        user_id: str,
        run_id: str,
        after: int,
        limit: int,
    ) -> EventPage:
        if after < 0 or limit < 1 or limit > 500:
            raise ValueError("invalid_event_page")
        with self._lock:
            self.get_for_user(user_id=user_id, run_id=run_id)
            selected = [event for event in self._events[run_id] if event.sequence > after][:limit]
            next_after = selected[-1].sequence if selected else after
            return EventPage(items=[item.model_copy(deep=True) for item in selected], next_after=next_after)

    def get_debug_view(self, *, user_id: str, run_id: str) -> RunDebugView:
        with self._lock:
            record = self.get_for_user(user_id=user_id, run_id=run_id)
            events = [
                DebugEventView(
                    sequence=event.sequence,
                    event_type=event.event_type,
                    payload=event.payload,
                    created_at=event.created_at,
                )
                for event in self._events[run_id][-500:]
            ]
            return RunDebugView(
                run_id=record.run_id,
                session_id=record.session_id,
                created_request_id=record.created_request_id,
                task_type=record.task_type,
                status=record.status,
                status_version=record.status_version,
                current_stage=record.current_stage,
                error_code=record.error_code,
                retry_count=record.retry_count,
                execution_manifest=record.execution_manifest,
                events=events,
                events_truncated=len(self._events[run_id]) > len(events),
            )

    def _owned_running_locked(self, *, user_id: str, run_id: str) -> RunRecord:
        record = self._runs.get(run_id)
        if record is None or record.user_id != user_id:
            raise RunNotFoundError("resource_not_found")
        if record.status != RunStatus.RUNNING:
            raise InvalidRunTransitionError(
                f"run must be RUNNING, found {record.status.value}"
            )
        return record

    def _append_event_locked(
        self,
        record: RunRecord,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        events = self._events[record.run_id]
        events.append(
            RunEventView(
                sequence=len(events) + 1,
                event_type=event_type,
                payload=payload,
                created_at=datetime.now(UTC),
            )
        )
