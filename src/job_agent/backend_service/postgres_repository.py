from __future__ import annotations

from uuid import uuid4

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from job_agent.agent_runtime.contracts import (
    AgentAction,
    ApprovalDecision,
    ApprovalRequest,
)
from job_agent.agent_runtime.policy import compute_action_digest
from job_agent.agent_runtime.failures import CheckpointError
from job_agent.backend_service.contracts import (
    AgentApprovalPause,
    ApprovalDecisionRequest,
    ClaimedRun,
    CreateRunRequest,
    DebugApprovalView,
    DebugAttemptView,
    DebugCheckpointView,
    DebugEventView,
    DebugToolOperationView,
    EventPage,
    ExecutionManifest,
    RunDebugView,
    RunDispatchMessage,
    RunEventView,
    RunRecord,
    RunStatus,
)
from job_agent.backend_service.career_snapshot import (
    CareerSnapshotConflictError,
    CareerSnapshotNotFoundError,
    CareerSnapshotPayload,
    CareerSnapshotRecord,
)
from job_agent.backend_service.tool_operations import (
    ToolOperationConflictError,
    ToolOperationLease,
    ToolOperationRecord,
    ToolOperationState,
)
from job_agent.backend_service.repository import (
    IdempotencyConflictError,
    InvalidRunTransitionError,
    QueueCapacityExceededError,
    QueueOwnershipError,
    RunNotFoundError,
)


class StaleAttemptError(RuntimeError):
    pass


class RunDispatchDeferredError(RuntimeError):
    """The message remains valid but cannot be ACKed yet."""

    pass


class ApprovalPersistenceError(RuntimeError):
    pass


class RunStateConflictError(RuntimeError):
    pass


class PostgresRunRepository:
    """PostgreSQL authority for Run, Attempt, Event and Outbox state."""

    def __init__(
        self,
        database_url: str,
        *,
        min_size: int = 1,
        max_size: int = 8,
        pool_timeout_seconds: float = 1.0,
    ) -> None:
        if min_size < 0 or max_size < 1 or min_size > max_size:
            raise ValueError("invalid_database_pool_size")
        if pool_timeout_seconds <= 0:
            raise ValueError("database_pool_timeout_must_be_positive")
        self._pool = ConnectionPool(
            database_url,
            min_size=min_size,
            max_size=max_size,
            timeout=pool_timeout_seconds,
            kwargs={"row_factory": dict_row},
            open=False,
        )

    def open(self) -> None:
        self._pool.open(wait=True)

    def close(self) -> None:
        self._pool.close()

    def ensure_users(self, user_ids: tuple[str, ...]) -> None:
        with self._pool.connection() as connection, connection.transaction():
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO users(user_id) VALUES (%s)
                    ON CONFLICT (user_id) DO NOTHING
                    """,
                    [(user_id,) for user_id in user_ids],
                )

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
        run_id = str(uuid4())
        message_id = str(uuid4())
        request_payload = request.model_dump(mode="json")
        with self._pool.connection() as connection, connection.transaction():
            if max_queued_global is not None or max_queued_per_user is not None:
                connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    ("job-agent:queued-run-admission",),
                )
                existing = connection.execute(
                    """
                    SELECT * FROM runs
                    WHERE user_id = %s AND idempotency_key = %s
                    """,
                    (user_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    if existing["request_hash"] != request_hash:
                        raise IdempotencyConflictError(
                            "idempotency_key_payload_mismatch"
                        )
                    return self._to_record(existing), False
                counts = connection.execute(
                    """
                    SELECT count(*) AS global_count,
                           count(*) FILTER (WHERE user_id = %s) AS user_count
                    FROM runs
                    WHERE status = 'QUEUED'
                    """,
                    (user_id,),
                ).fetchone()
                if (
                    max_queued_global is not None
                    and counts["global_count"] >= max_queued_global
                ):
                    raise QueueCapacityExceededError("global")
                if (
                    max_queued_per_user is not None
                    and counts["user_count"] >= max_queued_per_user
                ):
                    raise QueueCapacityExceededError("user")
            inserted = connection.execute(
                """
                INSERT INTO runs(
                    run_id, user_id, created_request_id, session_id, task_type,
                    status, status_version, dispatch_generation, idempotency_key,
                    request_hash, request_payload, execution_manifest
                )
                VALUES (
                    %s, %s, %s, %s, %s, 'QUEUED', 0, 1, %s, %s, %s, %s
                )
                ON CONFLICT (user_id, idempotency_key) DO NOTHING
                RETURNING *
                """,
                (
                    run_id,
                    user_id,
                    request_id,
                    request.session_id,
                    request.task_type,
                    idempotency_key,
                    request_hash,
                    Jsonb(request_payload),
                    (
                        Jsonb(execution_manifest.model_dump(mode="json"))
                        if execution_manifest is not None
                        else None
                    ),
                ),
            ).fetchone()
            if inserted is None:
                existing = connection.execute(
                    """
                    SELECT * FROM runs
                    WHERE user_id = %s AND idempotency_key = %s
                    """,
                    (user_id, idempotency_key),
                ).fetchone()
                if existing is None:
                    raise RuntimeError("idempotency_conflict_without_run")
                if existing["request_hash"] != request_hash:
                    raise IdempotencyConflictError("idempotency_key_payload_mismatch")
                return self._to_record(existing), False

            self._append_event(
                connection,
                user_id=user_id,
                run_id=run_id,
                event_type="run.queued",
                payload={"status": RunStatus.QUEUED.value},
            )
            connection.execute(
                """
                INSERT INTO outbox_messages(
                    message_id, user_id, aggregate_id, message_type,
                    dispatch_generation, payload
                )
                VALUES (%s, %s, %s, 'run.dispatch', 1, %s)
                """,
                (
                    message_id,
                    user_id,
                    run_id,
                    Jsonb(
                        {"user_id": user_id, "run_id": run_id, "dispatch_generation": 1}
                    ),
                ),
            )
            return self._to_record(inserted), True

    def put_career_snapshot(
        self,
        *,
        user_id: str,
        snapshot_revision: str,
        content_hash: str,
        payload: dict[str, object],
    ) -> CareerSnapshotRecord:
        with self._pool.connection() as connection, connection.transaction():
            inserted = connection.execute(
                """
                INSERT INTO career_snapshots(
                    user_id, snapshot_revision, content_hash, payload_private
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, snapshot_revision) DO NOTHING
                RETURNING *
                """,
                (user_id, snapshot_revision, content_hash, Jsonb(payload)),
            ).fetchone()
            row = inserted
            if row is None:
                row = connection.execute(
                    """
                    SELECT * FROM career_snapshots
                    WHERE user_id = %s AND snapshot_revision = %s
                    """,
                    (user_id, snapshot_revision),
                ).fetchone()
                if (
                    row is None
                    or row["content_hash"] != content_hash
                    or row["payload_private"] != payload
                ):
                    raise CareerSnapshotConflictError(
                        "career_snapshot_revision_conflict"
                    )
        return self._to_career_snapshot(row)

    def get_career_snapshot(
        self,
        *,
        user_id: str,
        snapshot_revision: str,
    ) -> CareerSnapshotRecord:
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM career_snapshots
                WHERE user_id = %s AND snapshot_revision = %s
                """,
                (user_id, snapshot_revision),
            ).fetchone()
        if row is None:
            raise CareerSnapshotNotFoundError("career_snapshot_not_found")
        return self._to_career_snapshot(row)

    def has_career_snapshot(
        self,
        *,
        user_id: str,
        snapshot_revision: str,
    ) -> bool:
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM career_snapshots
                WHERE user_id = %s AND snapshot_revision = %s
                """,
                (user_id, snapshot_revision),
            ).fetchone()
        return row is not None

    def get_for_user(self, *, user_id: str, run_id: str) -> RunRecord:
        with self._pool.connection() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE user_id = %s AND run_id = %s",
                (user_id, run_id),
            ).fetchone()
        if row is None:
            raise RunNotFoundError("resource_not_found")
        return self._to_record(row)

    def list_events(
        self, *, user_id: str, run_id: str, after: int, limit: int
    ) -> EventPage:
        if after < 0 or limit < 1 or limit > 500:
            raise ValueError("invalid_event_page")
        with self._pool.connection() as connection:
            owned = connection.execute(
                "SELECT 1 FROM runs WHERE user_id = %s AND run_id = %s",
                (user_id, run_id),
            ).fetchone()
            if owned is None:
                raise RunNotFoundError("resource_not_found")
            rows = connection.execute(
                """
                SELECT sequence, event_type, payload_sanitized, created_at
                FROM run_events
                WHERE user_id = %s AND run_id = %s AND sequence > %s
                ORDER BY sequence
                LIMIT %s
                """,
                (user_id, run_id, after, limit),
            ).fetchall()
        items = [
            RunEventView(
                sequence=row["sequence"],
                event_type=row["event_type"],
                payload=row["payload_sanitized"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
        return EventPage(items=items, next_after=items[-1].sequence if items else after)

    def get_debug_view(self, *, user_id: str, run_id: str) -> RunDebugView:
        """Return one repeatable-read, metadata-only provenance projection."""

        with self._pool.connection() as connection, connection.transaction():
            connection.execute(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
            )
            run_row = connection.execute(
                "SELECT * FROM runs WHERE user_id = %s AND run_id = %s",
                (user_id, run_id),
            ).fetchone()
            if run_row is None:
                raise RunNotFoundError("resource_not_found")
            attempt_rows = connection.execute(
                """
                SELECT attempt_id, attempt_no, status, execution_phase,
                       provider_call_count, error_code, recovery_kind,
                       started_at, finished_at
                FROM run_attempts
                WHERE user_id = %s AND run_id = %s
                ORDER BY attempt_no
                """,
                (user_id, run_id),
            ).fetchall()
            approval_rows = connection.execute(
                """
                SELECT approval_id, request_id, action_digest,
                       pending_action_json -> 'pending_action' ->> 'tool_name'
                           AS tool_name,
                       decision, version, created_at, resolved_at
                FROM run_approvals
                WHERE user_id = %s AND run_id = %s
                ORDER BY created_at
                """,
                (user_id, run_id),
            ).fetchall()
            checkpoint_row = connection.execute(
                """
                SELECT checkpoint_kind, checkpoint_id, checkpoint_version,
                       updated_at, expires_at
                FROM run_checkpoints
                WHERE user_id = %s AND run_id = %s
                """,
                (user_id, run_id),
            ).fetchone()
            operation_rows = connection.execute(
                """
                SELECT operation_id, attempt_id, tool_name, action_digest,
                       state, error_code, started_at, finished_at, updated_at
                FROM tool_operations
                WHERE user_id = %s AND run_id = %s
                ORDER BY created_at
                """,
                (user_id, run_id),
            ).fetchall()
            event_rows = connection.execute(
                """
                SELECT sequence, attempt_id, event_type, payload_sanitized,
                       created_at, count(*) OVER () AS total_count
                FROM run_events
                WHERE user_id = %s AND run_id = %s
                ORDER BY sequence DESC
                LIMIT 500
                """,
                (user_id, run_id),
            ).fetchall()

        record = self._to_record(run_row)
        events = [
            DebugEventView(
                sequence=row["sequence"],
                attempt_id=(
                    str(row["attempt_id"])
                    if row["attempt_id"] is not None
                    else None
                ),
                event_type=row["event_type"],
                payload=row["payload_sanitized"],
                created_at=row["created_at"],
            )
            for row in reversed(event_rows)
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
            attempts=[
                DebugAttemptView(
                    attempt_id=str(row["attempt_id"]),
                    attempt_no=row["attempt_no"],
                    status=row["status"],
                    execution_phase=row["execution_phase"],
                    provider_call_count=row["provider_call_count"],
                    error_code=row["error_code"],
                    recovery_kind=row["recovery_kind"],
                    started_at=row["started_at"],
                    finished_at=row["finished_at"],
                )
                for row in attempt_rows
            ],
            approvals=[
                DebugApprovalView(
                    approval_id=str(row["approval_id"]),
                    request_id=row["request_id"],
                    action_digest=row["action_digest"],
                    tool_name=row["tool_name"],
                    decision=row["decision"],
                    version=row["version"],
                    created_at=row["created_at"],
                    resolved_at=row["resolved_at"],
                )
                for row in approval_rows
            ],
            checkpoint=(
                DebugCheckpointView(
                    checkpoint_kind=checkpoint_row["checkpoint_kind"],
                    checkpoint_id=checkpoint_row["checkpoint_id"],
                    checkpoint_version=checkpoint_row["checkpoint_version"],
                    updated_at=checkpoint_row["updated_at"],
                    expires_at=checkpoint_row["expires_at"],
                )
                if checkpoint_row is not None
                else None
            ),
            tool_operations=[
                DebugToolOperationView(
                    operation_id=row["operation_id"],
                    attempt_id=(
                        str(row["attempt_id"])
                        if row["attempt_id"] is not None
                        else None
                    ),
                    tool_name=row["tool_name"],
                    action_digest=row["action_digest"],
                    state=row["state"],
                    error_code=row["error_code"],
                    started_at=row["started_at"],
                    finished_at=row["finished_at"],
                    updated_at=row["updated_at"],
                )
                for row in operation_rows
            ],
            events=events,
            events_truncated=(
                bool(event_rows) and event_rows[0]["total_count"] > len(event_rows)
            ),
        )

    def claim(
        self,
        message: RunDispatchMessage,
        *,
        worker_id: str,
        lease_seconds: float,
    ) -> ClaimedRun | None:
        with self._pool.connection() as connection, connection.transaction():
            row = connection.execute(
                """
                SELECT *, next_attempt_at <= now() AS dispatch_ready
                FROM runs WHERE run_id = %s FOR UPDATE
                """,
                (message.run_id,),
            ).fetchone()
            if row is None:
                return None
            if row["user_id"] != message.user_id:
                raise QueueOwnershipError("queue_owner_mismatch")
            if (
                row["status"] != RunStatus.QUEUED.value
                or row["dispatch_generation"] != message.dispatch_generation
            ):
                return None
            if not row["dispatch_ready"]:
                raise RunDispatchDeferredError("next_attempt_at_not_reached")
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f'{row["user_id"]}\x1f{row["session_id"]}',),
            )
            session_busy = connection.execute(
                """
                SELECT 1 FROM runs
                WHERE user_id = %s AND session_id = %s AND status = 'RUNNING'
                  AND run_id <> %s
                LIMIT 1
                """,
                (row["user_id"], row["session_id"], row["run_id"]),
            ).fetchone()
            if session_busy is not None:
                generation = row["dispatch_generation"] + 1
                connection.execute(
                    """
                    UPDATE runs
                    SET status_version = status_version + 1,
                        dispatch_generation = %s,
                        next_attempt_at = clock_timestamp() + interval '0.25 seconds',
                        current_stage = 'session_wait',
                        updated_at = now()
                    WHERE user_id = %s AND run_id = %s AND status = 'QUEUED'
                    """,
                    (generation, row["user_id"], row["run_id"]),
                )
                self._append_event(
                    connection,
                    user_id=row["user_id"],
                    run_id=str(row["run_id"]),
                    event_type="run.session_deferred",
                    payload={"reason": "session_busy"},
                )
                self._insert_outbox(
                    connection,
                    user_id=row["user_id"],
                    run_id=str(row["run_id"]),
                    dispatch_generation=generation,
                )
                return None
            attempt_no = connection.execute(
                """
                SELECT COALESCE(MAX(attempt_no), 0) + 1 AS next_no
                FROM run_attempts WHERE user_id = %s AND run_id = %s
                """,
                (message.user_id, message.run_id),
            ).fetchone()["next_no"]
            attempt_id = str(uuid4())
            lease_token = str(uuid4())
            updated = connection.execute(
                """
                UPDATE runs
                SET status = 'RUNNING', status_version = status_version + 1,
                    current_stage = 'starting', updated_at = now()
                WHERE user_id = %s AND run_id = %s
                RETURNING *
                """,
                (message.user_id, message.run_id),
            ).fetchone()
            attempt = connection.execute(
                """
                INSERT INTO run_attempts(
                    attempt_id, user_id, run_id, attempt_no, message_id,
                    dispatch_generation, worker_id, lease_token,
                    lease_expires_at, heartbeat_at, status
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    clock_timestamp() + (%s * interval '1 second'), clock_timestamp(), 'ACTIVE'
                )
                RETURNING lease_expires_at
                """,
                (
                    attempt_id,
                    message.user_id,
                    message.run_id,
                    attempt_no,
                    message.message_id,
                    message.dispatch_generation,
                    worker_id,
                    lease_token,
                    lease_seconds,
                ),
            ).fetchone()
            self._append_event(
                connection,
                user_id=message.user_id,
                run_id=message.run_id,
                attempt_id=attempt_id,
                event_type="run.running",
                payload={"status": RunStatus.RUNNING.value, "attempt_no": attempt_no},
            )
            return ClaimedRun(
                run=self._to_record(updated),
                attempt_id=attempt_id,
                attempt_no=attempt_no,
                lease_token=lease_token,
                lease_expires_at=attempt["lease_expires_at"],
            )

    def mark_agent_active(self, claim: ClaimedRun) -> None:
        with self._pool.connection() as connection, connection.transaction():
            self._require_active_attempt(connection, claim)
            connection.execute(
                """
                UPDATE run_attempts SET execution_phase = 'AGENT_ACTIVE'
                WHERE user_id = %s AND attempt_id = %s AND lease_token = %s
                """,
                (claim.run.user_id, claim.attempt_id, claim.lease_token),
            )

    def heartbeat(self, claim: ClaimedRun, *, lease_seconds: float) -> bool:
        with self._pool.connection() as connection, connection.transaction():
            updated = connection.execute(
                """
                UPDATE run_attempts
                SET heartbeat_at = clock_timestamp(),
                    lease_expires_at = clock_timestamp() + (%s * interval '1 second')
                WHERE user_id = %s AND attempt_id = %s AND lease_token = %s
                  AND status = 'ACTIVE' AND lease_expires_at > clock_timestamp()
                RETURNING attempt_id
                """,
                (
                    lease_seconds,
                    claim.run.user_id,
                    claim.attempt_id,
                    claim.lease_token,
                ),
            ).fetchone()
        return updated is not None

    def record_progress(self, claim: ClaimedRun, *, stage: str) -> None:
        with self._pool.connection() as connection, connection.transaction():
            self._require_active_attempt(connection, claim)
            connection.execute(
                """
                UPDATE runs SET current_stage = %s, updated_at = now()
                WHERE user_id = %s AND run_id = %s AND status = 'RUNNING'
                """,
                (stage, claim.run.user_id, claim.run.run_id),
            )
            self._append_event(
                connection,
                user_id=claim.run.user_id,
                run_id=claim.run.run_id,
                attempt_id=claim.attempt_id,
                event_type="run.progress",
                payload={"stage": stage},
            )

    def succeed(
        self,
        claim: ClaimedRun,
        *,
        result: dict[str, object],
        provider_call_count: int,
    ) -> RunRecord:
        with self._pool.connection() as connection, connection.transaction():
            self._require_active_attempt(connection, claim)
            updated = connection.execute(
                """
                UPDATE runs
                SET status = 'SUCCEEDED', status_version = status_version + 1,
                    current_stage = 'completed', result_payload = %s,
                    updated_at = now(), finished_at = now()
                WHERE user_id = %s AND run_id = %s AND status = 'RUNNING'
                RETURNING *
                """,
                (Jsonb(result), claim.run.user_id, claim.run.run_id),
            ).fetchone()
            if updated is None:
                raise InvalidRunTransitionError("run_is_not_running")
            connection.execute(
                """
                UPDATE run_attempts
                SET status = 'SUCCEEDED', execution_phase = 'TERMINAL',
                    provider_call_count = %s, finished_at = now()
                WHERE user_id = %s AND attempt_id = %s AND lease_token = %s
                """,
                (
                    provider_call_count,
                    claim.run.user_id,
                    claim.attempt_id,
                    claim.lease_token,
                ),
            )
            self._append_event(
                connection,
                user_id=claim.run.user_id,
                run_id=claim.run.run_id,
                attempt_id=claim.attempt_id,
                event_type="run.succeeded",
                payload={"status": RunStatus.SUCCEEDED.value},
            )
            return self._to_record(updated)

    def fail(self, claim: ClaimedRun, *, error_code: str) -> RunRecord:
        with self._pool.connection() as connection, connection.transaction():
            self._require_active_attempt(connection, claim)
            updated = connection.execute(
                """
                UPDATE runs
                SET status = 'FAILED', status_version = status_version + 1,
                    current_stage = 'failed', error_code = %s,
                    updated_at = now(), finished_at = now()
                WHERE user_id = %s AND run_id = %s AND status = 'RUNNING'
                RETURNING *
                """,
                (error_code, claim.run.user_id, claim.run.run_id),
            ).fetchone()
            if updated is None:
                raise InvalidRunTransitionError("run_is_not_running")
            connection.execute(
                """
                UPDATE run_attempts
                SET status = 'FAILED', execution_phase = 'TERMINAL',
                    error_code = %s, finished_at = now()
                WHERE user_id = %s AND attempt_id = %s AND lease_token = %s
                """,
                (
                    error_code,
                    claim.run.user_id,
                    claim.attempt_id,
                    claim.lease_token,
                ),
            )
            self._append_event(
                connection,
                user_id=claim.run.user_id,
                run_id=claim.run.run_id,
                attempt_id=claim.attempt_id,
                event_type="run.failed",
                payload={"status": RunStatus.FAILED.value, "error_code": error_code},
            )
            return self._to_record(updated)

    def retry(
        self,
        claim: ClaimedRun,
        *,
        error_code: str,
        delay_seconds: float,
    ) -> RunRecord:
        with self._pool.connection() as connection, connection.transaction():
            self._require_active_attempt(connection, claim)
            generation = claim.run.dispatch_generation + 1
            updated = connection.execute(
                """
                UPDATE runs
                SET status = 'QUEUED', status_version = status_version + 1,
                    dispatch_generation = %s, retry_count = retry_count + 1,
                    next_attempt_at = now() + (%s * interval '1 second'),
                    current_stage = 'retry_scheduled', error_code = %s,
                    updated_at = now(), finished_at = NULL
                WHERE user_id = %s AND run_id = %s AND status = 'RUNNING'
                RETURNING *
                """,
                (
                    generation,
                    max(delay_seconds, 0.0),
                    error_code,
                    claim.run.user_id,
                    claim.run.run_id,
                ),
            ).fetchone()
            if updated is None:
                raise InvalidRunTransitionError("run_is_not_running")
            connection.execute(
                """
                UPDATE run_attempts
                SET status = 'FAILED', execution_phase = 'TERMINAL',
                    error_code = %s, recovery_kind = 'bounded_service_retry',
                    finished_at = now()
                WHERE user_id = %s AND attempt_id = %s AND lease_token = %s
                """,
                (error_code, claim.run.user_id, claim.attempt_id, claim.lease_token),
            )
            self._append_event(
                connection,
                user_id=claim.run.user_id,
                run_id=claim.run.run_id,
                attempt_id=claim.attempt_id,
                event_type="run.retry_scheduled",
                payload={"error_code": error_code, "retry_count": updated["retry_count"]},
            )
            self._insert_outbox(
                connection,
                user_id=claim.run.user_id,
                run_id=claim.run.run_id,
                dispatch_generation=generation,
            )
            return self._to_record(updated)

    def defer_for_backpressure(
        self,
        claim: ClaimedRun,
        *,
        error_code: str,
        delay_seconds: float,
    ) -> RunRecord:
        """Release a claimed Run without consuming its provider retry budget."""

        with self._pool.connection() as connection, connection.transaction():
            self._require_active_attempt(connection, claim)
            generation = claim.run.dispatch_generation + 1
            updated = connection.execute(
                """
                UPDATE runs
                SET status = 'QUEUED', status_version = status_version + 1,
                    dispatch_generation = %s,
                    next_attempt_at = now() + (%s * interval '1 second'),
                    current_stage = 'provider_backpressure', error_code = %s,
                    updated_at = now()
                WHERE user_id = %s AND run_id = %s AND status = 'RUNNING'
                RETURNING *
                """,
                (
                    generation,
                    max(delay_seconds, 0.05),
                    error_code,
                    claim.run.user_id,
                    claim.run.run_id,
                ),
            ).fetchone()
            if updated is None:
                raise InvalidRunTransitionError("run_is_not_running")
            connection.execute(
                """
                UPDATE run_attempts
                SET status = 'FAILED', execution_phase = 'TERMINAL',
                    error_code = %s, recovery_kind = 'provider_admission_deferred',
                    finished_at = now()
                WHERE user_id = %s AND attempt_id = %s AND lease_token = %s
                """,
                (error_code, claim.run.user_id, claim.attempt_id, claim.lease_token),
            )
            self._append_event(
                connection,
                user_id=claim.run.user_id,
                run_id=claim.run.run_id,
                attempt_id=claim.attempt_id,
                event_type="run.backpressured",
                payload={"error_code": error_code},
            )
            self._insert_outbox(
                connection,
                user_id=claim.run.user_id,
                run_id=claim.run.run_id,
                dispatch_generation=generation,
            )
            return self._to_record(updated)

    def mark_uncertain(self, claim: ClaimedRun, *, error_code: str) -> RunRecord:
        with self._pool.connection() as connection, connection.transaction():
            self._require_active_attempt(connection, claim)
            updated = connection.execute(
                """
                UPDATE runs
                SET status = 'UNCERTAIN', status_version = status_version + 1,
                    current_stage = 'reconciliation_required', error_code = %s,
                    updated_at = now(), finished_at = now()
                WHERE user_id = %s AND run_id = %s AND status = 'RUNNING'
                RETURNING *
                """,
                (error_code, claim.run.user_id, claim.run.run_id),
            ).fetchone()
            if updated is None:
                raise InvalidRunTransitionError("run_is_not_running")
            connection.execute(
                """
                UPDATE tool_operations
                SET state = 'UNCERTAIN', error_code = %s,
                    finished_at = clock_timestamp(), updated_at = clock_timestamp()
                WHERE user_id = %s AND run_id = %s AND state = 'INFLIGHT'
                """,
                (error_code, claim.run.user_id, claim.run.run_id),
            )
            connection.execute(
                """
                UPDATE run_attempts
                SET status = 'UNCERTAIN', execution_phase = 'TERMINAL',
                    error_code = %s, finished_at = now()
                WHERE user_id = %s AND attempt_id = %s AND lease_token = %s
                """,
                (error_code, claim.run.user_id, claim.attempt_id, claim.lease_token),
            )
            self._append_event(
                connection,
                user_id=claim.run.user_id,
                run_id=claim.run.run_id,
                attempt_id=claim.attempt_id,
                event_type="run.uncertain",
                payload={"status": RunStatus.UNCERTAIN.value, "error_code": error_code},
            )
            return self._to_record(updated)

    def wait_for_approval(
        self,
        claim: ClaimedRun,
        *,
        pause: AgentApprovalPause,
    ) -> tuple[RunRecord, str]:
        request = pause.approval_request
        action = pause.pending_action
        if (
            request.run_id != claim.run.run_id
            or request.session_id != claim.run.session_id
            or request.action_digest != compute_action_digest(action)
        ):
            raise ApprovalPersistenceError("approval_scope_or_digest_mismatch")
        approval_id = str(uuid4())
        envelope = {
            "approval_request": request.model_dump(mode="json"),
            "pending_action": action.model_dump(mode="json"),
        }
        with self._pool.connection() as connection, connection.transaction():
            self._require_active_attempt(connection, claim)
            checkpoint = connection.execute(
                """
                SELECT 1 FROM run_checkpoints
                WHERE user_id = %s AND run_id = %s
                  AND (expires_at IS NULL OR expires_at > now())
                """,
                (claim.run.user_id, claim.run.run_id),
            ).fetchone()
            if checkpoint is None:
                raise CheckpointError("approval_checkpoint_required")
            connection.execute(
                """
                INSERT INTO run_approvals(
                    approval_id, user_id, run_id, session_id, request_id,
                    action_digest, pending_action_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    approval_id,
                    claim.run.user_id,
                    claim.run.run_id,
                    request.session_id,
                    request.request_id,
                    request.action_digest,
                    Jsonb(envelope),
                ),
            )
            updated = connection.execute(
                """
                UPDATE runs
                SET status = 'WAITING_APPROVAL', status_version = status_version + 1,
                    current_stage = 'waiting_approval', updated_at = now()
                WHERE user_id = %s AND run_id = %s AND status = 'RUNNING'
                RETURNING *
                """,
                (claim.run.user_id, claim.run.run_id),
            ).fetchone()
            if updated is None:
                raise InvalidRunTransitionError("run_is_not_running")
            connection.execute(
                """
                UPDATE run_attempts
                SET status = 'SUCCEEDED', execution_phase = 'TERMINAL',
                    provider_call_count = %s, recovery_kind = 'approval_pause',
                    finished_at = now()
                WHERE user_id = %s AND attempt_id = %s AND lease_token = %s
                """,
                (
                    pause.provider_call_count,
                    claim.run.user_id,
                    claim.attempt_id,
                    claim.lease_token,
                ),
            )
            self._append_event(
                connection,
                user_id=claim.run.user_id,
                run_id=claim.run.run_id,
                attempt_id=claim.attempt_id,
                event_type="run.waiting_approval",
                payload={
                    "approval_id": approval_id,
                    "request_id": request.request_id,
                    "action_digest": request.action_digest,
                    "tool_name": request.tool_name,
                },
            )
            return self._to_record(updated), approval_id

    def recover_expired_attempts(
        self,
        *,
        limit: int = 100,
        max_attempts: int = 3,
    ) -> int:
        """Requeue expired model-only attempts; Tool uncertainty is never handled here."""

        if max_attempts < 1:
            raise ValueError("max_attempts_must_be_positive")
        recovered = 0
        with self._pool.connection() as connection, connection.transaction():
            attempts = connection.execute(
                """
                SELECT a.*, r.status AS run_status, r.dispatch_generation
                FROM run_attempts a
                JOIN runs r ON r.user_id = a.user_id AND r.run_id = a.run_id
                WHERE a.status = 'ACTIVE' AND a.lease_expires_at <= clock_timestamp()
                ORDER BY a.lease_expires_at
                FOR UPDATE OF a, r SKIP LOCKED
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
            for attempt in attempts:
                connection.execute(
                    """
                    UPDATE run_attempts
                    SET status = 'EXPIRED', recovery_kind = 'lease_expired_safe_replay',
                        finished_at = now()
                    WHERE attempt_id = %s
                    """,
                    (attempt["attempt_id"],),
                )
                if attempt["run_status"] != RunStatus.RUNNING.value:
                    continue
                if attempt["attempt_no"] >= max_attempts:
                    connection.execute(
                        """
                        UPDATE runs
                        SET status = 'FAILED', status_version = status_version + 1,
                            current_stage = 'failed', error_code = 'worker_recovery_exhausted',
                            updated_at = now(), finished_at = now()
                        WHERE user_id = %s AND run_id = %s AND status = 'RUNNING'
                        """,
                        (attempt["user_id"], attempt["run_id"]),
                    )
                    self._append_event(
                        connection,
                        user_id=attempt["user_id"],
                        run_id=str(attempt["run_id"]),
                        attempt_id=str(attempt["attempt_id"]),
                        event_type="run.failed",
                        payload={
                            "status": RunStatus.FAILED.value,
                            "error_code": "worker_recovery_exhausted",
                        },
                    )
                    continue
                generation = attempt["dispatch_generation"] + 1
                connection.execute(
                    """
                    UPDATE runs
                    SET status = 'QUEUED', status_version = status_version + 1,
                        dispatch_generation = %s, retry_count = retry_count + 1,
                        next_attempt_at = now(), current_stage = 'recovered', updated_at = now()
                    WHERE user_id = %s AND run_id = %s AND status = 'RUNNING'
                    """,
                    (generation, attempt["user_id"], attempt["run_id"]),
                )
                self._append_event(
                    connection,
                    user_id=attempt["user_id"],
                    run_id=str(attempt["run_id"]),
                    attempt_id=str(attempt["attempt_id"]),
                    event_type="run.recovered",
                    payload={"recovery_kind": "lease_expired_safe_replay"},
                )
                self._insert_outbox(
                    connection,
                    user_id=attempt["user_id"],
                    run_id=str(attempt["run_id"]),
                    dispatch_generation=generation,
                )
                recovered += 1
        return recovered

    def pending_dispatches(self, *, limit: int = 100) -> list[RunDispatchMessage]:
        with self._pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT o.message_id, o.user_id, o.aggregate_id, o.dispatch_generation,
                       o.created_at
                FROM outbox_messages o
                JOIN runs r ON r.user_id = o.user_id AND r.run_id = o.aggregate_id
                WHERE o.published_at IS NULL AND r.next_attempt_at <= now()
                ORDER BY o.created_at
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [
            RunDispatchMessage(
                message_id=str(row["message_id"]),
                user_id=row["user_id"],
                run_id=str(row["aggregate_id"]),
                dispatch_generation=row["dispatch_generation"],
                enqueued_at=row["created_at"],
            )
            for row in rows
        ]

    def reconcile_ready_queued_runs(
        self,
        *,
        limit: int = 100,
        grace_seconds: float = 30.0,
    ) -> int:
        """Create at most one recovery wake-up per stale Run and grace window."""

        if grace_seconds < 0:
            raise ValueError("grace_seconds_must_not_be_negative")

        created = 0
        with self._pool.connection() as connection, connection.transaction():
            rows = connection.execute(
                """
                SELECT r.user_id, r.run_id, r.dispatch_generation
                FROM runs r
                WHERE r.status = 'QUEUED' AND r.next_attempt_at <= now()
                  AND r.updated_at <=
                      clock_timestamp() - (%s * interval '1 second')
                  AND NOT EXISTS (
                    SELECT 1 FROM outbox_messages o
                    WHERE o.user_id = r.user_id AND o.aggregate_id = r.run_id
                      AND o.dispatch_generation = r.dispatch_generation
                      AND o.published_at IS NULL
                  )
                ORDER BY r.next_attempt_at, r.created_at
                FOR UPDATE OF r SKIP LOCKED
                LIMIT %s
                """,
                (grace_seconds, limit),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """
                    UPDATE runs SET updated_at = clock_timestamp()
                    WHERE user_id = %s AND run_id = %s
                      AND status = 'QUEUED' AND dispatch_generation = %s
                    """,
                    (
                        row["user_id"],
                        row["run_id"],
                        row["dispatch_generation"],
                    ),
                )
                self._insert_outbox(
                    connection,
                    user_id=row["user_id"],
                    run_id=str(row["run_id"]),
                    dispatch_generation=row["dispatch_generation"],
                )
                self._append_event(
                    connection,
                    user_id=row["user_id"],
                    run_id=str(row["run_id"]),
                    event_type="run.dispatch_reconciled",
                    payload={"dispatch_generation": row["dispatch_generation"]},
                )
                created += 1
        return created

    def save_pending_approval(
        self,
        *,
        user_id: str,
        run_id: str,
        request: ApprovalRequest,
        pending_action: AgentAction,
    ) -> str:
        """Persist the existing Runtime approval envelope without redefining its digest."""

        if request.run_id != run_id or request.session_id is None:
            raise ApprovalPersistenceError("approval_scope_mismatch")
        if request.action_digest != compute_action_digest(pending_action):
            raise ApprovalPersistenceError("approval_digest_mismatch")
        approval_id = str(uuid4())
        envelope = {
            "approval_request": request.model_dump(mode="json"),
            "pending_action": pending_action.model_dump(mode="json"),
        }
        with self._pool.connection() as connection, connection.transaction():
            run = connection.execute(
                """
                SELECT session_id FROM runs
                WHERE user_id = %s AND run_id = %s
                FOR UPDATE
                """,
                (user_id, run_id),
            ).fetchone()
            if run is None:
                raise RunNotFoundError("resource_not_found")
            if run["session_id"] != request.session_id:
                raise ApprovalPersistenceError("approval_scope_mismatch")
            inserted = connection.execute(
                """
                INSERT INTO run_approvals(
                    approval_id, user_id, run_id, session_id, request_id,
                    action_digest, pending_action_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, run_id, request_id) DO NOTHING
                RETURNING approval_id
                """,
                (
                    approval_id,
                    user_id,
                    run_id,
                    request.session_id,
                    request.request_id,
                    request.action_digest,
                    Jsonb(envelope),
                ),
            ).fetchone()
            if inserted is not None:
                return str(inserted["approval_id"])
            existing = connection.execute(
                """
                SELECT approval_id, action_digest, pending_action_json
                FROM run_approvals
                WHERE user_id = %s AND run_id = %s AND request_id = %s
                """,
                (user_id, run_id, request.request_id),
            ).fetchone()
            if (
                existing is None
                or existing["action_digest"] != request.action_digest
                or existing["pending_action_json"] != envelope
            ):
                raise ApprovalPersistenceError("approval_request_conflict")
            return str(existing["approval_id"])

    def resolve_approval(
        self,
        *,
        user_id: str,
        run_id: str,
        decision: ApprovalDecision,
        expected_version: int = 0,
    ) -> None:
        if decision.run_id != run_id or decision.session_id is None:
            raise ApprovalPersistenceError("approval_scope_mismatch")
        value = "approved" if decision.approved else "rejected"
        with self._pool.connection() as connection, connection.transaction():
            approval = connection.execute(
                """
                SELECT * FROM run_approvals
                WHERE user_id = %s AND run_id = %s AND request_id = %s
                FOR UPDATE
                """,
                (user_id, run_id, decision.request_id),
            ).fetchone()
            if approval is None:
                raise RunNotFoundError("resource_not_found")
            if (
                approval["session_id"] != decision.session_id
                or approval["action_digest"] != decision.action_digest
            ):
                raise ApprovalPersistenceError("approval_scope_or_digest_mismatch")
            if approval["decision"] is not None:
                if approval["decision"] == value and approval["reason"] == decision.reason:
                    return
                raise ApprovalPersistenceError("approval_already_resolved")
            updated = connection.execute(
                """
                UPDATE run_approvals
                SET decision = %s, reason = %s, version = version + 1,
                    resolved_by_user_id = %s, resolved_at = now()
                WHERE user_id = %s AND approval_id = %s AND version = %s
                  AND decision IS NULL
                RETURNING approval_id
                """,
                (
                    value,
                    decision.reason,
                    user_id,
                    user_id,
                    approval["approval_id"],
                    expected_version,
                ),
            ).fetchone()
            if updated is None:
                raise ApprovalPersistenceError("approval_version_conflict")

    def load_resolved_approval_decisions(
        self,
        *,
        user_id: str,
        run_id: str,
    ) -> dict[str, ApprovalDecision]:
        """Return Runtime approval contracts keyed by their action digest."""
        with self._pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT request_id, action_digest, decision, reason, session_id
                FROM run_approvals
                WHERE user_id = %s AND run_id = %s AND decision IS NOT NULL
                ORDER BY resolved_at, created_at
                """,
                (user_id, run_id),
            ).fetchall()
        decisions: dict[str, ApprovalDecision] = {}
        for row in rows:
            decision = ApprovalDecision(
                request_id=row["request_id"],
                action_digest=row["action_digest"],
                approved=row["decision"] == "approved",
                reason=row["reason"] or "",
                session_id=row["session_id"],
                run_id=run_id,
            )
            existing = decisions.get(decision.action_digest)
            if existing is not None and existing != decision:
                raise ApprovalPersistenceError("approval_digest_decision_conflict")
            decisions[decision.action_digest] = decision
        return decisions

    def decide_approval_and_requeue(
        self,
        *,
        user_id: str,
        run_id: str,
        request: ApprovalDecisionRequest,
        approved: bool,
    ) -> RunRecord:
        value = "approved" if approved else "rejected"
        with self._pool.connection() as connection, connection.transaction():
            run = connection.execute(
                "SELECT * FROM runs WHERE user_id = %s AND run_id = %s FOR UPDATE",
                (user_id, run_id),
            ).fetchone()
            if run is None:
                raise RunNotFoundError("resource_not_found")
            approval = connection.execute(
                """
                SELECT * FROM run_approvals
                WHERE user_id = %s AND run_id = %s AND approval_id = %s
                FOR UPDATE
                """,
                (user_id, run_id, request.approval_id),
            ).fetchone()
            if approval is None:
                raise RunNotFoundError("resource_not_found")
            if (
                approval["session_id"] != run["session_id"]
                or approval["action_digest"] != request.action_digest
            ):
                raise ApprovalPersistenceError("approval_scope_or_digest_mismatch")
            # Construct the existing Runtime decision contract before persistence.
            ApprovalDecision(
                request_id=approval["request_id"],
                action_digest=request.action_digest,
                approved=approved,
                reason=request.reason,
                session_id=approval["session_id"],
                run_id=str(run["run_id"]),
            )
            if approval["decision"] is not None:
                if approval["decision"] == value and approval["reason"] == request.reason:
                    return self._to_record(run)
                raise ApprovalPersistenceError("approval_already_resolved")
            if run["status"] != RunStatus.WAITING_APPROVAL.value:
                raise RunStateConflictError("run_is_not_waiting_approval")
            if run["status_version"] != request.expected_status_version:
                raise RunStateConflictError("status_version_conflict")
            connection.execute(
                """
                UPDATE run_approvals
                SET decision = %s, reason = %s, version = version + 1,
                    resolved_by_user_id = %s, resolved_at = now()
                WHERE user_id = %s AND approval_id = %s AND decision IS NULL
                """,
                (value, request.reason, user_id, user_id, request.approval_id),
            )
            generation = run["dispatch_generation"] + 1
            updated = connection.execute(
                """
                UPDATE runs
                SET status = 'QUEUED', status_version = status_version + 1,
                    dispatch_generation = %s, next_attempt_at = now(),
                    current_stage = 'approval_resolved', error_code = NULL,
                    updated_at = now(), finished_at = NULL
                WHERE user_id = %s AND run_id = %s
                RETURNING *
                """,
                (generation, user_id, run_id),
            ).fetchone()
            self._append_event(
                connection,
                user_id=user_id,
                run_id=run_id,
                actor_user_id=user_id,
                event_type="run.approval_resolved",
                payload={
                    "approval_id": request.approval_id,
                    "decision": value,
                    "action_digest": request.action_digest,
                },
            )
            self._insert_outbox(
                connection,
                user_id=user_id,
                run_id=run_id,
                dispatch_generation=generation,
            )
            return self._to_record(updated)

    def resume_failed(
        self,
        *,
        user_id: str,
        run_id: str,
        expected_status_version: int,
        actor_user_id: str,
        resumable_error_codes: frozenset[str],
    ) -> RunRecord:
        with self._pool.connection() as connection, connection.transaction():
            run = connection.execute(
                "SELECT * FROM runs WHERE user_id = %s AND run_id = %s FOR UPDATE",
                (user_id, run_id),
            ).fetchone()
            if run is None:
                raise RunNotFoundError("resource_not_found")
            if run["status"] == RunStatus.UNCERTAIN.value:
                raise RunStateConflictError("uncertain_requires_reconciliation")
            if run["status"] != RunStatus.FAILED.value:
                raise RunStateConflictError("run_is_not_resumable")
            if run["error_code"] not in resumable_error_codes:
                raise RunStateConflictError("failure_is_not_safely_resumable")
            if run["status_version"] != expected_status_version:
                raise RunStateConflictError("status_version_conflict")
            generation = run["dispatch_generation"] + 1
            updated = connection.execute(
                """
                UPDATE runs
                SET status = 'QUEUED', status_version = status_version + 1,
                    dispatch_generation = %s, next_attempt_at = now(),
                    current_stage = 'explicit_resume', error_code = NULL,
                    updated_at = now(), finished_at = NULL
                WHERE user_id = %s AND run_id = %s
                RETURNING *
                """,
                (generation, user_id, run_id),
            ).fetchone()
            self._append_event(
                connection,
                user_id=user_id,
                run_id=run_id,
                actor_user_id=actor_user_id,
                event_type="run.resumed",
                payload={"status": RunStatus.QUEUED.value},
            )
            self._insert_outbox(
                connection,
                user_id=user_id,
                run_id=run_id,
                dispatch_generation=generation,
            )
            return self._to_record(updated)

    def save_private_checkpoint(
        self,
        *,
        claim: ClaimedRun,
        checkpoint_kind: str,
        checkpoint_id: str,
        payload: dict[str, object],
        expires_at=None,
    ) -> int:
        with self._pool.connection() as connection, connection.transaction():
            self._require_active_attempt(connection, claim)
            row = connection.execute(
                """
                INSERT INTO run_checkpoints(
                    user_id, run_id, checkpoint_kind, checkpoint_id,
                    checkpoint_version, payload_private, expires_at
                )
                VALUES (%s, %s, %s, %s, 1, %s, %s)
                ON CONFLICT (user_id, run_id) DO UPDATE
                SET checkpoint_kind = EXCLUDED.checkpoint_kind,
                    checkpoint_id = EXCLUDED.checkpoint_id,
                    checkpoint_version = run_checkpoints.checkpoint_version + 1,
                    payload_private = EXCLUDED.payload_private,
                    updated_at = now(), expires_at = EXCLUDED.expires_at
                RETURNING checkpoint_version
                """,
                (
                    claim.run.user_id,
                    claim.run.run_id,
                    checkpoint_kind,
                    checkpoint_id,
                    Jsonb(payload),
                    expires_at,
                ),
            ).fetchone()
            return int(row["checkpoint_version"])

    def begin_tool_operation(
        self,
        claim: ClaimedRun,
        *,
        operation_id: str,
        tool_name: str,
        action_digest: str,
        request_hash: str,
    ) -> ToolOperationLease:
        with self._pool.connection() as connection, connection.transaction():
            self._require_active_attempt(connection, claim)
            connection.execute(
                """
                INSERT INTO tool_operations(
                    operation_id, user_id, run_id, tool_name,
                    action_digest, request_hash, state
                )
                VALUES (%s, %s, %s, %s, %s, %s, 'PREPARED')
                ON CONFLICT (operation_id) DO NOTHING
                """,
                (
                    operation_id,
                    claim.run.user_id,
                    claim.run.run_id,
                    tool_name,
                    action_digest,
                    request_hash,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM tool_operations
                WHERE operation_id = %s
                FOR UPDATE
                """,
                (operation_id,),
            ).fetchone()
            if row is None or (
                row["user_id"] != claim.run.user_id
                or str(row["run_id"]) != claim.run.run_id
                or row["tool_name"] != tool_name
                or row["action_digest"] != action_digest
                or row["request_hash"] != request_hash
            ):
                raise ToolOperationConflictError("tool_operation_identity_conflict")

            state = ToolOperationState(row["state"])
            should_execute = False
            if state == ToolOperationState.PREPARED:
                row = connection.execute(
                    """
                    UPDATE tool_operations
                    SET state = 'INFLIGHT', attempt_id = %s,
                        started_at = clock_timestamp(), updated_at = clock_timestamp()
                    WHERE operation_id = %s AND state = 'PREPARED'
                    RETURNING *
                    """,
                    (claim.attempt_id, operation_id),
                ).fetchone()
                should_execute = True
            elif state == ToolOperationState.INFLIGHT:
                row = connection.execute(
                    """
                    UPDATE tool_operations
                    SET state = 'UNCERTAIN', error_code = %s,
                        finished_at = clock_timestamp(), updated_at = clock_timestamp()
                    WHERE operation_id = %s AND state = 'INFLIGHT'
                    RETURNING *
                    """,
                    ("non_idempotent_execution_uncertain", operation_id),
                ).fetchone()
            return ToolOperationLease(
                operation=self._to_tool_operation(row),
                should_execute=should_execute,
            )

    def complete_tool_operation(
        self,
        claim: ClaimedRun,
        *,
        operation_id: str,
        state: ToolOperationState,
        receipt: dict[str, object],
        error_code: str | None,
    ) -> ToolOperationRecord:
        if state not in {
            ToolOperationState.SUCCEEDED,
            ToolOperationState.FAILED,
        }:
            raise ValueError("tool_operation_terminal_state_required")
        with self._pool.connection() as connection, connection.transaction():
            self._require_active_attempt(connection, claim)
            row = connection.execute(
                """
                UPDATE tool_operations
                SET state = %s, receipt_private = %s, error_code = %s,
                    finished_at = clock_timestamp(), updated_at = clock_timestamp()
                WHERE operation_id = %s AND user_id = %s AND run_id = %s
                  AND attempt_id = %s AND state = 'INFLIGHT'
                RETURNING *
                """,
                (
                    state.value,
                    Jsonb(receipt),
                    error_code,
                    operation_id,
                    claim.run.user_id,
                    claim.run.run_id,
                    claim.attempt_id,
                ),
            ).fetchone()
            if row is None:
                raise StaleAttemptError("tool_operation_attempt_stale")
            return self._to_tool_operation(row)

    def get_tool_operation(
        self,
        *,
        user_id: str,
        run_id: str,
        operation_id: str,
    ) -> ToolOperationRecord:
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM tool_operations
                WHERE user_id = %s AND run_id = %s AND operation_id = %s
                """,
                (user_id, run_id, operation_id),
            ).fetchone()
        if row is None:
            raise RunNotFoundError("resource_not_found")
        return self._to_tool_operation(row)

    def load_private_checkpoint(
        self,
        *,
        user_id: str,
        run_id: str,
        checkpoint_kind: str,
    ) -> dict[str, object]:
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                SELECT payload_private FROM run_checkpoints
                WHERE user_id = %s AND run_id = %s AND checkpoint_kind = %s
                  AND (expires_at IS NULL OR expires_at > now())
                """,
                (user_id, run_id, checkpoint_kind),
            ).fetchone()
        if row is None:
            raise CheckpointError("checkpoint_not_found")
        return row["payload_private"]

    def has_private_checkpoint(
        self, *, user_id: str, run_id: str, checkpoint_kind: str
    ) -> bool:
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM run_checkpoints
                WHERE user_id = %s AND run_id = %s AND checkpoint_kind = %s
                  AND (expires_at IS NULL OR expires_at > now())
                """,
                (user_id, run_id, checkpoint_kind),
            ).fetchone()
        return row is not None

    def delete_private_checkpoint(
        self, *, claim: ClaimedRun, checkpoint_kind: str
    ) -> None:
        with self._pool.connection() as connection, connection.transaction():
            self._require_active_attempt(connection, claim)
            connection.execute(
                """
                DELETE FROM run_checkpoints
                WHERE user_id = %s AND run_id = %s AND checkpoint_kind = %s
                """,
                (claim.run.user_id, claim.run.run_id, checkpoint_kind),
            )

    def observability_snapshot(
        self,
        buckets_seconds: tuple[float, ...],
    ) -> dict[str, object]:
        """Aggregate public operational metrics without exposing user-owned payloads."""
        with self._pool.connection() as connection:
            status_rows = connection.execute(
                "SELECT status, count(*) AS count FROM runs GROUP BY status"
            ).fetchall()
            totals = connection.execute(
                """
                SELECT COALESCE(sum(retry_count), 0) AS retry_count,
                       (SELECT count(*) FROM run_events
                        WHERE event_type IN ('run.recovered', 'run.dispatch_reconciled'))
                           AS recovery_count,
                       (SELECT count(*) FROM run_approvals WHERE decision IS NULL)
                           AS approvals_waiting
                FROM runs
                """
            ).fetchone()
            run_latency = self._query_duration_histogram(
                connection,
                """
                WITH durations AS MATERIALIZED (
                    SELECT EXTRACT(EPOCH FROM (finished_at - created_at))::double precision
                               AS duration
                    FROM runs
                    WHERE finished_at IS NOT NULL AND finished_at >= created_at
                )
                SELECT bucket,
                       (SELECT count(*) FROM durations WHERE duration <= bucket) AS cumulative,
                       (SELECT count(*) FROM durations) AS total,
                       (SELECT COALESCE(sum(duration), 0) FROM durations) AS total_seconds
                FROM unnest(%s::double precision[]) AS bucket
                ORDER BY bucket
                """,
                buckets_seconds,
            )
            approval_wait_latency = self._query_duration_histogram(
                connection,
                """
                WITH durations AS MATERIALIZED (
                    SELECT EXTRACT(EPOCH FROM (
                               COALESCE(resolved_at, clock_timestamp()) - created_at
                           ))::double precision AS duration
                    FROM run_approvals
                    WHERE COALESCE(resolved_at, clock_timestamp()) >= created_at
                )
                SELECT bucket,
                       (SELECT count(*) FROM durations WHERE duration <= bucket) AS cumulative,
                       (SELECT count(*) FROM durations) AS total,
                       (SELECT COALESCE(sum(duration), 0) FROM durations) AS total_seconds
                FROM unnest(%s::double precision[]) AS bucket
                ORDER BY bucket
                """,
                buckets_seconds,
            )
            admission_deferral_rows = connection.execute(
                """
                SELECT error_code, count(*) AS count
                FROM run_attempts
                WHERE recovery_kind = 'provider_admission_deferred'
                GROUP BY error_code
                """
            ).fetchall()
        return {
            "run_status_counts": {
                row["status"]: int(row["count"]) for row in status_rows
            },
            "retry_count": int(totals["retry_count"]),
            "recovery_count": int(totals["recovery_count"]),
            "approvals_waiting": int(totals["approvals_waiting"]),
            "provider_admission_deferrals": {
                (row["error_code"] or "other"): int(row["count"])
                for row in admission_deferral_rows
            },
            "run_latency": run_latency,
            "approval_wait_latency": approval_wait_latency,
        }

    @staticmethod
    def _query_duration_histogram(
        connection,
        query: str,
        buckets_seconds: tuple[float, ...],
    ) -> dict[str, object]:
        rows = connection.execute(query, (list(buckets_seconds),)).fetchall()
        total = int(rows[0]["total"]) if rows else 0
        return {
            "buckets": {
                f'{float(row["bucket"]):g}': int(row["cumulative"])
                for row in rows
            }
            | {"+Inf": total},
            "count": total,
            "sum": float(rows[0]["total_seconds"]) if rows else 0.0,
        }

    def mark_dispatch_published(self, *, message_id: str) -> None:
        with self._pool.connection() as connection, connection.transaction():
            connection.execute(
                """
                UPDATE outbox_messages
                SET published_at = COALESCE(published_at, now()),
                    publish_attempts = publish_attempts + 1,
                    last_error_code = NULL
                WHERE message_id = %s
                """,
                (message_id,),
            )

    def mark_dispatch_failed(self, *, message_id: str, error_code: str) -> None:
        with self._pool.connection() as connection, connection.transaction():
            connection.execute(
                """
                UPDATE outbox_messages
                SET publish_attempts = publish_attempts + 1, last_error_code = %s
                WHERE message_id = %s AND published_at IS NULL
                """,
                (error_code, message_id),
            )

    def _require_active_attempt(self, connection, claim: ClaimedRun) -> None:
        row = connection.execute(
            """
            SELECT 1 FROM run_attempts
            WHERE user_id = %s AND run_id = %s AND attempt_id = %s
              AND lease_token = %s AND status = 'ACTIVE'
              AND lease_expires_at > clock_timestamp()
            FOR UPDATE
            """,
            (
                claim.run.user_id,
                claim.run.run_id,
                claim.attempt_id,
                claim.lease_token,
            ),
        ).fetchone()
        if row is None:
            raise StaleAttemptError("stale_or_expired_attempt")

    @staticmethod
    def _append_event(
        connection,
        *,
        user_id: str,
        run_id: str,
        event_type: str,
        payload: dict[str, object],
        attempt_id: str | None = None,
        actor_user_id: str | None = None,
    ) -> None:
        sequence = connection.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence
            FROM run_events WHERE user_id = %s AND run_id = %s
            """,
            (user_id, run_id),
        ).fetchone()["next_sequence"]
        connection.execute(
            """
            INSERT INTO run_events(
                event_id, user_id, run_id, attempt_id, actor_user_id, sequence,
                event_type, payload_sanitized
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(uuid4()),
                user_id,
                run_id,
                attempt_id,
                actor_user_id,
                sequence,
                event_type,
                Jsonb(payload),
            ),
        )

    @staticmethod
    def _insert_outbox(
        connection, *, user_id: str, run_id: str, dispatch_generation: int
    ) -> None:
        message_id = str(uuid4())
        connection.execute(
            """
            INSERT INTO outbox_messages(
                message_id, user_id, aggregate_id, message_type,
                dispatch_generation, payload
            )
            VALUES (%s, %s, %s, 'run.dispatch', %s, %s)
            """,
            (
                message_id,
                user_id,
                run_id,
                dispatch_generation,
                Jsonb(
                    {
                        "user_id": user_id,
                        "run_id": run_id,
                        "dispatch_generation": dispatch_generation,
                    }
                ),
            ),
        )

    @staticmethod
    def _to_record(row) -> RunRecord:
        return RunRecord(
            run_id=str(row["run_id"]),
            user_id=row["user_id"],
            session_id=row["session_id"],
            task_type=row["task_type"],
            status=RunStatus(row["status"]),
            status_version=row["status_version"],
            dispatch_generation=row["dispatch_generation"],
            created_request_id=row["created_request_id"],
            idempotency_key=row["idempotency_key"],
            request_hash=row["request_hash"],
            request=CreateRunRequest.model_validate(row["request_payload"]),
            execution_manifest=(
                ExecutionManifest.model_validate(row["execution_manifest"])
                if row["execution_manifest"] is not None
                else None
            ),
            current_stage=row["current_stage"],
            result=row["result_payload"],
            error_code=row["error_code"],
            retry_count=row["retry_count"],
            next_attempt_at=row["next_attempt_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            finished_at=row["finished_at"],
        )

    @staticmethod
    def _to_career_snapshot(row) -> CareerSnapshotRecord:
        return CareerSnapshotRecord(
            user_id=row["user_id"],
            snapshot_revision=row["snapshot_revision"],
            content_hash=row["content_hash"],
            payload=CareerSnapshotPayload.model_validate(row["payload_private"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _to_tool_operation(row) -> ToolOperationRecord:
        return ToolOperationRecord(
            operation_id=row["operation_id"],
            user_id=row["user_id"],
            run_id=str(row["run_id"]),
            attempt_id=(str(row["attempt_id"]) if row["attempt_id"] else None),
            tool_name=row["tool_name"],
            action_digest=row["action_digest"],
            request_hash=row["request_hash"],
            state=ToolOperationState(row["state"]),
            receipt=row["receipt_private"],
            error_code=row["error_code"],
        )
