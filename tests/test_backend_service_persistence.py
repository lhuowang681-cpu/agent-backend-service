from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from redis import Redis
from redis.exceptions import RedisError
from psycopg import sql

from job_agent.agent_runtime.contracts import (
    AgentAction,
    AuthorizationResult,
    ApprovalDecision,
    ApprovalRequest,
    ToolEffect,
    ToolContext,
)
from job_agent.agent_runtime.policy import compute_action_digest
from job_agent.backend_service.contracts import (
    AgentExecutionOutcome,
    CreateRunRequest,
    RequestContext,
    RunStatus,
)
from job_agent.backend_service.checkpoint_adapter import PostgresSemanticCheckpointStore
from job_agent.backend_service.career_snapshot import (
    CareerSnapshotConflictError,
    CareerSnapshotNotFoundError,
    VersionedCareerSnapshotStore,
)
from job_agent.backend_service.execution_manifest import build_execution_manifest
from job_agent.backend_service.api import ConfiguredHeaderIdentityAdapter, create_app
from job_agent.backend_service.metrics import (
    PrometheusMetricsService,
    RedisMetricsRecorder,
)
from job_agent.backend_service.migration import apply_migrations
from job_agent.backend_service.outbox import OutboxPublisher
from job_agent.backend_service.postgres_repository import (
    ApprovalPersistenceError,
    PostgresRunRepository,
    StaleAttemptError,
)
from job_agent.backend_service.redis_streams import RedisStreamDispatchQueue
from job_agent.backend_service.reliable_worker import ReliableWorker
from job_agent.backend_service.repository import IdempotencyConflictError
from job_agent.backend_service.service import RunService
from job_agent.backend_service.tool_operations import (
    LedgeredToolExecutor,
    ToolOperationOutcomeUncertainError,
    ToolOperationState,
    compute_tool_operation_id,
)
from job_agent.agent_runtime.tool_registry import ToolRegistry
from job_agent.runtime.semantic_checkpoint import SemanticGraphCheckpoint
from job_agent.schemas import StrictModel


ADMIN_DATABASE_URL = os.environ.get(
    "JOB_AGENT_TEST_ADMIN_DATABASE_URL",
    "postgresql://job_agent:job_agent_dev@127.0.0.1:54329/job_agent",
)
DATABASE_URL = os.environ.get(
    "JOB_AGENT_TEST_DATABASE_URL",
    "postgresql://job_agent:job_agent_dev@127.0.0.1:54329/job_agent_test",
)
REDIS_URL = os.environ.get("JOB_AGENT_TEST_REDIS_URL", "redis://127.0.0.1:63799/15")
USER_A = "persistent-user-a"
USER_B = "persistent-user-b"


def _payload(
    *,
    title: str = "LLM Intern",
    session_id: str = "persistent-session",
) -> dict[str, object]:
    return {
        "task_type": "semantic_job_flow",
        "session_id": session_id,
        "input": {
            "selected_job": {
                "job_id": "job-001",
                "company": "Example",
                "title": title,
                "desc": "queue reliability",
                "url": "https://example.invalid/job",
                "location": "Beijing",
            },
            "resume_ref": "artifact://demo/resume",
        },
        "provider_profile": "mock",
        "budget_profile": "quick",
    }


@pytest.fixture
def repository():
    try:
        with psycopg.connect(ADMIN_DATABASE_URL, autocommit=True) as connection:
            exists = connection.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                ("job_agent_test",),
            ).fetchone()
            if exists is None:
                connection.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier("job_agent_test"))
                )
        apply_migrations(DATABASE_URL)
    except psycopg.OperationalError:
        pytest.skip("Phase 3 PostgreSQL is not running")
    repo = PostgresRunRepository(DATABASE_URL)
    repo.open()
    repo.ensure_users((USER_A,))
    with psycopg.connect(DATABASE_URL) as connection:
        connection.execute(
            """
            TRUNCATE tool_operations, outbox_messages, run_events, run_checkpoints,
                     run_approvals, run_attempts, runs, career_snapshots CASCADE
            """
        )
    yield repo
    repo.close()


@pytest.fixture
def stream_queue():
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        client.ping()
    except RedisError as exc:
        pytest.skip(f"Phase 3 Redis is not running: {type(exc).__name__}")
    stream_name = "job-agent:test-runs"
    client.delete(stream_name)
    queue = RedisStreamDispatchQueue(
        REDIS_URL,
        stream_name=stream_name,
        group_name="test-workers",
    )
    queue.ensure_group()
    yield queue
    queue.close()
    client.delete(stream_name)
    client.close()


def _create(
    repository: PostgresRunRepository,
    *,
    key: str = "persistent-key",
    session_id: str = "persistent-session",
):
    service = RunService(repository=repository, dispatch_queue=None)
    return service.create_run(
        RequestContext(user_id=USER_A, request_id="request-001"),
        CreateRunRequest.model_validate(_payload(session_id=session_id)),
        idempotency_key=key,
    )


def _application_payload(snapshot_revision: str) -> dict[str, object]:
    payload = _payload()
    payload["task_type"] = "application_assistant_flow"
    payload["input"] = {
        "selected_job": payload["input"]["selected_job"],
        "career_snapshot_revision": snapshot_revision,
        "draft_channel": "email",
        "draft_target": "recruiting@example.invalid",
    }
    return payload


def test_versioned_career_snapshot_is_user_scoped_and_immutable(repository):
    repository.ensure_users((USER_B,))
    snapshots = VersionedCareerSnapshotStore(repository)
    first = snapshots.put(
        user_id=USER_A,
        source_ref="artifact://resume/v1",
        resume_text="Agent runtime and evaluation evidence.",
        evidence_refs=["project:agent-runtime"],
    )
    duplicate = snapshots.put(
        user_id=USER_A,
        source_ref="artifact://resume/v1",
        resume_text="Agent runtime and evaluation evidence.",
        evidence_refs=["project:agent-runtime"],
    )
    changed = snapshots.put(
        user_id=USER_A,
        source_ref="artifact://resume/v2",
        resume_text="Agent runtime, approval, and recovery evidence.",
    )

    assert duplicate.snapshot_revision == first.snapshot_revision
    assert changed.snapshot_revision != first.snapshot_revision
    assert snapshots.resolve(
        user_id=USER_A,
        snapshot_revision=first.snapshot_revision,
    ) == first
    with pytest.raises(CareerSnapshotNotFoundError, match="career_snapshot_not_found"):
        snapshots.resolve(
            user_id=USER_B,
            snapshot_revision=first.snapshot_revision,
        )
    with pytest.raises(
        CareerSnapshotConflictError,
        match="career_snapshot_revision_conflict",
    ):
        repository.put_career_snapshot(
            user_id=USER_A,
            snapshot_revision=first.snapshot_revision,
            content_hash="f" * 64,
            payload={
                "source_ref": "artifact://resume/conflict",
                "resume_text": "conflicting payload",
                "evidence_refs": [],
            },
        )


def test_application_run_freezes_snapshot_revision_and_execution_manifest(repository):
    snapshots = VersionedCareerSnapshotStore(repository)
    bound = snapshots.put(
        user_id=USER_A,
        source_ref="artifact://resume/bound",
        resume_text="Bound revision.",
    )
    request = CreateRunRequest.model_validate(
        _application_payload(bound.snapshot_revision)
    )
    service = RunService(
        repository=repository,
        dispatch_queue=None,
        career_snapshot_store=snapshots,
        execution_manifest_factory=lambda value: build_execution_manifest(
            value,
            provider_models={
                "mock": "fixture-model",
                "deepseek_flash": "deepseek-chat",
            },
        ),
    )
    created = service.create_run(
        RequestContext(user_id=USER_A, request_id="snapshot-bound-request"),
        request,
        idempotency_key="snapshot-bound-key",
    )
    snapshots.put(
        user_id=USER_A,
        source_ref="artifact://resume/newer",
        resume_text="Newer revision must not change the existing Run.",
    )

    run = repository.get_for_user(user_id=USER_A, run_id=created.run_id)
    assert run.request.input.career_snapshot_revision == bound.snapshot_revision
    assert run.execution_manifest is not None
    assert run.execution_manifest.career_snapshot_revision == bound.snapshot_revision
    assert run.execution_manifest.model == "fixture-model"


class _LedgerInput(StrictModel):
    value: str


class _LedgerOutput(StrictModel):
    receipt: str


def _ledger_contract(repository, claim, calls, *, failpoint=None):
    registry = ToolRegistry()

    def mutate(value, _context):
        calls.append(value.value)
        return _LedgerOutput(receipt=f"receipt-{len(calls)}")

    registry.register(
        name="sandbox.non_idempotent_write",
        description="Controlled non-idempotent sandbox write.",
        input_model=_LedgerInput,
        output_model=_LedgerOutput,
        handler=mutate,
        effect=ToolEffect.LOCAL_STATE_MUTATION,
        idempotent=False,
    )
    action = AgentAction(
        action_id="ledger-action",
        tool_name="sandbox.non_idempotent_write",
        tool_arguments={"value": "one"},
        expected_observation="sandbox receipt",
        progress_claim="write sandbox draft",
    )
    digest = compute_action_digest(action)
    authorization = AuthorizationResult(
        action_id=action.action_id,
        action_digest=digest,
        tool_name=action.tool_name,
        effect=ToolEffect.LOCAL_STATE_MUTATION,
        allowed=True,
        requires_approval=False,
        reason_code="approved",
    )
    executor = LedgeredToolExecutor(
        registry,
        repository=repository,
        claim=claim,
        failpoint=failpoint,
    )
    context = ToolContext(
        session_id=claim.run.session_id,
        run_id=claim.run.run_id,
        agent_id="ledger-agent",
        workspace_root=".",
        sandbox_root=".",
    )
    return executor, action, authorization, context


def test_tool_operation_success_receipt_prevents_duplicate_side_effect(repository):
    created = _create(repository, key="ledger-success")
    message = repository.pending_dispatches()[0]
    repository.mark_dispatch_published(message_id=message.message_id)
    claim = repository.claim(message, worker_id="ledger-worker", lease_seconds=5)
    assert claim is not None
    calls: list[str] = []
    executor, action, authorization, context = _ledger_contract(
        repository,
        claim,
        calls,
    )

    first = executor.execute(action, authorization, context)
    replay = executor.execute(action, authorization, context)

    assert replay == first
    assert calls == ["one"]
    operation = repository.get_tool_operation(
        user_id=USER_A,
        run_id=created.run_id,
        operation_id=compute_tool_operation_id(
            run_id=created.run_id,
            action=action,
        ),
    )
    assert operation.state == ToolOperationState.SUCCEEDED
    assert operation.receipt == first.model_dump(mode="json")


class _ToolCrash(BaseException):
    pass


def test_tool_inflight_without_receipt_becomes_uncertain_without_replay(repository):
    created = _create(repository, key="ledger-uncertain")
    message = repository.pending_dispatches()[0]
    repository.mark_dispatch_published(message_id=message.message_id)
    first_claim = repository.claim(
        message,
        worker_id="ledger-crashed-worker",
        lease_seconds=0.05,
    )
    assert first_claim is not None
    calls: list[str] = []

    def crash(point: str) -> None:
        if point == "after_tool_before_receipt":
            raise _ToolCrash()

    executor, action, authorization, context = _ledger_contract(
        repository,
        first_claim,
        calls,
        failpoint=crash,
    )
    with pytest.raises(_ToolCrash):
        executor.execute(action, authorization, context)

    deadline = time.monotonic() + 2
    while repository.recover_expired_attempts() == 0:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    next_message = repository.pending_dispatches()[0]
    repository.mark_dispatch_published(message_id=next_message.message_id)
    second_claim = repository.claim(
        next_message,
        worker_id="ledger-recovery-worker",
        lease_seconds=5,
    )
    assert second_claim is not None
    recovered, action, authorization, context = _ledger_contract(
        repository,
        second_claim,
        calls,
    )
    with pytest.raises(
        ToolOperationOutcomeUncertainError,
        match="non_idempotent_execution_uncertain",
    ):
        recovered.execute(action, authorization, context)

    assert calls == ["one"]
    operation = repository.get_tool_operation(
        user_id=USER_A,
        run_id=created.run_id,
        operation_id=compute_tool_operation_id(
            run_id=created.run_id,
            action=action,
        ),
    )
    assert operation.state == ToolOperationState.UNCERTAIN


def test_create_run_event_and_outbox_are_atomic_and_idempotent(repository):
    first = _create(repository)
    second = _create(repository)
    assert first.run_id == second.run_id
    with psycopg.connect(DATABASE_URL) as connection:
        counts = connection.execute(
            """
            SELECT
              (SELECT count(*) FROM runs) AS runs,
              (SELECT count(*) FROM run_events) AS events,
              (SELECT count(*) FROM outbox_messages) AS outbox
            """
        ).fetchone()
    assert counts == (1, 1, 1)


def test_concurrent_postgres_idempotency_creates_one_run_and_outbox(repository):
    with ThreadPoolExecutor(max_workers=8) as executor:
        run_ids = list(executor.map(lambda _index: _create(repository).run_id, range(8)))
    assert len(set(run_ids)) == 1
    with psycopg.connect(DATABASE_URL) as connection:
        assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM outbox_messages").fetchone()[0] == 1


def test_same_idempotency_key_with_different_payload_conflicts(repository):
    _create(repository)
    service = RunService(repository=repository, dispatch_queue=None)
    with pytest.raises(IdempotencyConflictError, match="payload_mismatch"):
        service.create_run(
            RequestContext(user_id=USER_A, request_id="request-002"),
            CreateRunRequest.model_validate(_payload(title="Changed")),
            idempotency_key="persistent-key",
        )


def test_queued_run_admission_is_user_scoped_global_and_idempotent(repository):
    repository.ensure_users((USER_B,))
    service = RunService(
        repository=repository,
        dispatch_queue=None,
        max_queued_global=2,
        max_queued_per_user=1,
    )
    app = create_app(
        service=service,
        worker=None,
        identity_adapter=ConfiguredHeaderIdentityAdapter((USER_A, USER_B)),
    )
    with TestClient(app) as client:
        first = client.post(
            "/api/v1/runs",
            headers={"X-User-ID": USER_A, "Idempotency-Key": "capacity-a-1"},
            json=_payload(),
        )
        duplicate = client.post(
            "/api/v1/runs",
            headers={"X-User-ID": USER_A, "Idempotency-Key": "capacity-a-1"},
            json=_payload(),
        )
        user_limited = client.post(
            "/api/v1/runs",
            headers={"X-User-ID": USER_A, "Idempotency-Key": "capacity-a-2"},
            json=_payload(),
        )
        second_user = client.post(
            "/api/v1/runs",
            headers={"X-User-ID": USER_B, "Idempotency-Key": "capacity-b-1"},
            json=_payload(),
        )
        global_limited = client.post(
            "/api/v1/runs",
            headers={"X-User-ID": USER_B, "Idempotency-Key": "capacity-b-2"},
            json=_payload(),
        )

    assert first.status_code == 202
    assert duplicate.status_code == 202
    assert duplicate.json()["run_id"] == first.json()["run_id"]
    assert user_limited.status_code == 429
    assert user_limited.headers["Retry-After"] == "1"
    assert user_limited.json()["detail"] == {
        "code": "queued_run_capacity_exceeded",
        "scope": "user",
    }
    assert second_user.status_code == 202
    assert global_limited.status_code == 429
    assert global_limited.json()["detail"]["scope"] == "global"
    with psycopg.connect(DATABASE_URL) as connection:
        counts = connection.execute(
            """
            SELECT (SELECT count(*) FROM runs) AS runs,
                   (SELECT count(*) FROM outbox_messages) AS outbox
            """
        ).fetchone()
    assert counts == (2, 2)


class _DatabaseUnavailableService:
    def create_run(self, *_args, **_kwargs):
        raise psycopg.OperationalError("private_database_detail")


class _CapturedMetrics:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def record_api_request(self, **kwargs) -> None:
        self.requests.append(kwargs)

    def record_worker_task(self, **_kwargs) -> None:
        return None

    def record_provider_call(self, **_kwargs) -> None:
        return None


def test_database_runtime_failure_returns_sanitized_503_and_is_measured():
    metrics = _CapturedMetrics()
    app = create_app(
        service=_DatabaseUnavailableService(),
        worker=None,
        identity_adapter=ConfiguredHeaderIdentityAdapter((USER_A,)),
        metrics_recorder=metrics,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/runs",
            headers={"X-User-ID": USER_A, "Idempotency-Key": "database-down"},
            json=_payload(),
        )
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert response.json() == {"detail": {"code": "database_unavailable"}}
    assert "private_database_detail" not in response.text
    assert metrics.requests[-1]["status_code"] == 503


def test_fault_after_run_insert_rolls_back_run_event_and_outbox(repository, monkeypatch):
    def fail_event(*_args, **_kwargs):
        raise RuntimeError("injected_event_failure")

    monkeypatch.setattr(repository, "_append_event", fail_event)
    with pytest.raises(RuntimeError, match="injected_event_failure"):
        _create(repository, key="rollback-key")
    with psycopg.connect(DATABASE_URL) as connection:
        assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM run_events").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM outbox_messages").fetchone()[0] == 0


class _CountingAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, _record, *, progress_callback, claim=None):
        self.calls += 1
        progress_callback("mock_provider")
        return AgentExecutionOutcome(
            result={"runtime_mode": "agent_api", "llm_call_count": 1},
            provider_call_count=1,
        )


def test_duplicate_stream_delivery_executes_once(repository, stream_queue):
    created = _create(repository)
    message = repository.pending_dispatches()[0]
    stream_queue.publish(message)
    stream_queue.publish(message)
    repository.mark_dispatch_published(message_id=message.message_id)
    adapter = _CountingAdapter()
    worker = ReliableWorker(
        repository=repository,
        dispatch_queue=stream_queue,
        execution_adapter=adapter,
        worker_id="worker-1",
        lease_seconds=5,
    )
    assert worker.receive_and_process(block_ms=100)
    assert worker.receive_and_process(block_ms=100)
    assert adapter.calls == 1
    assert repository.get_for_user(user_id=USER_A, run_id=created.run_id).status == RunStatus.SUCCEEDED
    assert stream_queue.pending() == 0


def test_backpressure_outbox_waits_until_run_is_dispatchable(repository):
    _create(repository, key="delayed-backpressure")
    message = repository.pending_dispatches()[0]
    repository.mark_dispatch_published(message_id=message.message_id)
    claim = repository.claim(message, worker_id="backpressure-worker", lease_seconds=5)
    assert claim is not None

    repository.defer_for_backpressure(
        claim,
        error_code="provider_user_backpressure",
        delay_seconds=0.15,
    )

    assert repository.pending_dispatches() == []
    time.sleep(0.18)
    ready = repository.pending_dispatches()
    assert len(ready) == 1
    assert ready[0].dispatch_generation == 2


def test_concurrent_claims_serialize_same_user_session(repository):
    first = _create(repository, key="same-session-1", session_id="shared-session")
    second = _create(repository, key="same-session-2", session_id="shared-session")
    messages = repository.pending_dispatches()
    assert len(messages) == 2
    for message in messages:
        repository.mark_dispatch_published(message_id=message.message_id)

    barrier = Barrier(2)

    def claim(message):
        barrier.wait()
        return repository.claim(message, worker_id=str(uuid4()), lease_seconds=5)

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(claim, messages))

    assert sum(item is not None for item in claims) == 1
    records = [
        repository.get_for_user(user_id=USER_A, run_id=run_id)
        for run_id in (first.run_id, second.run_id)
    ]
    assert sum(record.status == RunStatus.RUNNING for record in records) == 1
    queued = next(record for record in records if record.status == RunStatus.QUEUED)
    assert queued.current_stage == "session_wait"
    assert queued.dispatch_generation == 2
    assert repository.pending_dispatches() == []
    time.sleep(0.28)
    ready = repository.pending_dispatches()
    assert len(ready) == 1
    assert ready[0].run_id == queued.run_id


def test_same_user_different_sessions_can_be_claimed_concurrently(repository):
    _create(repository, key="different-session-1", session_id="session-a")
    _create(repository, key="different-session-2", session_id="session-b")
    messages = repository.pending_dispatches()

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(
            executor.map(
                lambda message: repository.claim(
                    message,
                    worker_id=str(uuid4()),
                    lease_seconds=5,
                ),
                messages,
            )
        )

    assert all(claim is not None for claim in claims)


def test_expired_lease_requeues_and_fences_old_worker(repository, stream_queue):
    created = _create(repository)
    message = repository.pending_dispatches()[0]
    claim = repository.claim(message, worker_id="crashed-worker", lease_seconds=0.15)
    assert claim is not None
    repository.mark_agent_active(claim)
    deadline = time.monotonic() + 2.0
    recovered_count = 0
    while recovered_count == 0 and time.monotonic() < deadline:
        recovered_count = repository.recover_expired_attempts()
        if recovered_count == 0:
            time.sleep(0.02)
    assert recovered_count == 1
    recovered = repository.get_for_user(user_id=USER_A, run_id=created.run_id)
    assert recovered.status == RunStatus.QUEUED
    assert recovered.dispatch_generation == 2
    with pytest.raises(StaleAttemptError, match="stale_or_expired_attempt"):
        repository.succeed(claim, result={"late": True}, provider_call_count=1)


def test_expired_lease_recovery_is_bounded(repository):
    created = _create(repository)
    message = repository.pending_dispatches()[0]
    claim = repository.claim(message, worker_id="crashed-worker", lease_seconds=0.05)
    assert claim is not None
    deadline = time.monotonic() + 2.0
    failed = repository.get_for_user(user_id=USER_A, run_id=created.run_id)
    while failed.status == RunStatus.RUNNING and time.monotonic() < deadline:
        assert repository.recover_expired_attempts(max_attempts=1) == 0
        failed = repository.get_for_user(user_id=USER_A, run_id=created.run_id)
        if failed.status == RunStatus.RUNNING:
            time.sleep(0.02)
    assert failed.status == RunStatus.FAILED
    assert failed.error_code == "worker_recovery_exhausted"


def test_outbox_publisher_moves_only_wakeup_metadata(repository, stream_queue):
    _create(repository)
    publisher = OutboxPublisher(repository=repository, dispatch_queue=stream_queue)
    assert publisher.publish_once() == 1
    assert stream_queue.depth() == 1
    delivery = stream_queue.receive(consumer_name="inspector", block_ms=100)
    assert delivery is not None
    assert stream_queue.depth() == 0
    assert stream_queue.pending() == 1
    assert delivery.message.user_id == USER_A
    assert delivery.message.run_id
    assert not hasattr(delivery.message, "request")
    assert stream_queue.ack(delivery.stream_message_id) == 1
    assert stream_queue.pending() == 0
    assert stream_queue.retained_entries() == 1


def test_approval_persistence_reuses_runtime_digest_and_is_idempotent(repository):
    created = _create(repository)
    action = AgentAction(
        action_id="action-001",
        tool_name="application.submit",
        tool_arguments={"job_id": "job-001"},
        expected_observation="submission receipt",
        progress_claim="submit application",
    )
    digest = compute_action_digest(action)
    request = ApprovalRequest(
        request_id="approval-request-001",
        action_digest=digest,
        tool_name=action.tool_name,
        effect=ToolEffect.EXTERNAL_IRREVERSIBLE_WRITE,
        target_summary="Example job",
        arguments_preview={"job_id": "job-001"},
        session_id="persistent-session",
        run_id=created.run_id,
    )
    first_id = repository.save_pending_approval(
        user_id=USER_A,
        run_id=created.run_id,
        request=request,
        pending_action=action,
    )
    second_id = repository.save_pending_approval(
        user_id=USER_A,
        run_id=created.run_id,
        request=request,
        pending_action=action,
    )
    assert first_id == second_id
    decision = ApprovalDecision(
        request_id=request.request_id,
        action_digest=digest,
        approved=True,
        reason="approved in test",
        session_id=request.session_id,
        run_id=request.run_id,
    )
    repository.resolve_approval(
        user_id=USER_A,
        run_id=created.run_id,
        decision=decision,
    )
    repository.resolve_approval(
        user_id=USER_A,
        run_id=created.run_id,
        decision=decision,
    )
    wrong = decision.model_copy(update={"action_digest": "0" * 64})
    with pytest.raises(ApprovalPersistenceError, match="scope_or_digest_mismatch"):
        repository.resolve_approval(
            user_id=USER_A,
            run_id=created.run_id,
            decision=wrong,
        )


def test_semantic_checkpoint_adapter_round_trips_existing_schema(repository):
    created = _create(repository)
    message = repository.pending_dispatches()[0]
    repository.mark_dispatch_published(message_id=message.message_id)
    claim = repository.claim(message, worker_id="checkpoint-worker", lease_seconds=5)
    assert claim is not None
    store = PostgresSemanticCheckpointStore(
        repository=repository,
        claim=claim,
    )
    checkpoint = SemanticGraphCheckpoint.create(
        session_id="persistent-session",
        run_id=created.run_id,
        input_hash="a" * 64,
        completed_node="jd_structurer",
        skill_versions={"jd-analysis": "v1"},
        prompt_versions={"jd": "v1"},
        state={"structured_jd": {"role_type": "backend"}},
        provider_traces=[],
    )
    store.save(checkpoint)
    assert store.exists()
    assert store.load() == checkpoint
    store.delete()
    assert not store.exists()


def test_stale_attempt_cannot_overwrite_or_delete_newer_checkpoint(repository):
    created = _create(repository, key="checkpoint-fence")
    first_message = repository.pending_dispatches()[0]
    repository.mark_dispatch_published(message_id=first_message.message_id)
    first_claim = repository.claim(
        first_message,
        worker_id="checkpoint-worker-old",
        lease_seconds=0.05,
    )
    assert first_claim is not None
    first_store = PostgresSemanticCheckpointStore(
        repository=repository,
        claim=first_claim,
    )
    first_checkpoint = SemanticGraphCheckpoint.create(
        session_id=first_claim.run.session_id,
        run_id=created.run_id,
        input_hash="a" * 64,
        completed_node="jd_structurer",
        skill_versions={"jd-analysis": "v1"},
        prompt_versions={"jd": "v1"},
        state={"attempt": "old"},
        provider_traces=[],
    )
    first_store.save(first_checkpoint)

    deadline = time.monotonic() + 2.0
    while repository.recover_expired_attempts() == 0:
        assert time.monotonic() < deadline
        time.sleep(0.02)

    second_message = repository.pending_dispatches()[0]
    repository.mark_dispatch_published(message_id=second_message.message_id)
    second_claim = repository.claim(
        second_message,
        worker_id="checkpoint-worker-new",
        lease_seconds=5,
    )
    assert second_claim is not None
    second_store = PostgresSemanticCheckpointStore(
        repository=repository,
        claim=second_claim,
    )
    second_checkpoint = SemanticGraphCheckpoint.create(
        session_id=second_claim.run.session_id,
        run_id=created.run_id,
        input_hash="a" * 64,
        completed_node="evidence_mapping",
        skill_versions={"jd-analysis": "v1"},
        prompt_versions={"jd": "v1"},
        state={"attempt": "new"},
        provider_traces=[],
    )
    second_store.save(second_checkpoint)

    with pytest.raises(StaleAttemptError, match="stale_or_expired_attempt"):
        first_store.save(first_checkpoint)
    with pytest.raises(StaleAttemptError, match="stale_or_expired_attempt"):
        first_store.delete()
    assert second_store.load() == second_checkpoint


def test_metrics_expose_control_worker_provider_queue_and_database_signals(
    repository, stream_queue
):
    prefix = f"job-agent:test-metrics:{uuid4()}"
    recorder = RedisMetricsRecorder(REDIS_URL, prefix=prefix)
    recorder.record_worker_task(outcome="succeeded", duration_seconds=0.25)
    recorder.record_provider_call(
        provider="mock",
        duration_seconds=0.01,
        succeeded=True,
        input_tokens=12,
        output_tokens=5,
    )
    metrics_service = PrometheusMetricsService(
        repository=repository,
        recorder=recorder,
        redis_url=REDIS_URL,
        stream_name=stream_queue.stream_name,
        consumer_group=stream_queue.group_name,
    )
    app = create_app(
        service=RunService(repository=repository, dispatch_queue=None),
        worker=None,
        identity_adapter=ConfiguredHeaderIdentityAdapter((USER_A,)),
        metrics_recorder=recorder,
        metrics_service=metrics_service,
    )
    headers = {"X-User-ID": USER_A, "X-Request-ID": "metrics-request"}
    with TestClient(app) as client:
        rejected = client.post("/api/v1/runs", headers=headers, json=_payload())
        assert rejected.status_code == 400
        accepted = client.post(
            "/api/v1/runs",
            headers=headers | {"Idempotency-Key": "metrics-idempotency"},
            json=_payload(),
        )
        assert accepted.status_code == 202
        message = repository.pending_dispatches()[0]
        repository.mark_dispatch_published(message_id=message.message_id)
        claim = repository.claim(
            message,
            worker_id="metrics-backpressure-worker",
            lease_seconds=5,
        )
        assert claim is not None
        repository.defer_for_backpressure(
            claim,
            error_code="provider_user_backpressure",
            delay_seconds=1,
        )
        response = client.get("/metrics")
    recorder.record_api_request(
        method="POST",
        route="/api/v1/runs",
        status_code=503,
        duration_seconds=0.01,
    )
    response = metrics_service.render()
    recorder.close()

    body = response
    assert 'job_agent_http_requests_total{method="POST",route="/api/v1/runs",status_class="2xx"} 1' in body
    assert 'job_agent_http_errors_total{method="POST",route="/api/v1/runs",status_class="4xx"} 1' in body
    assert 'job_agent_http_availability_errors_total{method="POST",route="/api/v1/runs",status_class="5xx"} 1' in body
    assert 'job_agent_http_expected_rejections_total{method="POST",route="/api/v1/runs",status_class="4xx"} 1' in body
    assert 'job_agent_worker_tasks_total{outcome="succeeded"} 1' in body
    assert 'job_agent_provider_calls_total{provider="mock",outcome="success"} 1' in body
    assert 'job_agent_provider_tokens_total{provider="mock",direction="input"} 12' in body
    assert 'job_agent_provider_tokens_total{provider="mock",direction="output"} 5' in body
    assert 'job_agent_provider_admission_deferrals_total{reason="provider_user_backpressure"} 1' in body
    assert 'job_agent_runs{status="QUEUED"} 1' in body
    assert "job_agent_queue_depth 0" in body
    assert "job_agent_run_end_to_end_seconds_count 0" in body
    assert "job_agent_approval_wait_seconds_count 0" in body


def test_metrics_recorder_cools_down_after_redis_failure() -> None:
    class FailingPipeline:
        def __init__(self, owner) -> None:
            self.owner = owner

        def hincrby(self, *_args) -> None:
            return None

        def hincrbyfloat(self, *_args) -> None:
            return None

        def execute(self) -> None:
            self.owner.execute_calls += 1
            raise RedisError("redis unavailable")

    class FailingRedis:
        def __init__(self) -> None:
            self.execute_calls = 0

        def pipeline(self, *, transaction: bool):
            assert transaction is False
            return FailingPipeline(self)

        def close(self) -> None:
            return None

    recorder = RedisMetricsRecorder(
        "redis://127.0.0.1:1/0",
        failure_cooldown_seconds=60,
    )
    recorder._redis.close()
    failing = FailingRedis()
    recorder._redis = failing

    recorder.record_api_request(
        method="POST",
        route="/api/v1/runs",
        status_code=202,
        duration_seconds=0.01,
    )
    recorder.record_api_request(
        method="GET",
        route="/health/live",
        status_code=200,
        duration_seconds=0.01,
    )

    assert failing.execute_calls == 1
    recorder.close()


def test_api_metrics_recording_does_not_block_response() -> None:
    class BlockingMetricsRecorder:
        def __init__(self) -> None:
            self.started = Event()
            self.release = Event()

        def record_api_request(self, **_kwargs) -> None:
            self.started.set()
            self.release.wait(timeout=2)

        def record_worker_task(self, **_kwargs) -> None:
            return None

        def record_provider_call(self, **_kwargs) -> None:
            return None

    recorder = BlockingMetricsRecorder()
    app = create_app(
        service=RunService(repository=object(), dispatch_queue=None),
        worker=None,
        identity_adapter=ConfiguredHeaderIdentityAdapter((USER_A,)),
        metrics_recorder=recorder,
    )
    started = time.perf_counter()
    with TestClient(app) as client:
        response = client.get("/health/live")
        elapsed = time.perf_counter() - started
        assert response.status_code == 200
        assert recorder.started.wait(timeout=1)
        assert elapsed < 0.5
        recorder.release.set()
