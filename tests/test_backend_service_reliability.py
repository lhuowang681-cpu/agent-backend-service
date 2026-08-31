from __future__ import annotations

import time
import json
from dataclasses import dataclass

import psycopg
import pytest
from fastapi.testclient import TestClient
from pydantic import Field
from psycopg import sql
from redis import Redis

from job_agent.agent_runtime.budgets import AgentBudget, BudgetManager, BudgetProfile
from job_agent.agent_runtime.contracts import (
    AgentAction,
    AgentFinish,
    ApprovalRequest,
    ToolContext,
    ToolEffect,
    VerificationResult,
)
from job_agent.agent_runtime.loop import AgentLoop, ScriptedAgentModel
from job_agent.agent_runtime.policy import PolicyContext, PolicyEngine, compute_action_digest
from job_agent.agent_runtime.tool_registry import ToolRegistry
from job_agent.backend_service.api import ConfiguredHeaderIdentityAdapter, create_app
from job_agent.backend_service.contracts import (
    AgentApprovalPause,
    AgentExecutionOutcome,
    ApprovalDecisionRequest,
    CreateRunRequest,
    RequestContext,
    RunStatus,
)
from job_agent.backend_service.execution import (
    AgentExecutionFailedError,
    DomainAgentExecutionAdapter,
    ManifestValidatingExecutionAdapter,
    RoutedExecutionAdapter,
    TaskRoutedExecutionAdapter,
)
from job_agent.backend_service.application_assistant import (
    build_application_assistant_loop,
)
from job_agent.backend_service.career_snapshot import VersionedCareerSnapshotStore
from job_agent.backend_service.execution_manifest import build_execution_manifest
from job_agent.backend_service.failures import FailureClassifier
from job_agent.backend_service.migration import apply_migrations
from job_agent.backend_service.outbox import OutboxPublisher
from job_agent.backend_service.postgres_repository import PostgresRunRepository
from job_agent.backend_service.provider_admission import (
    AdmissionControlledProvider,
    ProviderAdmissionRejected,
    RedisProviderAdmissionController,
)
from job_agent.backend_service.recovery import RecoveryCoordinator
from job_agent.backend_service.redis_streams import RedisStreamDispatchQueue
from job_agent.backend_service.reliable_worker import ReliableWorker
from job_agent.backend_service.repository import RunNotFoundError
from job_agent.backend_service.service import RunService
from job_agent.llm.provider import (
    ProviderRateLimitError,
    ProviderTransportError,
    TraceContext,
)
from job_agent.llm.providers.mock import MockLLMProvider
from job_agent.schemas import StrictModel


ADMIN_URL = "postgresql://job_agent:job_agent_dev@127.0.0.1:54329/job_agent"
DATABASE_URL = "postgresql://job_agent:job_agent_dev@127.0.0.1:54329/job_agent_test"
REDIS_URL = "redis://127.0.0.1:63799/14"
USER_A = "reliability-user-a"
USER_B = "reliability-user-b"


def _payload() -> dict[str, object]:
    return {
        "task_type": "semantic_job_flow",
        "session_id": "reliability-session",
        "input": {
            "selected_job": {
                "job_id": "job-reliability",
                "company": "Example",
                "title": "Agent Backend Intern",
                "desc": "reliability",
                "url": "https://example.invalid/reliability",
                "location": "Beijing",
            },
            "resume_ref": "artifact://demo/resume",
        },
        "provider_profile": "mock",
        "budget_profile": "quick",
    }


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


def _build_application_adapter(repository, workspace_root, *, tool_failpoint=None):
    snapshots = VersionedCareerSnapshotStore(repository)
    domain = DomainAgentExecutionAdapter(
        repository=repository,
        loop_factory=lambda record, checkpoint_store, claim: (
            build_application_assistant_loop(
                record=record,
                snapshot=snapshots.resolve(
                    user_id=record.user_id,
                    snapshot_revision=(
                        record.execution_manifest.career_snapshot_revision
                        if record.execution_manifest is not None
                        and record.execution_manifest.career_snapshot_revision is not None
                        else ""
                    ),
                ),
                checkpoint_store=checkpoint_store,
                repository=repository,
                claim=claim,
                tool_failpoint=tool_failpoint,
            )
        ),
        goal_factory=lambda record: record.request.input.model_dump(mode="json"),
        tool_context_factory=lambda record, agent_id: ToolContext(
            session_id=record.session_id,
            run_id=record.run_id,
            agent_id=agent_id,
            workspace_root=str(workspace_root / record.user_id / record.run_id),
            sandbox_root=str(
                workspace_root / record.user_id / record.run_id / "sandbox"
            ),
        ),
    )
    return ManifestValidatingExecutionAdapter(
        adapter=TaskRoutedExecutionAdapter({"application_assistant_flow": domain}),
        expected_manifest_factory=lambda record: build_execution_manifest(
            record.request,
            provider_models={
                "mock": "fixture-model",
                "deepseek_flash": "deepseek-chat",
            },
        ),
    )


@pytest.fixture
def repository():
    try:
        with psycopg.connect(ADMIN_URL, autocommit=True) as connection:
            if connection.execute(
                "SELECT 1 FROM pg_database WHERE datname = 'job_agent_test'"
            ).fetchone() is None:
                connection.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier("job_agent_test"))
                )
        apply_migrations(DATABASE_URL)
    except psycopg.OperationalError:
        pytest.skip("Phase 4 PostgreSQL is not running")
    repo = PostgresRunRepository(DATABASE_URL)
    repo.open()
    repo.ensure_users((USER_A, USER_B))
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
    except ConnectionError:
        pytest.skip("Phase 4 Redis is not running")
    stream = "job-agent:reliability-runs"
    client.delete(stream)
    queue = RedisStreamDispatchQueue(
        REDIS_URL,
        stream_name=stream,
        group_name="reliability-workers",
    )
    queue.ensure_group()
    yield queue
    queue.close()
    client.delete(stream)
    client.close()


def _create(repository: PostgresRunRepository, *, key: str):
    return RunService(repository=repository, dispatch_queue=None).create_run(
        RequestContext(user_id=USER_A, request_id=f"request-{key}"),
        CreateRunRequest.model_validate(_payload()),
        idempotency_key=key,
    )


def _publish_pending(repository, queue, *, message_index: int = 0):
    message = repository.pending_dispatches()[message_index]
    queue.publish(message)
    repository.mark_dispatch_published(message_id=message.message_id)
    return message


def _approval_pause(run_id: str) -> AgentApprovalPause:
    action = AgentAction(
        action_id="action-submit",
        tool_name="application.submit",
        tool_arguments={"job_id": "job-reliability"},
        expected_observation="receipt",
        progress_claim="submit",
    )
    return AgentApprovalPause(
        pending_action=action,
        approval_request=ApprovalRequest(
            request_id="approval-request",
            action_digest=compute_action_digest(action),
            tool_name=action.tool_name,
            effect=ToolEffect.EXTERNAL_IRREVERSIBLE_WRITE,
            target_summary="job application",
            arguments_preview={"job_id": "job-reliability"},
            session_id="reliability-session",
            run_id=run_id,
        ),
    )


class _ApprovalAdapter:
    def __init__(
        self,
        pause: AgentApprovalPause,
        *,
        repository: PostgresRunRepository,
    ) -> None:
        self.pause = pause
        self.repository = repository
        self.calls = 0

    def execute(self, _record, *, progress_callback, claim=None):
        assert claim is not None
        self.calls += 1
        self.repository.save_private_checkpoint(
            claim=claim,
            checkpoint_kind="test_domain",
            checkpoint_id="checkpoint-approval",
            payload={"pending": True},
        )
        progress_callback("approval_required")
        return self.pause


class _SequenceAdapter:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def execute(self, _record, *, progress_callback, claim=None):
        self.calls += 1
        progress_callback("provider")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, AgentExecutionFailedError):
            raise outcome
        return outcome


def test_waiting_approval_survives_restart_and_api_checks_digest_owner_and_cas(
    repository, stream_queue
):
    created = _create(repository, key="approval")
    _publish_pending(repository, stream_queue)
    adapter = _ApprovalAdapter(
        _approval_pause(created.run_id),
        repository=repository,
    )
    worker = ReliableWorker(
        repository=repository,
        dispatch_queue=stream_queue,
        execution_adapter=adapter,
        worker_id="approval-worker",
        lease_seconds=5,
    )
    assert worker.receive_and_process(block_ms=100)
    waiting = repository.get_for_user(user_id=USER_A, run_id=created.run_id)
    assert waiting.status == RunStatus.WAITING_APPROVAL

    reopened = PostgresRunRepository(DATABASE_URL)
    reopened.open()
    try:
        assert reopened.get_for_user(user_id=USER_A, run_id=created.run_id).status == RunStatus.WAITING_APPROVAL
    finally:
        reopened.close()

    event = repository.list_events(
        user_id=USER_A, run_id=created.run_id, after=0, limit=100
    ).items[-1]
    approval_id = str(event.payload["approval_id"])
    digest = str(event.payload["action_digest"])
    service = RunService(repository=repository, dispatch_queue=None)
    app = create_app(
        service=service,
        worker=None,
        identity_adapter=ConfiguredHeaderIdentityAdapter((USER_A, USER_B)),
    )
    with TestClient(app) as client:
        body = {
            "approval_id": approval_id,
            "action_digest": "0" * 64,
            "expected_status_version": waiting.status_version,
            "reason": "test",
        }
        mismatch = client.post(
            f"/api/v1/runs/{created.run_id}/approve",
            headers={"X-User-ID": USER_A},
            json=body,
        )
        assert mismatch.status_code == 409
        assert mismatch.json()["detail"]["code"] == "approval_scope_or_digest_mismatch"
        assert client.post(
            f"/api/v1/runs/{created.run_id}/approve",
            headers={"X-User-ID": USER_B},
            json={**body, "action_digest": digest},
        ).status_code == 404
        approved = client.post(
            f"/api/v1/runs/{created.run_id}/approve",
            headers={"X-User-ID": USER_A},
            json={**body, "action_digest": digest},
        )
        assert approved.status_code == 202
        assert approved.json()["status"] == "QUEUED"
        duplicate = client.post(
            f"/api/v1/runs/{created.run_id}/approve",
            headers={"X-User-ID": USER_A},
            json={**body, "action_digest": digest},
        )
        assert duplicate.status_code == 202
        assert duplicate.json()["status_version"] == approved.json()["status_version"]
    assert adapter.calls == 1


@pytest.mark.parametrize("error_code", ["rate_limit", "timeout"])
def test_provider_429_and_timeout_have_bounded_safe_retry(
    repository, stream_queue, error_code
):
    created = _create(repository, key=f"retry-{error_code}")
    _publish_pending(repository, stream_queue)
    adapter = _SequenceAdapter(
        [
            AgentExecutionFailedError(error_code),
            AgentExecutionOutcome(result={"ok": True}, provider_call_count=1),
        ]
    )
    worker = ReliableWorker(
        repository=repository,
        dispatch_queue=stream_queue,
        execution_adapter=adapter,
        worker_id=f"worker-{error_code}",
        lease_seconds=5,
        failure_classifier=FailureClassifier(
            max_attempts=2,
            base_delay_seconds=0,
            jitter=lambda: 0,
        ),
    )
    assert worker.receive_and_process(block_ms=100)
    queued = repository.get_for_user(user_id=USER_A, run_id=created.run_id)
    assert queued.status == RunStatus.QUEUED
    assert queued.retry_count == 1
    _publish_pending(repository, stream_queue)
    assert worker.receive_and_process(block_ms=100)
    assert repository.get_for_user(user_id=USER_A, run_id=created.run_id).status == RunStatus.SUCCEEDED
    assert adapter.calls == 2


def test_non_idempotent_unknown_result_enters_uncertain_without_replay(
    repository, stream_queue
):
    created = _create(repository, key="uncertain")
    message = _publish_pending(repository, stream_queue)
    stream_queue.publish(message)
    adapter = _SequenceAdapter(
        [AgentExecutionFailedError("non_idempotent_execution_uncertain")]
    )
    worker = ReliableWorker(
        repository=repository,
        dispatch_queue=stream_queue,
        execution_adapter=adapter,
        worker_id="uncertain-worker",
        lease_seconds=5,
    )
    assert worker.receive_and_process(block_ms=100)
    assert worker.receive_and_process(block_ms=100)
    uncertain = repository.get_for_user(user_id=USER_A, run_id=created.run_id)
    assert uncertain.status == RunStatus.UNCERTAIN
    assert adapter.calls == 1
    assert repository.recover_expired_attempts() == 0


class _SimulatedCrash(BaseException):
    pass


def test_model_completed_before_state_write_replays_after_lease_recovery(
    repository, stream_queue
):
    created = _create(repository, key="after-model-crash")
    _publish_pending(repository, stream_queue)
    adapter = _SequenceAdapter(
        [
            AgentExecutionOutcome(result={"attempt": 1}, provider_call_count=1),
            AgentExecutionOutcome(result={"attempt": 2}, provider_call_count=1),
        ]
    )

    def crash(point: str) -> None:
        if point == "after_agent_before_commit":
            raise _SimulatedCrash()

    crashed = ReliableWorker(
        repository=repository,
        dispatch_queue=stream_queue,
        execution_adapter=adapter,
        worker_id="crashed-after-model",
        lease_seconds=0.05,
        failpoint=crash,
    )
    with pytest.raises(_SimulatedCrash):
        crashed.receive_and_process(block_ms=100)
    time.sleep(0.2)
    assert repository.recover_expired_attempts(max_attempts=3) == 1
    _publish_pending(repository, stream_queue)
    recovered = ReliableWorker(
        repository=repository,
        dispatch_queue=stream_queue,
        execution_adapter=adapter,
        worker_id="recovered-after-model",
        lease_seconds=5,
    )
    assert recovered.receive_and_process(block_ms=100)
    run = repository.get_for_user(user_id=USER_A, run_id=created.run_id)
    assert run.status == RunStatus.SUCCEEDED
    assert run.result == {"attempt": 2}
    assert adapter.calls == 2


def test_worker_crash_before_claim_is_reclaimed_from_pel(repository, stream_queue):
    created = _create(repository, key="pel-reclaim")
    _publish_pending(repository, stream_queue)
    dead_delivery = stream_queue.receive(consumer_name="dead-worker", block_ms=100)
    assert dead_delivery is not None
    time.sleep(0.02)
    reclaimed = stream_queue.reclaim(
        consumer_name="live-worker",
        min_idle_ms=1,
    )
    assert len(reclaimed) == 1
    adapter = _SequenceAdapter(
        [AgentExecutionOutcome(result={"ok": True}, provider_call_count=1)]
    )
    worker = ReliableWorker(
        repository=repository,
        dispatch_queue=stream_queue,
        execution_adapter=adapter,
        worker_id="live-worker",
        lease_seconds=5,
    )
    worker.process(reclaimed[0])
    assert repository.get_for_user(user_id=USER_A, run_id=created.run_id).status == RunStatus.SUCCEEDED
    assert adapter.calls == 1


def test_redis_loss_is_rebuilt_from_postgres_outbox(repository, stream_queue):
    created = _create(repository, key="redis-loss")
    publisher = OutboxPublisher(repository=repository, dispatch_queue=stream_queue)
    assert publisher.publish_once() == 1
    raw = Redis.from_url(REDIS_URL, decode_responses=True)
    raw.delete(stream_queue.stream_name)
    coordinator = RecoveryCoordinator(
        repository=repository,
        dispatch_queue=stream_queue,
        publisher=publisher,
        max_attempts=3,
        dispatch_reconcile_grace_seconds=0.5,
    )
    with psycopg.connect(DATABASE_URL) as connection:
        connection.execute(
            """
            UPDATE runs SET updated_at = clock_timestamp() - interval '1 second'
            WHERE run_id = %s
            """,
            (created.run_id,),
        )
    summary = coordinator.run_once()
    assert summary.dispatches_reconciled == 1
    delivery = stream_queue.receive(consumer_name="after-redis-loss", block_ms=100)
    assert delivery is not None
    assert delivery.message.run_id == created.run_id
    raw.close()


def test_redis_loss_is_reconciled_while_other_backlog_remains(
    repository, stream_queue
):
    lost = _create(repository, key="redis-loss-with-backlog-1")
    retained = _create(repository, key="redis-loss-with-backlog-2")
    publisher = OutboxPublisher(repository=repository, dispatch_queue=stream_queue)
    assert publisher.publish_once() == 2

    raw = Redis.from_url(REDIS_URL, decode_responses=True)
    entries = raw.xrange(stream_queue.stream_name)
    lost_stream_id = next(
        stream_id
        for stream_id, fields in entries
        if fields["run_id"] == lost.run_id
    )
    assert raw.xdel(stream_queue.stream_name, lost_stream_id) == 1
    assert stream_queue.depth() == 1
    with psycopg.connect(DATABASE_URL) as connection:
        connection.execute(
            """
            UPDATE runs SET updated_at = clock_timestamp() - interval '1 second'
            WHERE run_id IN (%s, %s)
            """,
            (lost.run_id, retained.run_id),
        )

    coordinator = RecoveryCoordinator(
        repository=repository,
        dispatch_queue=stream_queue,
        publisher=publisher,
        max_attempts=3,
        dispatch_reconcile_grace_seconds=0.5,
    )
    first = coordinator.run_once()
    assert first.dispatches_reconciled == 2
    second = coordinator.run_once()
    assert second.dispatches_reconciled == 0

    deliveries = [
        stream_queue.receive(consumer_name="backlog-worker", block_ms=100)
        for _ in range(3)
    ]
    delivered_run_ids = {
        delivery.message.run_id for delivery in deliveries if delivery is not None
    }
    assert lost.run_id in delivered_run_ids
    assert retained.run_id in delivered_run_ids
    raw.close()


def test_post_remains_202_and_outbox_pending_when_redis_is_unavailable(repository):
    app = create_app(
        service=RunService(repository=repository, dispatch_queue=None),
        worker=None,
        identity_adapter=ConfiguredHeaderIdentityAdapter((USER_A, USER_B)),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/runs",
            headers={
                "X-User-ID": USER_A,
                "Idempotency-Key": "redis-unavailable",
            },
            json=_payload(),
        )
    assert response.status_code == 202
    unavailable = RedisStreamDispatchQueue(
        "redis://127.0.0.1:1/0",
        stream_name="unavailable",
        group_name="unavailable",
    )
    publisher = OutboxPublisher(repository=repository, dispatch_queue=unavailable)
    assert publisher.publish_once() == 0
    assert len(repository.pending_dispatches()) == 1
    with psycopg.connect(DATABASE_URL) as connection:
        assert connection.execute(
            "SELECT last_error_code FROM outbox_messages"
        ).fetchone()[0] == "redis_unavailable"
    unavailable.close()


def test_provider_admission_enforces_fairness_cooldown_circuit_and_fail_closed():
    raw = Redis.from_url(REDIS_URL, decode_responses=True)
    raw.flushdb()
    controller = RedisProviderAdmissionController(
        REDIS_URL,
        prefix="job-agent:test-provider",
        global_limit=2,
        per_user_limit=1,
        permit_ttl_seconds=1,
        circuit_failure_threshold=2,
        circuit_cooldown_seconds=0.1,
    )
    first = controller.acquire(user_id=USER_A, provider_key="mock")
    with pytest.raises(ProviderAdmissionRejected, match="provider_user_backpressure"):
        controller.acquire(user_id=USER_A, provider_key="mock")
    second = controller.acquire(user_id=USER_B, provider_key="mock")
    with pytest.raises(ProviderAdmissionRejected, match="provider_global_backpressure"):
        controller.acquire(user_id="user-c", provider_key="mock")
    controller.record_rate_limit(first, retry_after_seconds=0.1)
    with pytest.raises(ProviderAdmissionRejected, match="provider_user_cooldown"):
        controller.acquire(user_id=USER_A, provider_key="mock")
    controller.record_failure(second)
    probe = controller.acquire(user_id="user-c", provider_key="mock")
    controller.record_failure(probe)
    with pytest.raises(ProviderAdmissionRejected, match="provider_circuit_open"):
        controller.acquire(user_id=USER_B, provider_key="mock")
    time.sleep(0.12)
    allowed = controller.acquire(user_id=USER_B, provider_key="mock")
    controller.record_success(allowed)
    controller.close()
    raw.close()

    unavailable = RedisProviderAdmissionController("redis://127.0.0.1:1/0")
    with pytest.raises(ProviderAdmissionRejected, match="provider_limiter_unavailable"):
        unavailable.acquire(user_id=USER_A, provider_key="mock")
    unavailable.close()


class _ProviderOutput(StrictModel):
    ok: bool


def _call_provider(provider) -> None:
    provider.generate_structured(
        system_prompt="system",
        user_prompt="user",
        output_schema=_ProviderOutput,
        tools=(),
        temperature=0,
        max_output_tokens=32,
        trace=TraceContext(
            session_id="provider-session",
            run_id="provider-run",
            node_id="provider-node",
            skill_id="provider-skill",
            skill_version="v1",
            prompt_version="v1",
        ),
    )


def test_provider_admission_is_call_scoped_and_local_errors_do_not_open_circuit():
    raw = Redis.from_url(REDIS_URL, decode_responses=True)
    raw.flushdb()
    controller = RedisProviderAdmissionController(
        REDIS_URL,
        prefix="job-agent:test-call-scoped-provider",
        global_limit=1,
        per_user_limit=1,
        permit_ttl_seconds=1,
        circuit_failure_threshold=1,
        circuit_cooldown_seconds=0.1,
    )

    successful = AdmissionControlledProvider(
        MockLLMProvider([{"ok": True}]),
        controller=controller,
        user_id=USER_A,
        provider_key="mock",
    )
    _call_provider(successful)
    released = controller.acquire(user_id=USER_A, provider_key="mock")
    controller.release(released)

    local_failure = AdmissionControlledProvider(
        MockLLMProvider([ValueError("local_contract_bug")]),
        controller=controller,
        user_id=USER_A,
        provider_key="mock",
    )
    with pytest.raises(ValueError, match="local_contract_bug"):
        _call_provider(local_failure)
    still_healthy = controller.acquire(user_id=USER_B, provider_key="mock")
    controller.release(still_healthy)

    transport_failure = AdmissionControlledProvider(
        MockLLMProvider([ProviderTransportError("network_down")]),
        controller=controller,
        user_id=USER_A,
        provider_key="mock",
    )
    with pytest.raises(ProviderTransportError, match="network_down"):
        _call_provider(transport_failure)
    with pytest.raises(ProviderAdmissionRejected, match="provider_circuit_open"):
        controller.acquire(user_id=USER_B, provider_key="mock")

    time.sleep(0.12)
    rate_limited = AdmissionControlledProvider(
        MockLLMProvider([ProviderRateLimitError("slow_down")]),
        controller=controller,
        user_id=USER_A,
        provider_key="mock",
    )
    with pytest.raises(ProviderRateLimitError, match="slow_down"):
        _call_provider(rate_limited)
    with pytest.raises(ProviderAdmissionRejected, match="provider_user_cooldown"):
        controller.acquire(user_id=USER_A, provider_key="mock")
    other_user = controller.acquire(user_id=USER_B, provider_key="mock")
    controller.record_success(other_user)
    controller.close()
    raw.close()


def test_resume_allows_safe_failed_run_but_never_uncertain(repository, stream_queue):
    safe = _create(repository, key="safe-resume")
    safe_message = repository.pending_dispatches()[0]
    safe_claim = repository.claim(safe_message, worker_id="failer", lease_seconds=5)
    assert safe_claim is not None
    failed = repository.fail(safe_claim, error_code="worker_recovery_exhausted")

    uncertain_created = _create(repository, key="no-uncertain-resume")
    messages = repository.pending_dispatches()
    uncertain_message = next(
        message for message in messages if message.run_id == uncertain_created.run_id
    )
    uncertain_claim = repository.claim(
        uncertain_message, worker_id="uncertain", lease_seconds=5
    )
    assert uncertain_claim is not None
    uncertain = repository.mark_uncertain(
        uncertain_claim,
        error_code="non_idempotent_execution_uncertain",
    )

    app = create_app(
        service=RunService(repository=repository, dispatch_queue=None),
        worker=None,
        identity_adapter=ConfiguredHeaderIdentityAdapter((USER_A, USER_B)),
    )
    with TestClient(app) as client:
        resumed = client.post(
            f"/api/v1/runs/{safe.run_id}/resume",
            headers={"X-User-ID": USER_A},
            json={"expected_status_version": failed.status_version},
        )
        assert resumed.status_code == 202
        assert resumed.json()["status"] == "QUEUED"
        blocked = client.post(
            f"/api/v1/runs/{uncertain.run_id}/resume",
            headers={"X-User-ID": USER_A},
            json={"expected_status_version": uncertain.status_version},
        )
        assert blocked.status_code == 409
        assert blocked.json()["detail"]["code"] == "uncertain_requires_reconciliation"


class _MutationInput(StrictModel):
    value: str = Field(min_length=1)


class _MutationOutput(StrictModel):
    value: str


class _CompletedVerifier:
    def verify(self, _finish, *, goal, steps):
        return VerificationResult(
            passed=bool(steps),
            reason_code="complete" if steps else "missing_tool_result",
            feedback="complete" if steps else "execute the approved action",
        )


def test_domain_agent_adapter_reuses_checkpoint_and_resolved_approval(
    repository, stream_queue, tmp_path
):
    created = _create(repository, key="domain-adapter-approval")
    _publish_pending(repository, stream_queue)
    tool_calls: list[str] = []
    factory_calls = 0

    action = AgentAction(
        action_id="mutation-1",
        tool_name="state.mutate",
        tool_arguments={"value": "approved"},
        expected_observation="updated state",
        progress_claim="apply approved mutation",
    )
    finish = AgentFinish(
        result={"status": "done"},
        completion_evidence=["step:1:state.mutate"],
        confidence=0.9,
    )

    def loop_factory(record, checkpoint_store, _claim):
        nonlocal factory_calls
        factory_calls += 1
        registry = ToolRegistry()
        registry.register(
            name="state.mutate",
            description="Mutate test state",
            input_model=_MutationInput,
            output_model=_MutationOutput,
            handler=lambda value, context: tool_calls.append(value.value)
            or _MutationOutput(value=value.value),
            effect=ToolEffect.LOCAL_STATE_MUTATION,
            idempotent=False,
        )
        return AgentLoop(
            agent_id="domain-adapter-agent",
            model=ScriptedAgentModel([action] if factory_calls == 1 else [finish]),
            registry=registry,
            policy=PolicyEngine(),
            policy_context=PolicyContext(
                skill_allowed_tools=("state.mutate",),
                agent_allowed_tools=("state.mutate",),
                runtime_allowed_tools=("state.mutate",),
            ),
            budget=BudgetManager(AgentBudget.for_profile(BudgetProfile.QUICK)),
            verifier=_CompletedVerifier(),
            checkpoint_store=checkpoint_store,
            skill_version="domain-adapter-test-v1",
            prompt_version="domain-adapter-prompt-v1",
        )

    adapter = DomainAgentExecutionAdapter(
        repository=repository,
        loop_factory=loop_factory,
        goal_factory=lambda record: {"operation": "approved mutation"},
        tool_context_factory=lambda record, agent_id: ToolContext(
            session_id=record.session_id,
            run_id=record.run_id,
            agent_id=agent_id,
            workspace_root=str(tmp_path),
            sandbox_root=str(tmp_path / "sandbox"),
        ),
    )
    worker = ReliableWorker(
        repository=repository,
        dispatch_queue=stream_queue,
        execution_adapter=adapter,
        worker_id="domain-adapter-worker",
        lease_seconds=5,
    )
    assert worker.receive_and_process(block_ms=100)
    waiting = repository.get_for_user(user_id=USER_A, run_id=created.run_id)
    assert waiting.status == RunStatus.WAITING_APPROVAL
    assert tool_calls == []

    approval_event = repository.list_events(
        user_id=USER_A, run_id=created.run_id, after=0, limit=100
    ).items[-1]
    RunService(repository=repository, dispatch_queue=None).decide_approval(
        RequestContext(user_id=USER_A, request_id="approve-domain-adapter"),
        created.run_id,
        ApprovalDecisionRequest(
            approval_id=str(approval_event.payload["approval_id"]),
            action_digest=str(approval_event.payload["action_digest"]),
            expected_status_version=waiting.status_version,
        ),
        approved=True,
    )
    _publish_pending(repository, stream_queue)
    assert worker.receive_and_process(block_ms=100)

    completed = repository.get_for_user(user_id=USER_A, run_id=created.run_id)
    assert completed.status == RunStatus.SUCCEEDED
    assert completed.result == {"status": "done"}
    assert tool_calls == ["approved"]
    assert factory_calls == 2


def test_deployed_application_assistant_survives_restart_and_writes_once(
    repository,
    stream_queue,
    tmp_path,
):
    snapshots = VersionedCareerSnapshotStore(repository)
    snapshot = snapshots.put(
        user_id=USER_A,
        source_ref="artifact://career/application-assistant-v1",
        resume_text="Built Agent approval, checkpoint, and evaluation contracts.",
        evidence_refs=["project:job-agent"],
    )
    service = RunService(
        repository=repository,
        dispatch_queue=None,
        career_snapshot_store=snapshots,
        execution_manifest_factory=lambda request: build_execution_manifest(
            request,
            provider_models={
                "mock": "fixture-model",
                "deepseek_flash": "deepseek-chat",
            },
        ),
    )
    created = service.create_run(
        RequestContext(user_id=USER_A, request_id="application-assistant-create"),
        CreateRunRequest.model_validate(
            _application_payload(snapshot.snapshot_revision)
        ),
        idempotency_key="application-assistant-restart",
    )
    _publish_pending(repository, stream_queue)

    first_worker = ReliableWorker(
        repository=repository,
        dispatch_queue=stream_queue,
        execution_adapter=_build_application_adapter(repository, tmp_path),
        worker_id="application-assistant-before-restart",
        lease_seconds=5,
    )
    assert first_worker.receive_and_process(block_ms=100)
    waiting = repository.get_for_user(user_id=USER_A, run_id=created.run_id)
    assert waiting.status == RunStatus.WAITING_APPROVAL
    assert list(tmp_path.rglob("*.json")) == []
    waiting_debug = repository.get_debug_view(
        user_id=USER_A,
        run_id=created.run_id,
    )
    assert len(waiting_debug.attempts) == 1
    assert len(waiting_debug.approvals) == 1
    assert waiting_debug.approvals[0].decision is None
    assert waiting_debug.checkpoint is not None

    approval_event = repository.list_events(
        user_id=USER_A,
        run_id=created.run_id,
        after=0,
        limit=100,
    ).items[-1]
    service.decide_approval(
        RequestContext(user_id=USER_A, request_id="application-assistant-approve"),
        created.run_id,
        ApprovalDecisionRequest(
            approval_id=str(approval_event.payload["approval_id"]),
            action_digest=str(approval_event.payload["action_digest"]),
            expected_status_version=waiting.status_version,
        ),
        approved=True,
    )
    _publish_pending(repository, stream_queue)

    restarted_repository = PostgresRunRepository(DATABASE_URL)
    restarted_repository.open()
    try:
        restarted_worker = ReliableWorker(
            repository=restarted_repository,
            dispatch_queue=stream_queue,
            execution_adapter=_build_application_adapter(
                restarted_repository,
                tmp_path,
            ),
            worker_id="application-assistant-after-restart",
            lease_seconds=5,
        )
        assert restarted_worker.receive_and_process(block_ms=100)
        completed = restarted_repository.get_for_user(
            user_id=USER_A,
            run_id=created.run_id,
        )
    finally:
        restarted_repository.close()

    assert completed.status == RunStatus.SUCCEEDED
    assert completed.result is not None
    assert completed.result["snapshot_revision"] == snapshot.snapshot_revision
    assert "resume_text" not in json.dumps(completed.result)
    drafts = list(tmp_path.rglob("draft-*.json"))
    assert len(drafts) == 1
    with psycopg.connect(DATABASE_URL) as connection:
        operation = connection.execute(
            """
            SELECT state, count(*) OVER () AS operation_count
            FROM tool_operations
            WHERE user_id = %s AND run_id = %s
            """,
            (USER_A, created.run_id),
        ).fetchone()
    assert operation == ("SUCCEEDED", 1)
    debug = repository.get_debug_view(user_id=USER_A, run_id=created.run_id)
    assert debug.status == RunStatus.SUCCEEDED
    assert len(debug.attempts) == 2
    assert debug.approvals[0].decision == "approved"
    assert debug.tool_operations[0].state == "SUCCEEDED"
    assert debug.execution_manifest is not None
    serialized_debug = json.dumps(debug.model_dump(mode="json"))
    assert snapshot.payload.resume_text not in serialized_debug
    assert "recruiting@example.invalid" not in serialized_debug
    assert "receipt_private" not in serialized_debug
    with pytest.raises(RunNotFoundError, match="resource_not_found"):
        repository.get_debug_view(user_id=USER_B, run_id=created.run_id)


def test_application_tool_crash_recovers_to_run_and_ledger_uncertain_without_replay(
    repository,
    stream_queue,
    tmp_path,
):
    snapshots = VersionedCareerSnapshotStore(repository)
    snapshot = snapshots.put(
        user_id=USER_A,
        source_ref="artifact://career/application-assistant-uncertain",
        resume_text="Built recovery guards for non-idempotent Agent tools.",
        evidence_refs=["project:job-agent"],
    )
    service = RunService(
        repository=repository,
        dispatch_queue=None,
        career_snapshot_store=snapshots,
        execution_manifest_factory=lambda request: build_execution_manifest(
            request,
            provider_models={
                "mock": "fixture-model",
                "deepseek_flash": "deepseek-chat",
            },
        ),
    )
    created = service.create_run(
        RequestContext(user_id=USER_A, request_id="application-uncertain-create"),
        CreateRunRequest.model_validate(_application_payload(snapshot.snapshot_revision)),
        idempotency_key="application-assistant-tool-uncertain",
    )
    _publish_pending(repository, stream_queue)

    initial_worker = ReliableWorker(
        repository=repository,
        dispatch_queue=stream_queue,
        execution_adapter=_build_application_adapter(repository, tmp_path),
        worker_id="application-uncertain-before-approval",
        lease_seconds=5,
    )
    assert initial_worker.receive_and_process(block_ms=100)
    waiting = repository.get_for_user(user_id=USER_A, run_id=created.run_id)
    assert waiting.status == RunStatus.WAITING_APPROVAL

    approval_event = repository.list_events(
        user_id=USER_A,
        run_id=created.run_id,
        after=0,
        limit=100,
    ).items[-1]
    service.decide_approval(
        RequestContext(user_id=USER_A, request_id="application-uncertain-approve"),
        created.run_id,
        ApprovalDecisionRequest(
            approval_id=str(approval_event.payload["approval_id"]),
            action_digest=str(approval_event.payload["action_digest"]),
            expected_status_version=waiting.status_version,
        ),
        approved=True,
    )
    _publish_pending(repository, stream_queue)

    def crash_after_tool(point: str) -> None:
        if point == "after_tool_before_receipt":
            raise _SimulatedCrash()

    crashed_worker = ReliableWorker(
        repository=repository,
        dispatch_queue=stream_queue,
        execution_adapter=_build_application_adapter(
            repository,
            tmp_path,
            tool_failpoint=crash_after_tool,
        ),
        worker_id="application-uncertain-crashed",
        lease_seconds=0.05,
    )
    with pytest.raises(_SimulatedCrash):
        crashed_worker.receive_and_process(block_ms=100)

    assert len(list(tmp_path.rglob("draft-*.json"))) == 1
    with psycopg.connect(DATABASE_URL) as connection:
        before_recovery = connection.execute(
            """
            SELECT state, receipt_private
            FROM tool_operations
            WHERE user_id = %s AND run_id = %s
            """,
            (USER_A, created.run_id),
        ).fetchone()
    assert before_recovery == ("INFLIGHT", None)

    time.sleep(0.2)
    assert repository.recover_expired_attempts(max_attempts=3) == 1
    _publish_pending(repository, stream_queue)

    restarted_repository = PostgresRunRepository(DATABASE_URL)
    restarted_repository.open()
    try:
        restarted_worker = ReliableWorker(
            repository=restarted_repository,
            dispatch_queue=stream_queue,
            execution_adapter=_build_application_adapter(
                restarted_repository,
                tmp_path,
            ),
            worker_id="application-uncertain-restarted",
            lease_seconds=5,
        )
        assert restarted_worker.receive_and_process(block_ms=100)
        uncertain = restarted_repository.get_for_user(
            user_id=USER_A,
            run_id=created.run_id,
        )
    finally:
        restarted_repository.close()

    assert uncertain.status == RunStatus.UNCERTAIN
    assert uncertain.error_code == "non_idempotent_execution_uncertain"
    assert len(list(tmp_path.rglob("draft-*.json"))) == 1
    with psycopg.connect(DATABASE_URL) as connection:
        operation = connection.execute(
            """
            SELECT state, error_code, receipt_private,
                   count(*) OVER () AS operation_count
            FROM tool_operations
            WHERE user_id = %s AND run_id = %s
            """,
            (USER_A, created.run_id),
        ).fetchone()
    assert operation == (
        "UNCERTAIN",
        "non_idempotent_execution_uncertain",
        None,
        1,
    )
    debug = repository.get_debug_view(user_id=USER_A, run_id=created.run_id)
    assert debug.status == RunStatus.UNCERTAIN
    assert debug.current_stage == "reconciliation_required"
    assert debug.attempts[-1].error_code == "non_idempotent_execution_uncertain"
    assert debug.attempts[0].recovery_kind == "approval_pause"
    assert debug.attempts[1].recovery_kind == "lease_expired_safe_replay"
    assert debug.tool_operations[0].state == "UNCERTAIN"
    assert debug.tool_operations[0].attempt_id == debug.attempts[-2].attempt_id
    assert debug.events[-1].event_type == "run.uncertain"


def test_persisted_deepseek_profile_routes_to_configured_adapter(
    repository, stream_queue
):
    payload = _payload()
    payload["provider_profile"] = "deepseek_flash"
    created = RunService(repository=repository, dispatch_queue=None).create_run(
        RequestContext(user_id=USER_A, request_id="deepseek-route-request"),
        CreateRunRequest.model_validate(payload),
        idempotency_key="deepseek-route",
    )
    _publish_pending(repository, stream_queue)
    deepseek = _SequenceAdapter(
        [AgentExecutionOutcome(result={"provider": "deepseek"}, provider_call_count=1)]
    )
    worker = ReliableWorker(
        repository=repository,
        dispatch_queue=stream_queue,
        execution_adapter=RoutedExecutionAdapter({"deepseek_flash": deepseek}),
        worker_id="deepseek-route-worker",
        lease_seconds=5,
    )

    assert worker.receive_and_process(block_ms=100)
    completed = repository.get_for_user(user_id=USER_A, run_id=created.run_id)
    assert completed.status == RunStatus.SUCCEEDED
    assert completed.result == {"provider": "deepseek"}
    assert deepseek.calls == 1
