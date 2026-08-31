from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event, Lock

import pytest
from fastapi.testclient import TestClient

from job_agent.backend_service.bootstrap import create_tracer_stack
from job_agent.backend_service.api import ConfiguredHeaderIdentityAdapter
from job_agent.backend_service.contracts import (
    AgentExecutionOutcome,
    CreateRunRequest,
    RequestContext,
    RunDispatchMessage,
    RunStatus,
)
from job_agent.backend_service.execution import (
    AgentExecutionFailedError,
    CareerSnapshotNotFoundError,
    InMemoryCareerSnapshotStore,
    ManifestValidatingExecutionAdapter,
    SchemaMappedMockLLMProvider,
    SemanticJobExecutionAdapter,
)
from job_agent.llm.provider import TraceContext
from job_agent.schemas import EvidenceMappingResult
from job_agent.backend_service.execution_manifest import build_execution_manifest
from job_agent.backend_service.queueing import InMemoryDispatchQueue
from job_agent.backend_service.repository import InMemoryRunRepository, QueueOwnershipError
from job_agent.backend_service.service import RunService
from scripts.report_backend_load import build_report


USER_A = "user-a"
USER_B = "user-b"
RESUME_REF = "artifact://resume/default"


def _request_payload(*, session_id: str = "session-001", title: str = "LLM Intern"):
    return {
        "task_type": "semantic_job_flow",
        "session_id": session_id,
        "input": {
            "selected_job": {
                "job_id": "job-001",
                "company": "Example",
                "title": title,
                "desc": "SFT LoRA RLHF evaluation internship",
                "url": "https://example.invalid/job",
                "location": "Beijing",
            },
            "resume_ref": RESUME_REF,
        },
        "provider_profile": "mock",
        "budget_profile": "quick",
    }


def test_schema_mapped_mock_starts_from_checkpoint_requested_schema() -> None:
    provider = SchemaMappedMockLLMProvider(
        [
            {"unused_first_node_fixture": True},
            {
                "items": [
                    {
                        "evidence_id": "evidence-1",
                        "requirement_id": "requirement-1",
                        "claim": "checkpoint resumes at the requested schema",
                        "level": "C1",
                        "proof": "deterministic fixture",
                        "risk": "none",
                    }
                ]
            },
        ]
    )

    result = provider.generate_structured(
        system_prompt="system",
        user_prompt="resume from evidence mapping",
        output_schema=EvidenceMappingResult,
        tools=(),
        temperature=0,
        max_output_tokens=64,
        trace=TraceContext(
            session_id="session",
            run_id="run",
            node_id="evidence_mapping",
            skill_id="evidence-contract",
            skill_version="v1",
            prompt_version="v1",
        ),
    )

    assert result.schema_valid is True
    assert result.parsed_output["items"][0]["evidence_id"] == "evidence-1"


def test_application_assistant_request_rejects_semantic_input_shape():
    payload = _request_payload()
    payload["task_type"] = "application_assistant_flow"
    with pytest.raises(ValueError, match="task_input_contract_mismatch"):
        CreateRunRequest.model_validate(payload)


def test_application_assistant_request_accepts_versioned_snapshot_contract():
    payload = _request_payload()
    payload["task_type"] = "application_assistant_flow"
    payload["input"] = {
        "selected_job": payload["input"]["selected_job"],
        "career_snapshot_revision": "snapshot-0123456789abcdef",
        "draft_channel": "email",
        "draft_target": "recruiting@example.invalid",
    }
    request = CreateRunRequest.model_validate(payload)
    assert request.task_type == "application_assistant_flow"
    assert request.input.career_snapshot_revision == "snapshot-0123456789abcdef"


def test_execution_manifest_mismatch_fails_closed_before_adapter_call():
    class CountingAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _record, *, progress_callback, claim=None):
            self.calls += 1
            return AgentExecutionOutcome(result={"ok": True}, provider_call_count=0)

    repository = InMemoryRunRepository()
    service = RunService(
        repository=repository,
        dispatch_queue=None,
        execution_manifest_factory=lambda request: build_execution_manifest(
            request,
            provider_models={"mock": "fixture-model"},
        ),
    )
    created = service.create_run(
        RequestContext(user_id=USER_A, request_id="manifest-request"),
        CreateRunRequest.model_validate(_request_payload()),
        idempotency_key="manifest-key",
    )
    record = repository.get_for_user(user_id=USER_A, run_id=created.run_id)
    inner = CountingAdapter()
    guarded = ManifestValidatingExecutionAdapter(
        adapter=inner,
        expected_manifest_factory=lambda run: build_execution_manifest(
            run.request,
            provider_models={"mock": "changed-model"},
        ),
    )

    with pytest.raises(
        AgentExecutionFailedError,
        match="execution_manifest_incompatible",
    ):
        guarded.execute(record, progress_callback=lambda _stage: None)
    assert inner.calls == 0


def _headers(user_id: str, key: str = "request-key-001") -> dict[str, str]:
    return {"X-User-ID": user_id, "Idempotency-Key": key}


def test_worker_pool_report_separates_workers_users_and_queue_wait() -> None:
    started = datetime(2026, 8, 2, tzinfo=timezone.utc)
    rows = []
    for index, (user_id, worker_id) in enumerate(
        (
            ("demo-user-a", "worker-a"),
            ("demo-user-b", "worker-b"),
            ("demo-user-a", "worker-a"),
            ("demo-user-b", "worker-b"),
        ),
        start=1,
    ):
        attempt_started = started + timedelta(seconds=index)
        finished = started + timedelta(seconds=index + 2)
        rows.append(
            {
                "run_id": f"run-{index}",
                "user_id": user_id,
                "run_status": "SUCCEEDED",
                "retry_count": 0,
                "created_at": started,
                "run_finished_at": finished,
                "attempt_no": 1,
                "worker_id": worker_id,
                "attempt_status": "SUCCEEDED",
                "attempt_error_code": None,
                "recovery_kind": None,
                "started_at": attempt_started,
                "finished_at": finished,
                "provider_call_count": 2,
            }
        )

    report = build_report(rows, "fixture-batch")

    assert report["status_counts"] == {"SUCCEEDED": 4}
    assert report["success_by_user"] == {"demo-user-a": 2, "demo-user-b": 2}
    assert report["worker_count_observed"] == 2
    assert report["attempt_count"] == 4
    assert report["admission_deferred_count"] == 0
    assert report["max_completion_prefix_skew"] == 1
    assert report["provider_calls"] == 8


def _wait_for_terminal(client: TestClient, *, user_id: str, run_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/runs/{run_id}", headers={"X-User-ID": user_id})
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"SUCCEEDED", "FAILED", "CANCELLED", "UNCERTAIN"}:
            return body
        time.sleep(0.01)
    pytest.fail(f"run {run_id} did not reach a terminal state")


@pytest.fixture
def tracer_stack():
    snapshots = InMemoryCareerSnapshotStore()
    snapshots.put(
        user_id=USER_A,
        resume_ref=RESUME_REF,
        resume_text="LoRA loss checkpoint evaluation exposure",
    )
    snapshots.put(
        user_id=USER_B,
        resume_ref=RESUME_REF,
        resume_text="Backend API testing and queue reliability experience",
    )
    adapter = SemanticJobExecutionAdapter(snapshots=snapshots)
    return create_tracer_stack(
        allowed_user_ids=(USER_A, USER_B),
        snapshots=snapshots,
        execution_adapter=adapter,
    )


def test_tracer_bullet_runs_existing_semantic_agent_and_returns_sanitized_result(tracer_stack):
    with TestClient(tracer_stack.app) as client:
        created = client.post(
            "/api/v1/runs",
            headers={**_headers(USER_A), "X-Request-ID": "http-request-001"},
            json=_request_payload(),
        )
        assert created.status_code == 202
        run_id = created.json()["run_id"]

        terminal = _wait_for_terminal(client, user_id=USER_A, run_id=run_id)
        assert terminal["status"] == "SUCCEEDED"
        assert terminal["result"]["runtime_mode"] == "agent_api"
        assert terminal["result"]["llm_call_count"] == 5
        assert terminal["result"]["company"] == "Example"
        assert tracer_stack.execution_adapter.execution_count(run_id) == 1
        internal = tracer_stack.repository.get_for_user(user_id=USER_A, run_id=run_id)
        assert internal.created_request_id == "http-request-001"

        events = client.get(
            f"/api/v1/runs/{run_id}/events",
            headers={"X-User-ID": USER_A},
        )
        assert events.status_code == 200
        event_types = [item["event_type"] for item in events.json()["items"]]
        assert event_types[0:2] == ["run.queued", "run.running"]
        assert "run.progress" in event_types
        assert event_types[-1] == "run.succeeded"
        assert "LoRA loss checkpoint" not in events.text

        debug = client.get(
            f"/api/v1/runs/{run_id}/debug",
            headers={"X-User-ID": USER_A},
        )
        assert debug.status_code == 200
        debug_body = debug.json()
        assert debug_body["created_request_id"] == "http-request-001"
        assert debug_body["status"] == "SUCCEEDED"
        assert debug_body["execution_manifest"] is None
        assert debug_body["events"][-1]["event_type"] == "run.succeeded"
        assert "resume_ref" not in debug.text
        assert "LoRA loss checkpoint" not in debug.text


def test_same_user_duplicate_idempotency_key_returns_one_run_and_executes_once(tracer_stack):
    with TestClient(tracer_stack.app) as client:
        first = client.post(
            "/api/v1/runs",
            headers=_headers(USER_A, "duplicate-key"),
            json=_request_payload(),
        )
        second = client.post(
            "/api/v1/runs",
            headers=_headers(USER_A, "duplicate-key"),
            json=_request_payload(),
        )
        assert first.status_code == second.status_code == 202
        assert first.json()["run_id"] == second.json()["run_id"]

        run_id = first.json()["run_id"]
        assert _wait_for_terminal(client, user_id=USER_A, run_id=run_id)["status"] == "SUCCEEDED"
        assert tracer_stack.execution_adapter.execution_count(run_id) == 1


def test_concurrent_duplicate_create_publishes_only_one_message():
    repository = InMemoryRunRepository()
    dispatch_queue = InMemoryDispatchQueue()
    service = RunService(repository=repository, dispatch_queue=dispatch_queue)
    context = RequestContext(user_id=USER_A, request_id="concurrent-request")
    request = CreateRunRequest.model_validate(_request_payload())

    def create_once(_):
        return service.create_run(
            context,
            request,
            idempotency_key="concurrent-key",
        ).run_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        run_ids = list(executor.map(create_once, range(8)))

    assert len(set(run_ids)) == 1
    assert dispatch_queue.depth() == 1


def test_same_key_and_session_are_isolated_between_users(tracer_stack):
    with TestClient(tracer_stack.app) as client:
        first = client.post(
            "/api/v1/runs",
            headers=_headers(USER_A, "shared-client-key"),
            json=_request_payload(session_id="shared-session"),
        )
        second = client.post(
            "/api/v1/runs",
            headers=_headers(USER_B, "shared-client-key"),
            json=_request_payload(session_id="shared-session"),
        )
        assert first.status_code == second.status_code == 202
        first_id = first.json()["run_id"]
        second_id = second.json()["run_id"]
        assert first_id != second_id
        assert _wait_for_terminal(client, user_id=USER_A, run_id=first_id)["status"] == "SUCCEEDED"
        assert _wait_for_terminal(client, user_id=USER_B, run_id=second_id)["status"] == "SUCCEEDED"

        assert client.get(
            f"/api/v1/runs/{second_id}", headers={"X-User-ID": USER_A}
        ).status_code == 404
        assert client.get(
            f"/api/v1/runs/{second_id}/events", headers={"X-User-ID": USER_A}
        ).status_code == 404
        assert client.get(
            f"/api/v1/runs/{second_id}/debug", headers={"X-User-ID": USER_A}
        ).status_code == 404


def test_same_user_reusing_key_with_different_payload_conflicts(tracer_stack):
    with TestClient(tracer_stack.app) as client:
        first = client.post(
            "/api/v1/runs",
            headers=_headers(USER_A, "conflict-key"),
            json=_request_payload(title="First title"),
        )
        conflict = client.post(
            "/api/v1/runs",
            headers=_headers(USER_A, "conflict-key"),
            json=_request_payload(title="Changed title"),
        )
        assert first.status_code == 202
        assert conflict.status_code == 409
        assert conflict.json()["detail"]["code"] == "idempotency_key_payload_mismatch"


def test_identity_and_request_body_cannot_select_another_user(tracer_stack):
    payload = _request_payload()
    payload["user_id"] = USER_B
    with TestClient(tracer_stack.app) as client:
        assert client.post(
            "/api/v1/runs",
            headers=_headers(USER_A),
            json=payload,
        ).status_code == 422
        assert client.post(
            "/api/v1/runs",
            headers={"Idempotency-Key": "missing-user"},
            json=_request_payload(),
        ).status_code == 401
        assert client.post(
            "/api/v1/runs",
            headers=_headers("unknown-user"),
            json=_request_payload(),
        ).status_code == 401
        assert client.post(
            "/api/v1/runs",
            headers={"X-User-ID": USER_A},
            json=_request_payload(),
        ).status_code == 400


class _BlockingAdapter:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self._lock = Lock()
        self.calls = 0

    def execute(self, record, *, progress_callback, claim=None):
        with self._lock:
            self.calls += 1
        progress_callback("blocked_mock_provider")
        self.started.set()
        if not self.release.wait(timeout=3.0):
            raise AssertionError("test did not release blocking adapter")
        return AgentExecutionOutcome(
            result={"runtime_mode": "agent_api", "llm_call_count": 1},
            provider_call_count=1,
        )


def test_post_returns_202_without_waiting_for_worker_completion():
    adapter = _BlockingAdapter()
    stack = create_tracer_stack(
        allowed_user_ids=(USER_A,),
        execution_adapter=adapter,
    )
    with TestClient(stack.app) as client:
        response = client.post(
            "/api/v1/runs",
            headers=_headers(USER_A, "blocking-key"),
            json=_request_payload(),
        )
        assert response.status_code == 202
        assert adapter.started.wait(timeout=1.0)
        run_id = response.json()["run_id"]
        running = client.get(
            f"/api/v1/runs/{run_id}", headers={"X-User-ID": USER_A}
        )
        assert running.json()["status"] == RunStatus.RUNNING.value
        adapter.release.set()
        assert _wait_for_terminal(client, user_id=USER_A, run_id=run_id)["status"] == "SUCCEEDED"
        assert adapter.calls == 1


def test_career_snapshots_are_user_scoped():
    snapshots = InMemoryCareerSnapshotStore()
    snapshots.put(user_id=USER_B, resume_ref=RESUME_REF, resume_text="private-b")
    with pytest.raises(CareerSnapshotNotFoundError, match="career_snapshot_not_found"):
        snapshots.resolve(user_id=USER_A, resume_ref=RESUME_REF)


def test_queue_message_owner_must_match_persisted_run_owner():
    repository = InMemoryRunRepository()
    dispatch_queue = InMemoryDispatchQueue()
    service = RunService(repository=repository, dispatch_queue=dispatch_queue)
    created = service.create_run(
        context=RequestContext(user_id=USER_A, request_id="request-queue"),
        request=CreateRunRequest.model_validate(_request_payload()),
        idempotency_key="queue-owner-key",
    )
    record = repository.get_for_user(user_id=USER_A, run_id=created.run_id)
    with pytest.raises(QueueOwnershipError, match="queue_owner_mismatch"):
        repository.claim(
            RunDispatchMessage(
                message_id="forged-message",
                user_id=USER_B,
                run_id=record.run_id,
                dispatch_generation=record.dispatch_generation,
                enqueued_at=record.created_at,
            )
        )
    assert repository.get_for_user(user_id=USER_A, run_id=record.run_id).status == RunStatus.QUEUED


def test_unsigned_header_identity_adapter_is_forbidden_in_production():
    with pytest.raises(ValueError, match="unsigned_header_identity_is_forbidden_in_production"):
        ConfiguredHeaderIdentityAdapter((USER_A,), production=True)
