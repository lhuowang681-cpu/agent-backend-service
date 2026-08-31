from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from threading import Event

import psycopg
from psycopg_pool import PoolTimeout
from redis.exceptions import RedisError

from job_agent.backend_service.execution import (
    AgentExecutionFailedError,
    DomainAgentExecutionAdapter,
    InMemoryCareerSnapshotStore,
    ManifestValidatingExecutionAdapter,
    RoutedExecutionAdapter,
    SemanticJobExecutionAdapter,
    TaskRoutedExecutionAdapter,
)
from job_agent.backend_service.checkpoint_adapter import PostgresSemanticCheckpointStore
from job_agent.backend_service.migration import apply_migrations
from job_agent.backend_service.outbox import OutboxPublisher
from job_agent.backend_service.postgres_repository import PostgresRunRepository
from job_agent.backend_service.redis_streams import RedisStreamDispatchQueue
from job_agent.backend_service.reliable_worker import ReliableWorker
from job_agent.backend_service.provider_admission import RedisProviderAdmissionController
from job_agent.backend_service.failures import FailureClassifier
from job_agent.backend_service.recovery import RecoveryCoordinator
from job_agent.backend_service.settings import BackendSettings
from job_agent.backend_service.metrics import RedisMetricsRecorder
from job_agent.backend_service.execution_manifest import build_execution_manifest
from job_agent.backend_service.application_assistant import build_application_assistant_loop
from job_agent.backend_service.career_snapshot import VersionedCareerSnapshotStore
from job_agent.agent_runtime.contracts import ToolContext
from job_agent.llm.providers.deepseek_compatible import DeepSeekCompatibleProvider
from job_agent.llm.harness import NodePolicy


def main() -> None:
    settings = BackendSettings.from_env()
    apply_migrations(settings.database_url)
    repository = PostgresRunRepository(settings.database_url)
    repository.open()
    repository.ensure_users(settings.allowed_user_ids)
    queue = RedisStreamDispatchQueue(
        settings.redis_url,
        stream_name=settings.stream_name,
        group_name=settings.consumer_group,
    )
    semantic_snapshots = InMemoryCareerSnapshotStore()
    for user_id in settings.allowed_user_ids:
        semantic_snapshots.put(
            user_id=user_id,
            resume_ref="artifact://demo/resume",
            resume_text="Backend API queue lease checkpoint evaluation experience",
        )
    admission = RedisProviderAdmissionController(
        settings.redis_url,
        global_limit=settings.provider_global_limit,
        per_user_limit=settings.provider_per_user_limit,
        permit_ttl_seconds=settings.provider_permit_ttl_seconds,
        circuit_failure_threshold=settings.provider_circuit_threshold,
        circuit_cooldown_seconds=settings.provider_circuit_cooldown_seconds,
    )
    metrics_recorder = RedisMetricsRecorder(
        settings.redis_url,
        prefix=settings.metrics_prefix,
    )

    def record_provider_trace(
        provider: str,
        duration: float,
        succeeded: bool,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> None:
        metrics_recorder.record_provider_call(
            provider=provider,
            duration_seconds=duration,
            succeeded=succeeded,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    mock_adapter = SemanticJobExecutionAdapter(
        snapshots=semantic_snapshots,
        checkpoint_store_factory=lambda _record, claim: PostgresSemanticCheckpointStore(
            repository=repository,
            claim=claim,
        ),
        provider_trace_callback=record_provider_trace,
        provider_admission=admission,
    )

    def deepseek_provider_factory(_record, _resume_path):
        api_key = os.environ.get(settings.deepseek_api_key_env, "").strip()
        if not api_key:
            raise AgentExecutionFailedError("provider_credential_missing")
        return DeepSeekCompatibleProvider(
            api_key=api_key,
            model=settings.deepseek_model,
            base_url=settings.deepseek_base_url,
            timeout_s=settings.deepseek_timeout_seconds,
        )

    deepseek_adapter = SemanticJobExecutionAdapter(
        snapshots=semantic_snapshots,
        checkpoint_store_factory=lambda _record, claim: PostgresSemanticCheckpointStore(
            repository=repository,
            claim=claim,
        ),
        provider_trace_callback=record_provider_trace,
        provider_factory=deepseek_provider_factory,
        provider_admission=admission,
        policy=NodePolicy(
            timeout_s=settings.deepseek_timeout_seconds,
            max_retries=0,
            max_schema_repairs=0,
            temperature=0.0,
            max_output_tokens=1024,
            allow_rule_fallback=False,
        ),
    )
    career_snapshots = VersionedCareerSnapshotStore(repository)
    workspace_root = Path(settings.backend_workspace_root).resolve()
    application_adapter = DomainAgentExecutionAdapter(
        repository=repository,
        loop_factory=lambda record, checkpoint_store, claim: (
            build_application_assistant_loop(
                record=record,
                snapshot=career_snapshots.resolve(
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
    worker = ReliableWorker(
        repository=repository,
        dispatch_queue=queue,
        execution_adapter=ManifestValidatingExecutionAdapter(
            adapter=TaskRoutedExecutionAdapter(
                {
                    "semantic_job_flow": RoutedExecutionAdapter(
                        {
                            "mock": mock_adapter,
                            "deepseek_flash": deepseek_adapter,
                        }
                    ),
                    "application_assistant_flow": application_adapter,
                }
            ),
            expected_manifest_factory=lambda record: build_execution_manifest(
                record.request,
                provider_models={
                    "mock": "fixture-model",
                    "deepseek_flash": settings.deepseek_model,
                },
            ),
        ),
        worker_id=settings.worker_id,
        lease_seconds=settings.lease_seconds,
        failure_classifier=FailureClassifier(max_attempts=settings.max_attempts),
        metrics_recorder=metrics_recorder,
    )
    publisher = OutboxPublisher(repository=repository, dispatch_queue=queue)
    recovery = RecoveryCoordinator(
        repository=repository,
        dispatch_queue=queue,
        publisher=publisher,
        max_attempts=settings.max_attempts,
        dispatch_reconcile_grace_seconds=(
            settings.dispatch_reconcile_grace_seconds
        ),
    )
    stop = Event()

    def request_stop(_signum, _frame) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    last_recovery = 0.0
    try:
        while not stop.is_set():
            try:
                queue.ensure_group()
                publisher.publish_once()
                worker.receive_and_process(block_ms=250)
                now = time.monotonic()
                if now - last_recovery >= settings.recovery_interval_seconds:
                    recovery.run_once()
                    worker.reclaim_and_process(
                        min_idle_ms=max(1, int(settings.lease_seconds * 1000))
                    )
                    last_recovery = now
            except RedisError:
                stop.wait(settings.publisher_interval_seconds)
            except (psycopg.Error, PoolTimeout):
                stop.wait(settings.publisher_interval_seconds)
    finally:
        metrics_recorder.close()
        admission.close()
        queue.close()
        repository.close()


if __name__ == "__main__":
    main()
