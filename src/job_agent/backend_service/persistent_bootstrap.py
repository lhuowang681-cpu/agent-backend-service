from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI

from job_agent.backend_service.api import ConfiguredHeaderIdentityAdapter, create_app
from job_agent.backend_service.migration import apply_migrations
from job_agent.backend_service.postgres_repository import PostgresRunRepository
from job_agent.backend_service.service import RunService
from job_agent.backend_service.execution_manifest import build_execution_manifest
from job_agent.backend_service.career_snapshot import VersionedCareerSnapshotStore
from job_agent.backend_service.settings import BackendSettings
from job_agent.backend_service.metrics import (
    PrometheusMetricsService,
    RedisMetricsRecorder,
)


@dataclass(frozen=True)
class PersistentApiStack:
    app: FastAPI
    repository: PostgresRunRepository
    service: RunService


class _RepositoryLifecycle:
    def __init__(
        self,
        repository: PostgresRunRepository,
        metrics_recorder: RedisMetricsRecorder,
    ) -> None:
        self.repository = repository
        self.metrics_recorder = metrics_recorder

    def start(self) -> None:
        return None

    def stop(self) -> None:
        self.metrics_recorder.close()
        self.repository.close()


def create_persistent_api_stack(settings: BackendSettings) -> PersistentApiStack:
    apply_migrations(settings.database_url)
    repository = PostgresRunRepository(settings.database_url)
    repository.open()
    repository.ensure_users(settings.allowed_user_ids)
    career_snapshots = VersionedCareerSnapshotStore(repository)
    service = RunService(
        repository=repository,
        dispatch_queue=None,
        max_queued_global=settings.max_queued_runs_global,
        max_queued_per_user=settings.max_queued_runs_per_user,
        execution_manifest_factory=lambda request: build_execution_manifest(
            request,
            provider_models={
                "mock": "fixture-model",
                "deepseek_flash": settings.deepseek_model,
            },
        ),
        career_snapshot_store=career_snapshots,
    )
    metrics_recorder = RedisMetricsRecorder(
        settings.redis_url,
        prefix=settings.metrics_prefix,
    )
    metrics_service = PrometheusMetricsService(
        repository=repository,
        recorder=metrics_recorder,
        redis_url=settings.redis_url,
        stream_name=settings.stream_name,
        consumer_group=settings.consumer_group,
    )
    app = create_app(
        service=service,
        worker=_RepositoryLifecycle(repository, metrics_recorder),
        identity_adapter=ConfiguredHeaderIdentityAdapter(settings.allowed_user_ids),
        metrics_recorder=metrics_recorder,
        metrics_service=metrics_service,
    )
    return PersistentApiStack(app=app, repository=repository, service=service)
