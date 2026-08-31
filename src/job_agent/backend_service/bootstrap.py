from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI

from job_agent.backend_service.api import ConfiguredHeaderIdentityAdapter, create_app
from job_agent.backend_service.execution import (
    AgentExecutionAdapter,
    InMemoryCareerSnapshotStore,
    SemanticJobExecutionAdapter,
)
from job_agent.backend_service.queueing import InMemoryDispatchQueue
from job_agent.backend_service.repository import InMemoryRunRepository
from job_agent.backend_service.service import RunService
from job_agent.backend_service.worker import WorkerEngine


@dataclass(frozen=True)
class TracerStack:
    app: FastAPI
    repository: InMemoryRunRepository
    dispatch_queue: InMemoryDispatchQueue
    snapshots: InMemoryCareerSnapshotStore
    execution_adapter: AgentExecutionAdapter
    service: RunService
    worker: WorkerEngine


def create_tracer_stack(
    *,
    allowed_user_ids: tuple[str, ...],
    snapshots: InMemoryCareerSnapshotStore | None = None,
    execution_adapter: AgentExecutionAdapter | None = None,
) -> TracerStack:
    snapshot_store = snapshots or InMemoryCareerSnapshotStore()
    adapter = execution_adapter or SemanticJobExecutionAdapter(snapshots=snapshot_store)
    repository = InMemoryRunRepository()
    dispatch_queue = InMemoryDispatchQueue()
    service = RunService(repository=repository, dispatch_queue=dispatch_queue)
    worker = WorkerEngine(
        repository=repository,
        dispatch_queue=dispatch_queue,
        execution_adapter=adapter,
    )
    app = create_app(
        service=service,
        worker=worker,
        identity_adapter=ConfiguredHeaderIdentityAdapter(allowed_user_ids),
    )
    return TracerStack(
        app=app,
        repository=repository,
        dispatch_queue=dispatch_queue,
        snapshots=snapshot_store,
        execution_adapter=adapter,
        service=service,
        worker=worker,
    )
