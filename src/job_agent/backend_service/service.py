from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4
from typing import Callable

from job_agent.backend_service.contracts import (
    ApplicationAssistantInput,
    ApprovalDecisionRequest,
    CreateRunRequest,
    CreateRunResponse,
    EventPage,
    RequestContext,
    RunDispatchMessage,
    RunDebugView,
    RunView,
    RunMutationResponse,
    ExecutionManifest,
)
from job_agent.backend_service.career_snapshot import VersionedCareerSnapshotStore
from job_agent.backend_service.failures import FailureClassifier
from job_agent.backend_service.queueing import InMemoryDispatchQueue
from job_agent.backend_service.repository import InMemoryRunRepository


class IdempotencyKeyRequiredError(ValueError):
    pass


class RunService:
    def __init__(
        self,
        *,
        repository,
        dispatch_queue: InMemoryDispatchQueue | None,
        max_queued_global: int | None = None,
        max_queued_per_user: int | None = None,
        execution_manifest_factory: Callable[
            [CreateRunRequest], ExecutionManifest
        ]
        | None = None,
        career_snapshot_store: VersionedCareerSnapshotStore | None = None,
    ) -> None:
        if max_queued_global is not None and max_queued_global < 1:
            raise ValueError("max_queued_global_must_be_positive")
        if max_queued_per_user is not None and max_queued_per_user < 1:
            raise ValueError("max_queued_per_user_must_be_positive")
        self.repository = repository
        self.dispatch_queue = dispatch_queue
        self.max_queued_global = max_queued_global
        self.max_queued_per_user = max_queued_per_user
        self.execution_manifest_factory = execution_manifest_factory
        self.career_snapshot_store = career_snapshot_store

    def create_run(
        self,
        context: RequestContext,
        request: CreateRunRequest,
        *,
        idempotency_key: str | None,
    ) -> CreateRunResponse:
        key = (idempotency_key or "").strip()
        if not key or len(key) > 255:
            raise IdempotencyKeyRequiredError("idempotency_key_required")
        if request.task_type == "application_assistant_flow":
            if not isinstance(request.input, ApplicationAssistantInput):
                raise ValueError("task_input_contract_mismatch")
            if self.career_snapshot_store is None:
                raise ValueError("career_snapshot_store_not_configured")
            self.career_snapshot_store.require_owned(
                user_id=context.user_id,
                snapshot_revision=request.input.career_snapshot_revision,
            )
        request_hash = self._request_hash(request)
        execution_manifest = (
            self.execution_manifest_factory(request)
            if self.execution_manifest_factory is not None
            else None
        )
        record, created = self.repository.create_or_get(
            user_id=context.user_id,
            request_id=context.request_id,
            idempotency_key=key,
            request_hash=request_hash,
            request=request,
            execution_manifest=execution_manifest,
            max_queued_global=self.max_queued_global,
            max_queued_per_user=self.max_queued_per_user,
        )
        if created and self.dispatch_queue is not None:
            self.dispatch_queue.publish(
                RunDispatchMessage(
                    message_id=str(uuid4()),
                    user_id=record.user_id,
                    run_id=record.run_id,
                    dispatch_generation=record.dispatch_generation,
                    enqueued_at=datetime.now(UTC),
                )
            )
        return CreateRunResponse(run_id=record.run_id, status=record.status)

    def get_run(self, context: RequestContext, run_id: str) -> RunView:
        return RunView.from_record(
            self.repository.get_for_user(user_id=context.user_id, run_id=run_id)
        )

    def list_events(
        self,
        context: RequestContext,
        run_id: str,
        *,
        after: int,
        limit: int,
    ) -> EventPage:
        return self.repository.list_events(
            user_id=context.user_id,
            run_id=run_id,
            after=after,
            limit=limit,
        )

    def get_debug_view(self, context: RequestContext, run_id: str) -> RunDebugView:
        return self.repository.get_debug_view(
            user_id=context.user_id,
            run_id=run_id,
        )

    def decide_approval(
        self,
        context: RequestContext,
        run_id: str,
        request: ApprovalDecisionRequest,
        *,
        approved: bool,
    ) -> RunMutationResponse:
        record = self.repository.decide_approval_and_requeue(
            user_id=context.user_id,
            run_id=run_id,
            request=request,
            approved=approved,
        )
        return RunMutationResponse(
            run_id=record.run_id,
            status=record.status,
            status_version=record.status_version,
        )

    def resume_run(
        self,
        context: RequestContext,
        run_id: str,
        *,
        expected_status_version: int,
    ) -> RunMutationResponse:
        record = self.repository.resume_failed(
            user_id=context.user_id,
            run_id=run_id,
            expected_status_version=expected_status_version,
            actor_user_id=context.user_id,
            resumable_error_codes=FailureClassifier.EXPLICITLY_RESUMABLE,
        )
        return RunMutationResponse(
            run_id=record.run_id,
            status=record.status,
            status_version=record.status_version,
        )

    @staticmethod
    def _request_hash(request: CreateRunRequest) -> str:
        encoded = json.dumps(
            request.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
