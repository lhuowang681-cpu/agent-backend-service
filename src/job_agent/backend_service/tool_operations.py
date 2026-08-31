from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Callable, Protocol

from pydantic import Field

from job_agent.agent_runtime.contracts import (
    AgentAction,
    AuthorizationResult,
    ToolContext,
    ToolObservation,
    ToolObservationStatus,
)
from job_agent.agent_runtime.policy import compute_action_digest
from job_agent.agent_runtime.tool_executor import ToolExecutor
from job_agent.agent_runtime.tool_registry import ToolRegistry, ToolRegistryError
from job_agent.backend_service.contracts import ClaimedRun
from job_agent.schemas import StrictModel


class ToolOperationState(str, Enum):
    PREPARED = "PREPARED"
    INFLIGHT = "INFLIGHT"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNCERTAIN = "UNCERTAIN"


class ToolOperationRecord(StrictModel):
    operation_id: str
    user_id: str
    run_id: str
    attempt_id: str | None = None
    tool_name: str
    action_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    request_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    state: ToolOperationState
    receipt: dict[str, object] | None = None
    error_code: str | None = None


class ToolOperationLease(StrictModel):
    operation: ToolOperationRecord
    should_execute: bool


class ToolOperationConflictError(RuntimeError):
    pass


class ToolOperationOutcomeUncertainError(RuntimeError):
    pass


class ToolOperationRepository(Protocol):
    def begin_tool_operation(
        self,
        claim: ClaimedRun,
        *,
        operation_id: str,
        tool_name: str,
        action_digest: str,
        request_hash: str,
    ) -> ToolOperationLease: ...

    def complete_tool_operation(
        self,
        claim: ClaimedRun,
        *,
        operation_id: str,
        state: ToolOperationState,
        receipt: dict[str, object],
        error_code: str | None,
    ) -> ToolOperationRecord: ...


def compute_tool_operation_id(*, run_id: str, action: AgentAction) -> str:
    digest = compute_action_digest(action)
    return "operation-" + hashlib.sha256(
        f"{run_id}\0{action.tool_name}\0{digest}".encode("utf-8")
    ).hexdigest()[:32]


class LedgeredToolExecutor(ToolExecutor):
    """Persist non-idempotent operation intent and receipt around ToolExecutor."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        repository: ToolOperationRepository,
        claim: ClaimedRun,
        failpoint: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(registry)
        self.repository = repository
        self.claim = claim
        self.failpoint = failpoint

    def execute(
        self,
        action: AgentAction,
        authorization: AuthorizationResult,
        context: ToolContext,
    ) -> ToolObservation:
        try:
            tool = self.registry.get(action.tool_name)
        except ToolRegistryError:
            return super().execute(action, authorization, context)
        if not authorization.allowed or tool.spec.idempotent:
            return super().execute(action, authorization, context)

        digest = compute_action_digest(action)
        request_hash = hashlib.sha256(
            json.dumps(
                action.tool_arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        operation_id = compute_tool_operation_id(
            run_id=context.run_id,
            action=action,
        )
        lease = self.repository.begin_tool_operation(
            self.claim,
            operation_id=operation_id,
            tool_name=action.tool_name,
            action_digest=digest,
            request_hash=request_hash,
        )
        operation = lease.operation
        if operation.state == ToolOperationState.SUCCEEDED:
            assert operation.receipt is not None
            return ToolObservation.model_validate(operation.receipt)
        if operation.state == ToolOperationState.FAILED:
            assert operation.receipt is not None
            return ToolObservation.model_validate(operation.receipt)
        if operation.state == ToolOperationState.UNCERTAIN or not lease.should_execute:
            raise ToolOperationOutcomeUncertainError(
                "non_idempotent_execution_uncertain"
            )

        observation = super().execute(action, authorization, context)
        if self.failpoint is not None:
            self.failpoint("after_tool_before_receipt")
        terminal_state = (
            ToolOperationState.SUCCEEDED
            if observation.status in {ToolObservationStatus.OK, ToolObservationStatus.EMPTY}
            else ToolOperationState.FAILED
        )
        self.repository.complete_tool_operation(
            self.claim,
            operation_id=operation_id,
            state=terminal_state,
            receipt=observation.model_dump(mode="json"),
            error_code=observation.error_code,
        )
        return observation
