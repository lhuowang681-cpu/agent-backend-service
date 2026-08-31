from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from job_agent.schemas import RawJob, StrictModel
from job_agent.agent_runtime.contracts import AgentAction, ApprovalRequest


class RunStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNCERTAIN = "UNCERTAIN"


class RequestContext(StrictModel):
    user_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)


class SemanticJobInput(StrictModel):
    selected_job: RawJob
    resume_ref: str = Field(min_length=1, max_length=512)


class ApplicationAssistantInput(StrictModel):
    selected_job: RawJob
    career_snapshot_revision: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^snapshot-[a-f0-9]{16,64}$",
    )
    draft_channel: Literal["email", "form"] = "email"
    draft_target: str = Field(min_length=1, max_length=512)


class ExecutionManifest(StrictModel):
    manifest_version: Literal[1] = 1
    provider_profile: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=256)
    skill_version: str = Field(min_length=1, max_length=128)
    prompt_version: str = Field(min_length=1, max_length=128)
    tool_registry_version: str = Field(min_length=1, max_length=128)
    policy_version: str = Field(min_length=1, max_length=128)
    career_snapshot_revision: str | None = Field(default=None, max_length=128)
    checkpoint_schema_version: str = Field(min_length=1, max_length=128)


class CreateRunRequest(StrictModel):
    task_type: Literal[
        "semantic_job_flow",
        "application_assistant_flow",
    ] = "semantic_job_flow"
    session_id: str = Field(min_length=1, max_length=256)
    input: SemanticJobInput | ApplicationAssistantInput
    provider_profile: Literal["mock", "deepseek_flash"] = "mock"
    budget_profile: Literal["quick"] = "quick"

    @model_validator(mode="after")
    def validate_task_input_contract(self):
        expected = {
            "semantic_job_flow": SemanticJobInput,
            "application_assistant_flow": ApplicationAssistantInput,
        }[self.task_type]
        if not isinstance(self.input, expected):
            raise ValueError("task_input_contract_mismatch")
        if (
            self.task_type == "application_assistant_flow"
            and self.provider_profile != "mock"
        ):
            raise ValueError("application_assistant_provider_not_supported")
        return self


class RunRecord(StrictModel):
    run_id: str
    user_id: str
    session_id: str
    task_type: str
    status: RunStatus
    status_version: int = Field(ge=0)
    dispatch_generation: int = Field(ge=1)
    created_request_id: str
    idempotency_key: str
    request_hash: str
    request: CreateRunRequest
    execution_manifest: ExecutionManifest | None = None
    current_stage: str | None = None
    result: dict[str, object] | None = None
    error_code: str | None = None
    retry_count: int = Field(default=0, ge=0)
    next_attempt_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None


class CreateRunResponse(StrictModel):
    run_id: str
    status: RunStatus


class RunView(StrictModel):
    run_id: str
    session_id: str
    task_type: str
    status: RunStatus
    status_version: int
    current_stage: str | None = None
    result: dict[str, object] | None = None
    error_code: str | None = None
    retry_count: int = 0
    next_attempt_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None

    @classmethod
    def from_record(cls, record: RunRecord) -> "RunView":
        return cls.model_validate(
            record.model_dump(
                exclude={
                    "user_id",
                    "dispatch_generation",
                    "created_request_id",
                    "idempotency_key",
                    "request_hash",
                    "request",
                    "execution_manifest",
                }
            )
        )


class RunEventView(StrictModel):
    sequence: int = Field(ge=1)
    event_type: str
    payload: dict[str, object] = Field(default_factory=dict)
    created_at: datetime


class EventPage(StrictModel):
    items: list[RunEventView]
    next_after: int


class DebugAttemptView(StrictModel):
    attempt_id: str
    attempt_no: int = Field(ge=1)
    status: str
    execution_phase: str
    provider_call_count: int = Field(ge=0)
    error_code: str | None = None
    recovery_kind: str | None = None
    started_at: datetime
    finished_at: datetime | None = None


class DebugApprovalView(StrictModel):
    approval_id: str
    request_id: str
    action_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    tool_name: str
    decision: str | None = None
    version: int = Field(ge=0)
    created_at: datetime
    resolved_at: datetime | None = None


class DebugCheckpointView(StrictModel):
    checkpoint_kind: str
    checkpoint_id: str
    checkpoint_version: int = Field(ge=1)
    updated_at: datetime
    expires_at: datetime | None = None


class DebugToolOperationView(StrictModel):
    operation_id: str
    attempt_id: str | None = None
    tool_name: str
    action_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    state: str
    error_code: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime


class DebugEventView(StrictModel):
    sequence: int = Field(ge=1)
    attempt_id: str | None = None
    event_type: str
    payload: dict[str, object] = Field(default_factory=dict)
    created_at: datetime


class RunDebugView(StrictModel):
    run_id: str
    session_id: str
    created_request_id: str
    task_type: str
    status: RunStatus
    status_version: int = Field(ge=0)
    current_stage: str | None = None
    error_code: str | None = None
    retry_count: int = Field(ge=0)
    execution_manifest: ExecutionManifest | None = None
    attempts: list[DebugAttemptView] = Field(default_factory=list)
    approvals: list[DebugApprovalView] = Field(default_factory=list)
    checkpoint: DebugCheckpointView | None = None
    tool_operations: list[DebugToolOperationView] = Field(default_factory=list)
    events: list[DebugEventView] = Field(default_factory=list)
    events_truncated: bool = False


class RunDispatchMessage(StrictModel):
    message_id: str
    user_id: str
    run_id: str
    dispatch_generation: int = Field(ge=1)
    enqueued_at: datetime


class ClaimedRun(StrictModel):
    run: RunRecord
    attempt_id: str
    attempt_no: int = Field(ge=1)
    lease_token: str
    lease_expires_at: datetime


class StreamDelivery(StrictModel):
    stream_message_id: str
    message: RunDispatchMessage


class AgentExecutionOutcome(StrictModel):
    result: dict[str, object]
    provider_call_count: int = Field(ge=0)


class AgentApprovalPause(StrictModel):
    approval_request: ApprovalRequest
    pending_action: AgentAction
    provider_call_count: int = Field(default=0, ge=0)


class ApprovalDecisionRequest(StrictModel):
    approval_id: str = Field(min_length=1, max_length=128)
    action_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    expected_status_version: int = Field(ge=0)
    reason: str = Field(default="", max_length=1000)


class ResumeRunRequest(StrictModel):
    expected_status_version: int = Field(ge=0)


class RunMutationResponse(StrictModel):
    run_id: str
    status: RunStatus
    status_version: int
