from __future__ import annotations

from enum import Enum
from time import monotonic
from typing import Annotated, Any, Callable, Literal, Union

from pydantic import Field, TypeAdapter, model_validator

from job_agent.schemas import StrictModel


class ToolEffect(str, Enum):
    READ_ONLY = "read_only"
    LOCAL_DRAFT_WRITE = "local_draft_write"
    LOCAL_STATE_MUTATION = "local_state_mutation"
    EXTERNAL_DRAFT = "external_draft"
    EXTERNAL_REVERSIBLE_WRITE = "external_reversible_write"
    EXTERNAL_IRREVERSIBLE_WRITE = "external_irreversible_write"
    SENSITIVE_READ = "sensitive_read"

    @property
    def is_external(self) -> bool:
        return self in {
            ToolEffect.EXTERNAL_DRAFT,
            ToolEffect.EXTERNAL_REVERSIBLE_WRITE,
            ToolEffect.EXTERNAL_IRREVERSIBLE_WRITE,
        }


class RuntimeToolSpec(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.-]*$")
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    effect: ToolEffect
    timeout_s: float = Field(default=30.0, gt=0)
    timeout_mode: Literal["cooperative"] = "cooperative"
    idempotent: bool = True
    requires_approval: bool = False
    sandbox_only: bool = False

    @model_validator(mode="after")
    def validate_effect_contract(self):
        if self.effect == ToolEffect.SENSITIVE_READ and not self.requires_approval:
            raise ValueError("sensitive-read tools must require approval")
        if self.effect.is_external and not self.requires_approval:
            raise ValueError("external tools must require approval")
        return self


class AgentAction(StrictModel):
    kind: Literal["action"] = "action"
    action_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    expected_observation: str = Field(min_length=1, max_length=1000)
    progress_claim: str = Field(min_length=1, max_length=1000)


class AgentFinish(StrictModel):
    kind: Literal["finish"] = "finish"
    result: dict[str, Any]
    completion_evidence: list[str] = Field(min_length=1)
    unresolved_items: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class UserInputRequest(StrictModel):
    request_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    response_schema: dict[str, Any] = Field(default_factory=dict)
    context_summary: str = ""
    allow_cancel: bool = True


class UserInputResponse(StrictModel):
    request_id: str = Field(min_length=1)
    value: dict[str, Any] = Field(default_factory=dict)
    cancelled: bool = False


class AgentInputRequest(StrictModel):
    kind: Literal["input_request"] = "input_request"
    request: UserInputRequest
    progress_claim: str = Field(min_length=1, max_length=1000)


AgentDecision = Annotated[
    Union[AgentAction, AgentFinish, AgentInputRequest],
    Field(discriminator="kind"),
]
_DECISION_ADAPTER = TypeAdapter(AgentDecision)


def parse_agent_decision(value: Any) -> AgentDecision:
    return _DECISION_ADAPTER.validate_python(value)


class AgentDecisionEnvelope(StrictModel):
    decision: AgentDecision


class AgentStepRecord(StrictModel):
    step_index: int = Field(ge=0)
    action: AgentAction
    observation: "ToolObservation"


class UserInputRecord(StrictModel):
    step_index: int = Field(ge=0)
    request: UserInputRequest
    response: UserInputResponse


class AgentModelContext(StrictModel):
    goal: dict[str, Any]
    steps: list[AgentStepRecord] = Field(default_factory=list)
    user_inputs: list[UserInputRecord] = Field(default_factory=list)
    verifier_feedback: list[str] = Field(default_factory=list)
    allowed_tools: list[RuntimeToolSpec] = Field(default_factory=list)
    budget_usage: dict[str, Any] = Field(default_factory=dict)


class AgentModelResponse(StrictModel):
    decision: AgentDecision
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0.0)
    provider: str = "unknown"
    model: str = "unknown"


class ToolObservationStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    RETRYABLE_ERROR = "retryable_error"
    FATAL_ERROR = "fatal_error"


class ToolObservation(StrictModel):
    action_id: str
    tool_name: str
    status: ToolObservationStatus
    data: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    duration_ms: int = Field(ge=0)
    error_code: str | None = None

    @model_validator(mode="after")
    def validate_error_contract(self):
        is_error = self.status in {
            ToolObservationStatus.RETRYABLE_ERROR,
            ToolObservationStatus.FATAL_ERROR,
        }
        if is_error and not self.error_code:
            raise ValueError("error observations require an error_code")
        if not is_error and self.error_code is not None:
            raise ValueError("successful observations cannot include an error_code")
        return self


class ToolContext(StrictModel):
    session_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    workspace_root: str
    sandbox_root: str | None = None
    deadline_monotonic: float | None = None
    user_inputs: dict[str, UserInputResponse] = Field(default_factory=dict)

    def remaining_seconds(self, *, clock: Callable[[], float] = monotonic) -> float | None:
        if self.deadline_monotonic is None:
            return None
        return max(self.deadline_monotonic - clock(), 0.0)

    def ensure_active(self, *, clock: Callable[[], float] = monotonic) -> None:
        remaining = self.remaining_seconds(clock=clock)
        if remaining is not None and remaining <= 0:
            from job_agent.agent_runtime.failures import ToolTimeoutError

            raise ToolTimeoutError()


class ApprovalRequest(StrictModel):
    request_id: str
    action_digest: str
    tool_name: str
    effect: ToolEffect
    target_summary: str
    arguments_preview: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    run_id: str | None = None

    @model_validator(mode="after")
    def validate_scope(self):
        if (self.session_id is None) != (self.run_id is None):
            raise ValueError("approval scope requires both session_id and run_id")
        return self


class ApprovalDecision(StrictModel):
    request_id: str
    action_digest: str
    approved: bool
    decided_by: Literal["user"] = "user"
    reason: str = ""
    session_id: str | None = None
    run_id: str | None = None

    @model_validator(mode="after")
    def validate_scope(self):
        if (self.session_id is None) != (self.run_id is None):
            raise ValueError("approval scope requires both session_id and run_id")
        return self


class AuthorizationResult(StrictModel):
    action_id: str
    action_digest: str
    tool_name: str
    effect: ToolEffect
    allowed: bool
    requires_approval: bool
    reason_code: str
    approval_request: ApprovalRequest | None = None

    @model_validator(mode="after")
    def validate_authorization_contract(self):
        if self.requires_approval and self.approval_request is None:
            raise ValueError("approval-required authorization needs an approval request")
        if self.allowed and self.requires_approval:
            raise ValueError("an unresolved approval request cannot already be allowed")
        return self


class VerificationResult(StrictModel):
    passed: bool
    reason_code: str
    feedback: str
    evidence_refs: list[str] = Field(default_factory=list)


class AgentRunStatus(str, Enum):
    RUNNING = "RUNNING"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    CANCELLED = "CANCELLED"


class AgentRunState(StrictModel):
    session_id: str
    run_id: str
    agent_id: str
    status: AgentRunStatus = AgentRunStatus.RUNNING
    goal: dict[str, Any]
    step_index: int = Field(default=0, ge=0)
    budget_profile: str
    context_refs: list[str] = Field(default_factory=list)
    last_checkpoint_id: str | None = None


class AgentRunResult(StrictModel):
    state: AgentRunState
    result: dict[str, Any] = Field(default_factory=dict)
    unresolved_items: list[str] = Field(default_factory=list)
    error_code: str | None = None
    pending_action: AgentAction | None = None
    pending_approval: ApprovalRequest | None = None
    pending_user_input: UserInputRequest | None = None
