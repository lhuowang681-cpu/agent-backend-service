from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Union
from uuid import uuid4

from pydantic import Field, TypeAdapter

from job_agent.agent_runtime.contracts import (
    AgentDecision,
    AgentRunStatus,
    ApprovalDecision,
    AuthorizationResult,
    ToolObservation,
    VerificationResult,
    UserInputResponse,
)
from job_agent.schemas import StrictModel


class EventIdentity(StrictModel):
    event_version: Literal[1] = 1
    event_id: str
    timestamp: str
    session_id: str
    run_id: str
    agent_id: str
    step_index: int = Field(ge=0)


class RunStartedEvent(EventIdentity):
    event_type: Literal["run_started"] = "run_started"
    goal: dict[str, Any]
    budget_profile: str


class ModelDecisionEvent(EventIdentity):
    event_type: Literal["model_decision"] = "model_decision"
    decision: AgentDecision
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class PolicyDecisionEvent(EventIdentity):
    event_type: Literal["policy_decision"] = "policy_decision"
    authorization: AuthorizationResult


class ToolObservedEvent(EventIdentity):
    event_type: Literal["tool_observed"] = "tool_observed"
    observation: ToolObservation


class VerificationEvent(EventIdentity):
    event_type: Literal["verification"] = "verification"
    verification: VerificationResult


class RunFinishedEvent(EventIdentity):
    event_type: Literal["run_finished"] = "run_finished"
    status: AgentRunStatus
    result: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None


class RunResumedEvent(EventIdentity):
    event_type: Literal["run_resumed"] = "run_resumed"
    checkpoint_id: str


class ApprovalResolvedEvent(EventIdentity):
    event_type: Literal["approval_resolved"] = "approval_resolved"
    approval: ApprovalDecision


class UserInputResolvedEvent(EventIdentity):
    event_type: Literal["user_input_resolved"] = "user_input_resolved"
    response: UserInputResponse


AgentEvent = Annotated[
    Union[
        RunStartedEvent,
        ModelDecisionEvent,
        PolicyDecisionEvent,
        ToolObservedEvent,
        VerificationEvent,
        RunFinishedEvent,
        RunResumedEvent,
        ApprovalResolvedEvent,
        UserInputResolvedEvent,
    ],
    Field(discriminator="event_type"),
]
_EVENT_ADAPTER = TypeAdapter(AgentEvent)


def parse_agent_event(value: Any) -> AgentEvent:
    return _EVENT_ADAPTER.validate_python(value)


def event_identity(*, session_id: str, run_id: str, agent_id: str, step_index: int) -> dict[str, Any]:
    return {
        "event_id": f"event-{uuid4().hex}",
        "timestamp": datetime.now(UTC).isoformat(),
        "session_id": session_id,
        "run_id": run_id,
        "agent_id": agent_id,
        "step_index": step_index,
    }
