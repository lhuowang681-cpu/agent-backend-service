from __future__ import annotations

from pydantic import Field

from job_agent.agent_runtime.budgets import AgentBudget, BudgetManager, BudgetProfile
from job_agent.agent_runtime.contracts import (
    AgentAction,
    AgentFinish,
    AgentRunStatus,
    ApprovalDecision,
    ToolContext,
    ToolEffect,
    VerificationResult,
)
from job_agent.agent_runtime.loop import AgentLoop, ScriptedAgentModel
from job_agent.agent_runtime.policy import PolicyContext, PolicyEngine, compute_action_digest
from job_agent.agent_runtime.tool_registry import ToolRegistry
from job_agent.agent_runtime.trajectory import InMemoryTrajectory
from job_agent.schemas import StrictModel


class EchoInput(StrictModel):
    value: str = Field(min_length=1)


class EchoOutput(StrictModel):
    value: str


class EvidenceVerifier:
    def verify(self, finish, *, goal, steps):
        if not steps:
            return VerificationResult(
                passed=False,
                reason_code="missing_observation",
                feedback="Call an evidence tool before finishing.",
            )
        return VerificationResult(
            passed=True,
            reason_code="complete",
            feedback="complete",
            evidence_refs=[f"step:{steps[0].step_index}"],
        )


def _action(action_id="a-1", value="hello") -> AgentAction:
    return AgentAction(
        action_id=action_id,
        tool_name="echo.read",
        tool_arguments={"value": value},
        expected_observation="echo",
        progress_claim="read evidence",
    )


def _finish() -> AgentFinish:
    return AgentFinish(
        result={"answer": "done"},
        completion_evidence=["step:2"],
        confidence=0.9,
    )


def _loop(
    model,
    *,
    effect=ToolEffect.READ_ONLY,
    trajectory=None,
    max_same_verifier_reason=None,
) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(
        name="echo.read",
        description="Echo approved test data",
        input_model=EchoInput,
        output_model=EchoOutput,
        handler=lambda value, context: EchoOutput(value=value.value),
        effect=effect,
        sandbox_only=effect.is_external,
    )
    return AgentLoop(
        agent_id="test-agent",
        model=model,
        registry=registry,
        policy=PolicyEngine(),
        policy_context=PolicyContext(
            skill_allowed_tools=("echo.read",),
            agent_allowed_tools=("echo.read",),
            runtime_allowed_tools=("echo.read",),
        ),
        budget=BudgetManager(
            AgentBudget.for_profile(
                BudgetProfile.EVAL,
                max_model_calls=8,
                max_tool_calls=8,
                max_active_seconds=60,
            )
        ),
        verifier=EvidenceVerifier(),
        trajectory=trajectory,
        max_same_verifier_reason=max_same_verifier_reason,
    )


def _tool_context() -> ToolContext:
    return ToolContext(
        session_id="session-1",
        run_id="run-1",
        agent_id="test-agent",
        workspace_root="fixtures/workspace",
        sandbox_root="fixtures/workspace/sandbox",
    )


def test_loop_uses_verifier_feedback_and_tool_observation() -> None:
    model = ScriptedAgentModel([_finish(), _action(), _finish()])
    trajectory = InMemoryTrajectory()
    result = _loop(model, trajectory=trajectory).run(
        session_id="session-1",
        run_id="run-1",
        goal={"answer": "grounded"},
        tool_context=_tool_context(),
    )

    assert result.state.status == AgentRunStatus.COMPLETED
    assert result.result == {"answer": "done"}
    assert model.calls[1].verifier_feedback == ["Call an evidence tool before finishing."]
    assert model.calls[2].steps[0].observation.data == {"value": "hello"}
    assert [event.event_type for event in trajectory.events] == [
        "run_started",
        "model_decision",
        "verification",
        "model_decision",
        "policy_decision",
        "tool_observed",
        "model_decision",
        "verification",
        "run_finished",
    ]


def test_loop_stops_after_repeated_identical_verifier_failure() -> None:
    model = ScriptedAgentModel([_finish(), _finish(), _finish()])
    trajectory = InMemoryTrajectory()

    result = _loop(
        model,
        trajectory=trajectory,
        max_same_verifier_reason=2,
    ).run(
        session_id="session-1",
        run_id="run-1",
        goal={"answer": "grounded"},
        tool_context=_tool_context(),
    )

    assert result.state.status == AgentRunStatus.FAILED
    assert result.error_code == "repeated_verifier_failure"
    assert len(model.calls) == 2
    assert [event.event_type for event in trajectory.events].count("verification") == 2


def test_loop_returns_waiting_before_mutation_handler_runs() -> None:
    calls = []
    registry = ToolRegistry()
    registry.register(
        name="echo.read",
        description="Mutate approved test state",
        input_model=EchoInput,
        output_model=EchoOutput,
        handler=lambda value, context: calls.append(value.value) or EchoOutput(value=value.value),
        effect=ToolEffect.LOCAL_STATE_MUTATION,
    )
    action = _action()
    model = ScriptedAgentModel([action])
    loop = AgentLoop(
        agent_id="test-agent",
        model=model,
        registry=registry,
        policy=PolicyEngine(),
        policy_context=PolicyContext(
            skill_allowed_tools=("echo.read",),
            agent_allowed_tools=("echo.read",),
            runtime_allowed_tools=("echo.read",),
        ),
        budget=BudgetManager(AgentBudget.for_profile(BudgetProfile.QUICK)),
        verifier=EvidenceVerifier(),
    )
    result = loop.run(
        session_id="session-1",
        run_id="run-1",
        goal={},
        tool_context=_tool_context(),
    )
    assert result.state.status == AgentRunStatus.WAITING_FOR_USER
    assert result.pending_approval is not None
    assert calls == []


def test_loop_executes_approved_action_with_bound_digest() -> None:
    action = _action()
    pending_model = ScriptedAgentModel([action])
    pending = _loop(pending_model, effect=ToolEffect.LOCAL_STATE_MUTATION).run(
        session_id="session-1",
        run_id="run-1",
        goal={},
        tool_context=_tool_context(),
    )
    request = pending.pending_approval
    assert request is not None
    approval = ApprovalDecision(
        request_id=request.request_id,
        action_digest=request.action_digest,
        approved=True,
        session_id=request.session_id,
        run_id=request.run_id,
    )
    model = ScriptedAgentModel([action, _finish()])
    result = _loop(model, effect=ToolEffect.LOCAL_STATE_MUTATION).run(
        session_id="session-1",
        run_id="run-1",
        goal={},
        tool_context=_tool_context(),
        approvals={compute_action_digest(action): approval},
    )
    assert result.state.status == AgentRunStatus.COMPLETED


def test_loop_rejects_approval_replayed_into_another_run() -> None:
    action = _action()
    pending = _loop(
        ScriptedAgentModel([action]),
        effect=ToolEffect.LOCAL_STATE_MUTATION,
    ).run(
        session_id="session-1",
        run_id="run-1",
        goal={},
        tool_context=_tool_context(),
    )
    request = pending.pending_approval
    assert request is not None
    approval = ApprovalDecision(
        request_id=request.request_id,
        action_digest=request.action_digest,
        approved=True,
        session_id=request.session_id,
        run_id=request.run_id,
    )

    model = ScriptedAgentModel([action, _finish()])
    result = _loop(model, effect=ToolEffect.LOCAL_STATE_MUTATION).run(
        session_id="session-1",
        run_id="run-2",
        goal={},
        tool_context=_tool_context().model_copy(update={"run_id": "run-2"}),
        approvals={compute_action_digest(action): approval},
    )

    assert result.state.status == AgentRunStatus.COMPLETED
    assert model.calls[-1].steps[0].observation.error_code == "approval_scope_mismatch"


def test_loop_stops_cleanly_when_scripted_model_fails() -> None:
    result = _loop(ScriptedAgentModel([])).run(
        session_id="session-1",
        run_id="run-1",
        goal={},
        tool_context=_tool_context(),
    )
    assert result.state.status == AgentRunStatus.FAILED
    assert result.error_code == "model_error"
