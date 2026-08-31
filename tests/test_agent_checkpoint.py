from __future__ import annotations

import json

from pydantic import Field

from job_agent.agent_runtime.budgets import AgentBudget, BudgetManager, BudgetProfile
from job_agent.agent_runtime.checkpoint import JsonCheckpointStore
from job_agent.agent_runtime.contracts import (
    AgentAction,
    AgentFinish,
    AgentInputRequest,
    AgentRunStatus,
    ApprovalDecision,
    ToolContext,
    ToolEffect,
    VerificationResult,
    UserInputRequest,
    UserInputResponse,
)
from job_agent.agent_runtime.loop import AgentLoop, ScriptedAgentModel
from job_agent.agent_runtime.policy import PolicyContext, PolicyEngine, compute_action_digest
from job_agent.agent_runtime.tool_registry import ToolRegistry
from job_agent.agent_runtime.trajectory import InMemoryTrajectory
from job_agent.schemas import StrictModel


class MutationInput(StrictModel):
    value: str = Field(min_length=1)


class MutationOutput(StrictModel):
    value: str


class HasObservationVerifier:
    def verify(self, finish, *, goal, steps):
        return VerificationResult(
            passed=bool(steps),
            reason_code="complete" if steps else "missing_observation",
            feedback="complete" if steps else "Use the approved tool first.",
        )


class AlwaysVerifier:
    def verify(self, finish, *, goal, steps):
        return VerificationResult(passed=True, reason_code="complete", feedback="complete")


def _action() -> AgentAction:
    return AgentAction(
        action_id="mutation-1",
        tool_name="state.mutate",
        tool_arguments={"value": "approved"},
        expected_observation="updated sandbox state",
        progress_claim="apply approved sandbox mutation",
    )


def _finish() -> AgentFinish:
    return AgentFinish(
        result={"status": "done"},
        completion_evidence=["step:1:state.mutate"],
        confidence=0.9,
    )


def _loop(
    model,
    store,
    calls,
    *,
    trajectory=None,
    description="Mutate sandbox state",
    verifier=None,
) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(
        name="state.mutate",
        description=description,
        input_model=MutationInput,
        output_model=MutationOutput,
        handler=lambda value, context: calls.append(value.value) or MutationOutput(value=value.value),
        effect=ToolEffect.LOCAL_STATE_MUTATION,
        idempotent=False,
    )
    return AgentLoop(
        agent_id="checkpoint-agent",
        model=model,
        registry=registry,
        policy=PolicyEngine(),
        policy_context=PolicyContext(
            skill_allowed_tools=("state.mutate",),
            agent_allowed_tools=("state.mutate",),
            runtime_allowed_tools=("state.mutate",),
        ),
        budget=BudgetManager(AgentBudget.for_profile(BudgetProfile.QUICK)),
        verifier=verifier or HasObservationVerifier(),
        trajectory=trajectory,
        checkpoint_store=store,
        skill_version="checkpoint-fixture-v1",
        prompt_version="checkpoint-prompt-v1",
    )


def _context(tmp_path) -> ToolContext:
    return ToolContext(
        session_id="session-1",
        run_id="run-1",
        agent_id="checkpoint-agent",
        workspace_root=str(tmp_path),
        sandbox_root=str(tmp_path / "sandbox"),
    )


def test_waiting_checkpoint_resumes_same_run_without_replanning_action(tmp_path) -> None:
    path = tmp_path / ".internal" / "run-1" / "checkpoint.json"
    store = JsonCheckpointStore(path)
    calls: list[str] = []
    action = _action()

    waiting = _loop(ScriptedAgentModel([action]), store, calls).run(
        session_id="session-1",
        run_id="run-1",
        goal={"operation": "sandbox update"},
        tool_context=_context(tmp_path),
    )
    assert waiting.state.status == AgentRunStatus.WAITING_FOR_USER
    assert waiting.state.last_checkpoint_id
    assert calls == []
    checkpoint = store.load()
    assert checkpoint.pending_action == action
    assert checkpoint.budget_usage.model_calls == 1

    request = waiting.pending_approval
    assert request is not None
    approval = ApprovalDecision(
        request_id=request.request_id,
        action_digest=request.action_digest,
        approved=True,
        session_id=request.session_id,
        run_id=request.run_id,
    )
    model = ScriptedAgentModel([_finish()])
    trajectory = InMemoryTrajectory()
    completed = _loop(model, store, calls, trajectory=trajectory).run(
        session_id="session-1",
        run_id="run-1",
        goal={"operation": "sandbox update"},
        tool_context=_context(tmp_path),
        approvals={compute_action_digest(action): approval},
        resume=True,
    )

    assert completed.state.status == AgentRunStatus.COMPLETED
    assert calls == ["approved"]
    assert model.calls[0].steps[0].observation.data == {"value": "approved"}
    assert model.calls[0].budget_usage["model_calls"] == 1
    assert model.calls[0].budget_usage["tool_calls"] == 1
    assert [event.event_type for event in trajectory.events][:3] == [
        "run_resumed",
        "policy_decision",
        "approval_resolved",
    ]
    assert not path.exists()


def test_resume_without_approval_remains_paused_and_does_not_call_model_or_tool(tmp_path) -> None:
    store = JsonCheckpointStore(tmp_path / "checkpoint.json")
    calls: list[str] = []
    action = _action()
    _loop(ScriptedAgentModel([action]), store, calls).run(
        session_id="session-1",
        run_id="run-1",
        goal={"operation": "sandbox update"},
        tool_context=_context(tmp_path),
    )
    model = ScriptedAgentModel([_finish()])
    waiting = _loop(model, store, calls).run(
        session_id="session-1",
        run_id="run-1",
        goal={"operation": "sandbox update"},
        tool_context=_context(tmp_path),
        resume=True,
    )
    assert waiting.state.status == AgentRunStatus.WAITING_FOR_USER
    assert model.calls == []
    assert calls == []


def test_corrupt_or_incompatible_checkpoint_fails_closed_and_is_preserved(tmp_path) -> None:
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text('{"checkpoint_version":', encoding="utf-8")
    corrupt_store = JsonCheckpointStore(corrupt_path)
    result = _loop(ScriptedAgentModel([]), corrupt_store, []).run(
        session_id="session-1",
        run_id="run-1",
        goal={},
        tool_context=_context(tmp_path),
        resume=True,
    )
    assert result.state.status == AgentRunStatus.FAILED
    assert result.error_code == "checkpoint_invalid"
    assert corrupt_path.exists()

    store = JsonCheckpointStore(tmp_path / "compatible.json")
    action = _action()
    _loop(ScriptedAgentModel([action]), store, []).run(
        session_id="session-1",
        run_id="run-1",
        goal={"operation": "sandbox update"},
        tool_context=_context(tmp_path),
    )
    incompatible = _loop(
        ScriptedAgentModel([]),
        store,
        [],
        description="Changed tool contract",
    ).run(
        session_id="session-1",
        run_id="run-1",
        goal={"operation": "sandbox update"},
        tool_context=_context(tmp_path),
        resume=True,
    )
    assert incompatible.state.status == AgentRunStatus.FAILED
    assert incompatible.error_code == "checkpoint_registry_mismatch"
    assert json.loads(store.path.read_text(encoding="utf-8"))["pending_action"]["action_id"] == "mutation-1"


def test_user_input_interrupt_resumes_into_model_context_without_counting_wait_time(tmp_path) -> None:
    store = JsonCheckpointStore(tmp_path / "user-input.json")
    request = UserInputRequest(
        request_id="answer-1",
        prompt="Explain the design tradeoff.",
        response_schema={"type": "object", "required": ["answer"]},
        context_summary="mock interview question",
    )
    waiting = _loop(
        ScriptedAgentModel(
            [AgentInputRequest(request=request, progress_claim="ask one adaptive question")]
        ),
        store,
        [],
        verifier=AlwaysVerifier(),
    ).run(
        session_id="session-1",
        run_id="run-1",
        goal={"mode": "mock"},
        tool_context=_context(tmp_path),
    )
    assert waiting.state.status == AgentRunStatus.WAITING_FOR_USER
    assert waiting.pending_user_input == request

    response = UserInputResponse(request_id="answer-1", value={"answer": "Use explicit state."})
    model = ScriptedAgentModel([_finish()])
    trajectory = InMemoryTrajectory()
    result = _loop(
        model,
        store,
        [],
        trajectory=trajectory,
        verifier=AlwaysVerifier(),
    ).run(
        session_id="session-1",
        run_id="run-1",
        goal={"mode": "mock"},
        tool_context=_context(tmp_path),
        user_inputs={"answer-1": response},
        resume=True,
    )
    assert result.state.status == AgentRunStatus.COMPLETED
    assert model.calls[0].user_inputs[0].response.value == {"answer": "Use explicit state."}
    assert [event.event_type for event in trajectory.events][:2] == [
        "run_resumed",
        "user_input_resolved",
    ]
