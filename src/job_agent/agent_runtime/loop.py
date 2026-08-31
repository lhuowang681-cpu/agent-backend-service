from __future__ import annotations

import hashlib
import json
from typing import Mapping, Protocol, Sequence

from job_agent.agent_runtime.budgets import BudgetManager
from job_agent.agent_runtime.checkpoint import AgentCheckpoint, CheckpointStore
from job_agent.agent_runtime.contracts import (
    AgentAction,
    AgentFinish,
    AgentInputRequest,
    AgentModelContext,
    AgentModelResponse,
    AgentRunResult,
    AgentRunState,
    AgentRunStatus,
    AgentStepRecord,
    ApprovalDecision,
    ApprovalRequest,
    ToolContext,
    ToolObservation,
    ToolObservationStatus,
    UserInputRecord,
    UserInputRequest,
    UserInputResponse,
    VerificationResult,
)
from job_agent.agent_runtime.events import (
    ApprovalResolvedEvent,
    ModelDecisionEvent,
    PolicyDecisionEvent,
    RunFinishedEvent,
    RunResumedEvent,
    RunStartedEvent,
    ToolObservedEvent,
    VerificationEvent,
    UserInputResolvedEvent,
    event_identity,
)
from job_agent.agent_runtime.failures import BudgetExceededError, CheckpointError, ToolRegistryError
from job_agent.agent_runtime.policy import PolicyContext, PolicyEngine, compute_action_digest
from job_agent.agent_runtime.tool_executor import ToolExecutor
from job_agent.agent_runtime.tool_registry import ToolRegistry
from job_agent.agent_runtime.trajectory import InMemoryTrajectory, TrajectorySink


class AgentModel(Protocol):
    def decide(self, context: AgentModelContext) -> AgentModelResponse:
        ...


class FinishVerifier(Protocol):
    def verify(
        self,
        finish: AgentFinish,
        *,
        goal: dict,
        steps: Sequence[AgentStepRecord],
    ) -> VerificationResult:
        ...


class ScriptedAgentModel:
    """Deterministic model used for trajectory replay and contract tests."""

    def __init__(
        self,
        responses: Sequence[AgentModelResponse | AgentAction | AgentFinish | AgentInputRequest],
    ) -> None:
        self._responses = list(responses)
        self.calls: list[AgentModelContext] = []

    def decide(self, context: AgentModelContext) -> AgentModelResponse:
        self.calls.append(context.model_copy(deep=True))
        if not self._responses:
            raise RuntimeError("scripted agent model exhausted")
        response = self._responses.pop(0)
        if isinstance(response, AgentModelResponse):
            return response
        return AgentModelResponse(decision=response, provider="scripted", model="scripted")


class AgentLoop:
    def __init__(
        self,
        *,
        agent_id: str,
        model: AgentModel,
        registry: ToolRegistry,
        policy: PolicyEngine,
        policy_context: PolicyContext,
        budget: BudgetManager,
        verifier: FinishVerifier,
        trajectory: TrajectorySink | None = None,
        checkpoint_store: CheckpointStore | None = None,
        runtime_version: str = "agent-loop-v1",
        skill_version: str = "unknown",
        prompt_version: str = "unknown",
        max_same_verifier_reason: int | None = None,
        executor: ToolExecutor | None = None,
    ) -> None:
        if max_same_verifier_reason is not None and max_same_verifier_reason <= 0:
            raise ValueError("max_same_verifier_reason must be positive")
        self.agent_id = agent_id
        self.model = model
        self.registry = registry
        self.policy = policy
        self.policy_context = policy_context
        self.budget = budget
        self.verifier = verifier
        self.trajectory = trajectory or InMemoryTrajectory()
        self.executor = executor or ToolExecutor(registry)
        self.checkpoint_store = checkpoint_store
        self.runtime_version = runtime_version
        self.skill_version = skill_version
        self.prompt_version = prompt_version
        self.max_same_verifier_reason = max_same_verifier_reason

    def run(
        self,
        *,
        session_id: str,
        run_id: str,
        goal: dict,
        tool_context: ToolContext,
        approvals: Mapping[str, ApprovalDecision] | None = None,
        user_inputs: Mapping[str, UserInputResponse] | None = None,
        resume: bool = False,
    ) -> AgentRunResult:
        approvals = approvals or {}
        user_inputs = user_inputs or {}
        completed_action_ids: set[str] = set()
        completed_non_idempotent_keys: set[str] = set()
        inflight_non_idempotent_key: str | None = None
        pending_action: AgentAction | None = None
        pending_user_input: UserInputRequest | None = None
        user_input_records: list[UserInputRecord] = []
        verifier_reason_counts: dict[str, int] = {}

        if resume:
            try:
                checkpoint = self._load_checkpoint(
                    session_id=session_id,
                    run_id=run_id,
                    goal=goal,
                )
            except CheckpointError as exc:
                return self._checkpoint_failure(
                    session_id=session_id,
                    run_id=run_id,
                    goal=goal,
                    error_code=str(exc),
                )
            state = checkpoint.state.model_copy(
                update={
                    "status": AgentRunStatus.RUNNING,
                    "last_checkpoint_id": checkpoint.checkpoint_id,
                }
            )
            steps = list(checkpoint.steps)
            user_input_records = list(checkpoint.user_inputs)
            verifier_feedback = list(checkpoint.verifier_feedback)
            completed_action_ids = set(checkpoint.completed_action_ids)
            completed_non_idempotent_keys = set(checkpoint.completed_non_idempotent_keys)
            inflight_non_idempotent_key = checkpoint.inflight_non_idempotent_key
            pending_action = checkpoint.pending_action
            pending_user_input = checkpoint.pending_user_input
            self.budget.restore(checkpoint.budget_usage, paused=True)
            self.trajectory.append(
                RunResumedEvent(
                    **event_identity(
                        session_id=session_id,
                        run_id=run_id,
                        agent_id=self.agent_id,
                        step_index=state.step_index,
                    ),
                    checkpoint_id=checkpoint.checkpoint_id,
                )
            )
            if inflight_non_idempotent_key is not None:
                return self._finish_run(
                    state=state,
                    status=AgentRunStatus.FAILED,
                    error_code="non_idempotent_execution_uncertain",
                    preserve_checkpoint=True,
                )
            if pending_action is not None:
                approval = approvals.get(compute_action_digest(pending_action))
                if approval is None:
                    waiting = checkpoint.state.model_copy(
                        update={"last_checkpoint_id": checkpoint.checkpoint_id}
                    )
                    return AgentRunResult(
                        state=waiting,
                        pending_action=pending_action,
                        pending_approval=checkpoint.pending_approval,
                    )
            if pending_user_input is not None:
                response = user_inputs.get(pending_user_input.request_id)
                if response is None:
                    waiting = checkpoint.state.model_copy(
                        update={"last_checkpoint_id": checkpoint.checkpoint_id}
                    )
                    return AgentRunResult(
                        state=waiting,
                        pending_user_input=pending_user_input,
                    )
                if response.request_id != pending_user_input.request_id:
                    return self._checkpoint_failure(
                        session_id=session_id,
                        run_id=run_id,
                        goal=goal,
                        error_code="user_input_request_mismatch",
                    )
                user_input_records.append(
                    UserInputRecord(
                        step_index=state.step_index,
                        request=pending_user_input,
                        response=response,
                    )
                )
                self.trajectory.append(
                    UserInputResolvedEvent(
                        **event_identity(
                            session_id=session_id,
                            run_id=run_id,
                            agent_id=self.agent_id,
                            step_index=state.step_index,
                        ),
                        response=response,
                    )
                )
                pending_user_input = None
            self.budget.resume_from_user()
            if checkpoint.pending_user_input is not None:
                self._save_checkpoint(
                    state=state,
                    steps=steps,
                    user_inputs=user_input_records,
                    verifier_feedback=verifier_feedback,
                    completed_action_ids=completed_action_ids,
                    completed_non_idempotent_keys=completed_non_idempotent_keys,
                )
        else:
            state = AgentRunState(
                session_id=session_id,
                run_id=run_id,
                agent_id=self.agent_id,
                goal=goal,
                budget_profile=self.budget.budget.profile.value,
            )
            steps: list[AgentStepRecord] = []
            user_input_records = []
            verifier_feedback: list[str] = []
            self.trajectory.append(
                RunStartedEvent(
                    **event_identity(
                        session_id=session_id,
                        run_id=run_id,
                        agent_id=self.agent_id,
                        step_index=0,
                    ),
                    goal=goal,
                    budget_profile=self.budget.budget.profile.value,
                )
            )

        while True:
            if pending_action is None:
                try:
                    self.budget.ensure_available()
                except BudgetExceededError as exc:
                    return self._finish_run(
                        state=state,
                        status=AgentRunStatus.BUDGET_EXCEEDED,
                        error_code=exc.reason_code,
                        preserve_checkpoint=True,
                    )

                context = AgentModelContext(
                    goal=goal,
                    steps=steps,
                    user_inputs=user_input_records,
                    verifier_feedback=verifier_feedback,
                    allowed_tools=[
                        self.registry.get(name).spec
                        for name in self.policy_context.agent_allowed_tools
                    ],
                    budget_usage=self.budget.snapshot().model_dump(mode="json"),
                )
                try:
                    response = self.model.decide(context)
                except Exception:
                    return self._finish_run(
                        state=state,
                        status=AgentRunStatus.FAILED,
                        error_code="model_error",
                        preserve_checkpoint=True,
                    )
                self.budget.record_model_call(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cost=response.cost,
                )
                state = state.model_copy(update={"step_index": state.step_index + 1})
                self.trajectory.append(
                    ModelDecisionEvent(
                        **event_identity(
                            session_id=session_id,
                            run_id=run_id,
                            agent_id=self.agent_id,
                            step_index=state.step_index,
                        ),
                        decision=response.decision,
                        provider=response.provider,
                        model=response.model,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                    )
                )

                if isinstance(response.decision, AgentFinish):
                    verification = self.verifier.verify(response.decision, goal=goal, steps=steps)
                    self.trajectory.append(
                        VerificationEvent(
                            **event_identity(
                                session_id=session_id,
                                run_id=run_id,
                                agent_id=self.agent_id,
                                step_index=state.step_index,
                            ),
                            verification=verification,
                        )
                    )
                    if verification.passed:
                        return self._finish_run(
                            state=state,
                            status=AgentRunStatus.COMPLETED,
                            result=response.decision.result,
                            unresolved_items=response.decision.unresolved_items,
                        )
                    verifier_reason_counts[verification.reason_code] = (
                        verifier_reason_counts.get(verification.reason_code, 0) + 1
                    )
                    if (
                        self.max_same_verifier_reason is not None
                        and verifier_reason_counts[verification.reason_code]
                        >= self.max_same_verifier_reason
                    ):
                        return self._finish_run(
                            state=state,
                            status=AgentRunStatus.FAILED,
                            error_code="repeated_verifier_failure",
                        )
                    verifier_feedback.append(verification.feedback)
                    self._save_checkpoint(
                        state=state,
                        steps=steps,
                        user_inputs=user_input_records,
                        verifier_feedback=verifier_feedback,
                        completed_action_ids=completed_action_ids,
                        completed_non_idempotent_keys=completed_non_idempotent_keys,
                    )
                    continue

                if isinstance(response.decision, AgentInputRequest):
                    self.budget.pause_for_user()
                    waiting = state.model_copy(update={"status": AgentRunStatus.WAITING_FOR_USER})
                    checkpoint_id = self._save_checkpoint(
                        state=waiting,
                        steps=steps,
                        user_inputs=user_input_records,
                        verifier_feedback=verifier_feedback,
                        completed_action_ids=completed_action_ids,
                        completed_non_idempotent_keys=completed_non_idempotent_keys,
                        pending_user_input=response.decision.request,
                    )
                    if checkpoint_id is not None:
                        waiting = waiting.model_copy(update={"last_checkpoint_id": checkpoint_id})
                    return AgentRunResult(
                        state=waiting,
                        pending_user_input=response.decision.request,
                    )

                action = response.decision
            else:
                action = pending_action
                pending_action = None

            if action.action_id in completed_action_ids:
                observation = self._policy_observation(action, "duplicate_action_id")
                self._record_observation(state, observation)
                steps.append(
                    AgentStepRecord(step_index=state.step_index, action=action, observation=observation)
                )
                self._record_budget(action, observation)
                continue

            try:
                tool = self.registry.get(action.tool_name)
            except ToolRegistryError:
                observation = self._policy_observation(action, "unknown_tool")
                self._record_observation(state, observation)
                steps.append(
                    AgentStepRecord(step_index=state.step_index, action=action, observation=observation)
                )
                self._record_budget(action, observation)
                continue

            approval = approvals.get(compute_action_digest(action))
            authorization = self.policy.authorize(
                action,
                tool.spec,
                self.policy_context,
                approval=approval,
                session_id=session_id,
                run_id=run_id,
            )
            self.trajectory.append(
                PolicyDecisionEvent(
                    **event_identity(
                        session_id=session_id,
                        run_id=run_id,
                        agent_id=self.agent_id,
                        step_index=state.step_index,
                    ),
                    authorization=authorization,
                )
            )
            if approval is not None:
                self.trajectory.append(
                    ApprovalResolvedEvent(
                        **event_identity(
                            session_id=session_id,
                            run_id=run_id,
                            agent_id=self.agent_id,
                            step_index=state.step_index,
                        ),
                        approval=approval,
                    )
                )
            if authorization.requires_approval:
                self.budget.pause_for_user()
                waiting = state.model_copy(update={"status": AgentRunStatus.WAITING_FOR_USER})
                checkpoint_id = self._save_checkpoint(
                    state=waiting,
                    steps=steps,
                    user_inputs=user_input_records,
                    verifier_feedback=verifier_feedback,
                    completed_action_ids=completed_action_ids,
                    completed_non_idempotent_keys=completed_non_idempotent_keys,
                    pending_action=action,
                    pending_approval=authorization.approval_request,
                )
                if checkpoint_id is not None:
                    waiting = waiting.model_copy(update={"last_checkpoint_id": checkpoint_id})
                return AgentRunResult(
                    state=waiting,
                    pending_action=action,
                    pending_approval=authorization.approval_request,
                )

            if not authorization.allowed:
                observation = self._policy_observation(action, authorization.reason_code)
            else:
                execution_key = self._execution_key(action)
                if not tool.spec.idempotent and execution_key in completed_non_idempotent_keys:
                    observation = self._policy_observation(action, "non_idempotent_replay_blocked")
                else:
                    if not tool.spec.idempotent:
                        inflight_non_idempotent_key = execution_key
                        self._save_checkpoint(
                            state=state,
                            steps=steps,
                            user_inputs=user_input_records,
                            verifier_feedback=verifier_feedback,
                            completed_action_ids=completed_action_ids,
                            completed_non_idempotent_keys=completed_non_idempotent_keys,
                            inflight_non_idempotent_key=inflight_non_idempotent_key,
                        )
                    effective_tool_context = tool_context.model_copy(
                        update={
                            "user_inputs": {
                                item.request.request_id: item.response
                                for item in user_input_records
                            }
                        }
                    )
                    observation = self.executor.execute(
                        action,
                        authorization,
                        effective_tool_context,
                    )
                    completed_action_ids.add(action.action_id)
                    if not tool.spec.idempotent:
                        completed_non_idempotent_keys.add(execution_key)
                        inflight_non_idempotent_key = None

            self._record_observation(state, observation)
            steps.append(AgentStepRecord(step_index=state.step_index, action=action, observation=observation))
            self._record_budget(action, observation)
            self._save_checkpoint(
                state=state,
                steps=steps,
                user_inputs=user_input_records,
                verifier_feedback=verifier_feedback,
                completed_action_ids=completed_action_ids,
                completed_non_idempotent_keys=completed_non_idempotent_keys,
                inflight_non_idempotent_key=inflight_non_idempotent_key,
            )

    def _record_observation(self, state: AgentRunState, observation: ToolObservation) -> None:
        self.trajectory.append(
            ToolObservedEvent(
                **event_identity(
                    session_id=state.session_id,
                    run_id=state.run_id,
                    agent_id=state.agent_id,
                    step_index=state.step_index,
                ),
                observation=observation,
            )
        )

    def _record_budget(self, action: AgentAction, observation: ToolObservation) -> None:
        observation_digest = hashlib.sha256(
            json.dumps(
                observation.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.budget.record_tool_call(
            fingerprint=self._execution_key(action),
            observation_digest=observation_digest,
        )

    @staticmethod
    def _execution_key(action: AgentAction) -> str:
        payload = {"tool_name": action.tool_name, "tool_arguments": action.tool_arguments}
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _policy_observation(action: AgentAction, error_code: str) -> ToolObservation:
        return ToolObservation(
            action_id=action.action_id,
            tool_name=action.tool_name,
            status=ToolObservationStatus.FATAL_ERROR,
            duration_ms=0,
            error_code=error_code,
        )

    def _load_checkpoint(self, *, session_id: str, run_id: str, goal: dict) -> AgentCheckpoint:
        if self.checkpoint_store is None:
            raise CheckpointError("checkpoint_store_not_configured")
        checkpoint = self.checkpoint_store.load()
        expected_identity = (session_id, run_id, self.agent_id)
        actual_identity = (
            checkpoint.state.session_id,
            checkpoint.state.run_id,
            checkpoint.state.agent_id,
        )
        if actual_identity != expected_identity:
            raise CheckpointError("checkpoint_identity_mismatch")
        if checkpoint.state.goal != goal:
            raise CheckpointError("checkpoint_goal_mismatch")
        if checkpoint.state.budget_profile != self.budget.budget.profile.value:
            raise CheckpointError("checkpoint_budget_profile_mismatch")
        compatibility = (
            (checkpoint.registry_fingerprint, self.registry.fingerprint(), "checkpoint_registry_mismatch"),
            (checkpoint.runtime_version, self.runtime_version, "checkpoint_runtime_mismatch"),
            (checkpoint.skill_version, self.skill_version, "checkpoint_skill_mismatch"),
            (checkpoint.prompt_version, self.prompt_version, "checkpoint_prompt_mismatch"),
        )
        for actual, expected, error_code in compatibility:
            if actual != expected:
                raise CheckpointError(error_code)
        return checkpoint

    def _save_checkpoint(
        self,
        *,
        state: AgentRunState,
        steps: Sequence[AgentStepRecord],
        user_inputs: Sequence[UserInputRecord],
        verifier_feedback: Sequence[str],
        completed_action_ids: set[str],
        completed_non_idempotent_keys: set[str],
        inflight_non_idempotent_key: str | None = None,
        pending_action: AgentAction | None = None,
        pending_approval: ApprovalRequest | None = None,
        pending_user_input: UserInputRequest | None = None,
    ) -> str | None:
        if self.checkpoint_store is None:
            return None
        checkpoint = AgentCheckpoint.create(
            state=state,
            budget_usage=self.budget.snapshot(),
            steps=list(steps),
            user_inputs=list(user_inputs),
            verifier_feedback=list(verifier_feedback),
            completed_action_ids=sorted(completed_action_ids),
            completed_non_idempotent_keys=sorted(completed_non_idempotent_keys),
            inflight_non_idempotent_key=inflight_non_idempotent_key,
            pending_action=pending_action,
            pending_approval=pending_approval,
            pending_user_input=pending_user_input,
            registry_fingerprint=self.registry.fingerprint(),
            runtime_version=self.runtime_version,
            skill_version=self.skill_version,
            prompt_version=self.prompt_version,
        )
        self.checkpoint_store.save(checkpoint)
        return checkpoint.checkpoint_id

    def _checkpoint_failure(
        self,
        *,
        session_id: str,
        run_id: str,
        goal: dict,
        error_code: str,
    ) -> AgentRunResult:
        state = AgentRunState(
            session_id=session_id,
            run_id=run_id,
            agent_id=self.agent_id,
            status=AgentRunStatus.FAILED,
            goal=goal,
            budget_profile=self.budget.budget.profile.value,
        )
        return AgentRunResult(state=state, error_code=error_code)

    def _finish_run(
        self,
        *,
        state: AgentRunState,
        status: AgentRunStatus,
        result: dict | None = None,
        unresolved_items: list[str] | None = None,
        error_code: str | None = None,
        preserve_checkpoint: bool = False,
    ) -> AgentRunResult:
        final_state = state.model_copy(update={"status": status})
        final_result = result or {}
        self.trajectory.append(
            RunFinishedEvent(
                **event_identity(
                    session_id=state.session_id,
                    run_id=state.run_id,
                    agent_id=state.agent_id,
                    step_index=state.step_index,
                ),
                status=status,
                result=final_result,
                error_code=error_code,
            )
        )
        if self.checkpoint_store is not None and not preserve_checkpoint:
            self.checkpoint_store.delete()
        return AgentRunResult(
            state=final_state,
            result=final_result,
            unresolved_items=unresolved_items or [],
            error_code=error_code,
        )
