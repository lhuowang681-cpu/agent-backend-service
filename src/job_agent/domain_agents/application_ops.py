from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from pydantic import Field, ValidationError

from job_agent.agent_runtime.budgets import AgentBudget, BudgetManager, BudgetProfile
from job_agent.agent_runtime.checkpoint import CheckpointStore
from job_agent.agent_runtime.contracts import (
    AgentFinish,
    AgentRunResult,
    AgentStepRecord,
    ApprovalDecision,
    ToolContext,
    ToolEffect,
    ToolObservationStatus,
    VerificationResult,
)
from job_agent.agent_runtime.loop import AgentLoop, AgentModel
from job_agent.agent_runtime.policy import PolicyContext, PolicyEngine
from job_agent.agent_runtime.schema_diagnostics import safe_validation_error_detail
from job_agent.agent_runtime.tool_registry import ToolRegistry
from job_agent.schemas import ApplicationRecord, ApplicationStatus, StrictModel
from job_agent.tools.application_tracker import (
    ApplicationTrackerStore,
    VALID_TRANSITIONS,
)


READ_TRACKER_TOOL = "ops.read_tracker"
VALIDATE_TRANSITION_TOOL = "ops.validate_transition"
APPLY_TRANSITION_TOOL = "ops.apply_transition"
SANDBOX_ACTION_TOOL = "ops.sandbox_action"
APPLICATION_OPS_TOOLS = (
    READ_TRACKER_TOOL,
    VALIDATE_TRANSITION_TOOL,
    APPLY_TRANSITION_TOOL,
    SANDBOX_ACTION_TOOL,
)


class ApplicationOpsGoal(StrictModel):
    application_id: str = Field(min_length=1)
    desired_status: ApplicationStatus
    allow_sandbox_action: bool = False


class ReadTrackerInput(StrictModel):
    application_id: str = Field(min_length=1)


class ReadTrackerOutput(StrictModel):
    revision: int = Field(ge=0)
    application: ApplicationRecord


class ValidateTransitionInput(StrictModel):
    application_id: str = Field(min_length=1)
    new_status: ApplicationStatus
    expected_revision: int = Field(ge=0)
    notes: str = ""


class TrackerTransitionProposal(StrictModel):
    application_id: str
    from_status: ApplicationStatus
    to_status: ApplicationStatus
    expected_revision: int = Field(ge=0)
    notes: str = ""
    valid: bool
    reason_code: str


class ApplyTransitionInput(ValidateTransitionInput):
    pass


class ApplyTransitionOutput(StrictModel):
    application: ApplicationRecord
    tracker_revision: int = Field(ge=0)


class SandboxActionInput(StrictModel):
    channel: str = Field(pattern=r"^(email|calendar|form)$")
    target: str = Field(min_length=1)
    content: dict = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")


class SandboxActionOutput(StrictModel):
    sandbox: bool = True
    receipt_id: str
    path: str


class ApplicationOpsResult(StrictModel):
    proposal: TrackerTransitionProposal
    applied_record: ApplicationRecord | None = None
    tracker_revision: int | None = Field(default=None, ge=0)
    sandbox_receipts: list[SandboxActionOutput] = Field(default_factory=list)


def build_application_ops_registry(
    tracker_store: ApplicationTrackerStore,
    *,
    user_id: str,
) -> ToolRegistry:
    registry = ToolRegistry()

    def _find(application_id: str):
        tracker = tracker_store.load(user_id=user_id)
        application = next(
            (item for item in tracker.applications if item.id == application_id),
            None,
        )
        if application is None:
            raise ValueError("application not found")
        return tracker, application

    def read_tracker(value: ReadTrackerInput, context: ToolContext) -> ReadTrackerOutput:
        tracker, application = _find(value.application_id)
        return ReadTrackerOutput(revision=tracker.revision, application=application)

    def validate_transition(
        value: ValidateTransitionInput,
        context: ToolContext,
    ) -> TrackerTransitionProposal:
        tracker, application = _find(value.application_id)
        if tracker.revision != value.expected_revision:
            valid = False
            reason = "stale_tracker_revision"
        elif value.new_status == application.current_state:
            valid = True
            reason = "no_change"
        elif value.new_status in VALID_TRANSITIONS[application.current_state]:
            valid = True
            reason = "valid_transition"
        else:
            valid = False
            reason = "invalid_transition"
        return TrackerTransitionProposal(
            application_id=value.application_id,
            from_status=application.current_state,
            to_status=value.new_status,
            expected_revision=value.expected_revision,
            notes=value.notes,
            valid=valid,
            reason_code=reason,
        )

    def apply_transition(value: ApplyTransitionInput, context: ToolContext) -> ApplyTransitionOutput:
        tracker, application = _find(value.application_id)
        if tracker.revision != value.expected_revision:
            raise ValueError("stale tracker revision")
        if value.new_status != application.current_state and value.new_status not in VALID_TRANSITIONS[
            application.current_state
        ]:
            raise ValueError("invalid transition")
        updated = tracker_store.update_status(
            user_id=user_id,
            application_id=value.application_id,
            new_status=value.new_status,
            notes=value.notes,
            user_confirmed=True,
            timestamp=tracker_store.current_timestamp(),
        )
        revision = tracker_store.load(user_id=user_id).revision
        return ApplyTransitionOutput(application=updated, tracker_revision=revision)

    def sandbox_action(value: SandboxActionInput, context: ToolContext) -> SandboxActionOutput:
        if not context.sandbox_root:
            raise ValueError("sandbox root is required")
        root = Path(context.sandbox_root).resolve()
        outbox = (root / "external_outbox").resolve()
        if outbox != root and not outbox.is_relative_to(root):
            raise ValueError("sandbox path escape")
        outbox.mkdir(parents=True, exist_ok=True)
        path = (outbox / f"{value.idempotency_key}.json").resolve()
        if path.parent != outbox:
            raise ValueError("sandbox action path escape")
        payload = {
            "sandbox": True,
            "channel": value.channel,
            "target": value.target,
            "content": value.content,
            "idempotency_key": value.idempotency_key,
        }
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise ValueError("sandbox idempotency conflict")
        else:
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(path)
        return SandboxActionOutput(
            receipt_id=f"sandbox-{value.idempotency_key}",
            path=str(path),
        )

    registry.register(
        name=READ_TRACKER_TOOL,
        description="Read one application from the approved local tracker.",
        input_model=ReadTrackerInput,
        output_model=ReadTrackerOutput,
        handler=read_tracker,
        effect=ToolEffect.READ_ONLY,
    )
    registry.register(
        name=VALIDATE_TRANSITION_TOOL,
        description="Validate a tracker transition and expected revision without mutation.",
        input_model=ValidateTransitionInput,
        output_model=TrackerTransitionProposal,
        handler=validate_transition,
        effect=ToolEffect.READ_ONLY,
    )
    registry.register(
        name=APPLY_TRANSITION_TOOL,
        description="Apply one approved local tracker transition with stale-revision protection.",
        input_model=ApplyTransitionInput,
        output_model=ApplyTransitionOutput,
        handler=apply_transition,
        effect=ToolEffect.LOCAL_STATE_MUTATION,
        idempotent=False,
    )
    registry.register(
        name=SANDBOX_ACTION_TOOL,
        description="Write an email/calendar/form action only to the approved local sandbox outbox.",
        input_model=SandboxActionInput,
        output_model=SandboxActionOutput,
        handler=sandbox_action,
        effect=ToolEffect.EXTERNAL_REVERSIBLE_WRITE,
        idempotent=True,
        sandbox_only=True,
    )
    return registry


class ApplicationOpsVerifier:
    def verify(
        self,
        finish: AgentFinish,
        *,
        goal: dict,
        steps: Sequence[AgentStepRecord],
    ) -> VerificationResult:
        try:
            typed_goal = ApplicationOpsGoal.model_validate(goal)
            result = ApplicationOpsResult.model_validate(finish.result)
        except ValidationError as exc:
            detail = safe_validation_error_detail(
                exc,
                allowed_fields={
                    "proposal",
                    "applied_record",
                    "tracker_revision",
                    "sandbox_receipts",
                },
            )
            return VerificationResult(
                passed=False,
                reason_code=f"application_ops_schema_error:{detail}",
                feedback=(
                    "Return a valid proposal grounded in ops tool observations. "
                    f"Shape diagnostic: {detail}"
                ),
            )
        except Exception:
            return VerificationResult(
                passed=False,
                reason_code="application_ops_schema_error",
                feedback="Return a valid proposal grounded in ops tool observations.",
            )
        proposals: list[TrackerTransitionProposal] = []
        applied: list[ApplyTransitionOutput] = []
        receipts: list[SandboxActionOutput] = []
        for step in steps:
            if step.observation.status != ToolObservationStatus.OK:
                continue
            try:
                if step.action.tool_name == VALIDATE_TRANSITION_TOOL:
                    proposals.append(TrackerTransitionProposal.model_validate(step.observation.data))
                elif step.action.tool_name == APPLY_TRANSITION_TOOL:
                    applied.append(ApplyTransitionOutput.model_validate(step.observation.data))
                elif step.action.tool_name == SANDBOX_ACTION_TOOL:
                    receipts.append(SandboxActionOutput.model_validate(step.observation.data))
            except Exception:
                continue
        if not proposals or result.proposal != proposals[-1]:
            return VerificationResult(
                passed=False,
                reason_code="tracker_proposal_mismatch",
                feedback="Use the latest deterministic transition proposal.",
            )
        if (
            result.proposal.application_id != typed_goal.application_id
            or result.proposal.to_status != typed_goal.desired_status
        ):
            return VerificationResult(
                passed=False,
                reason_code="tracker_goal_mismatch",
                feedback="Proposal must match the approved application and desired status.",
            )
        if applied:
            if result.applied_record != applied[-1].application or result.tracker_revision != applied[-1].tracker_revision:
                return VerificationResult(
                    passed=False,
                    reason_code="tracker_apply_mismatch",
                    feedback="Copy the approved tracker mutation observation exactly.",
                )
        elif result.applied_record is not None or result.tracker_revision is not None:
            return VerificationResult(
                passed=False,
                reason_code="unobserved_tracker_mutation",
                feedback="Do not claim a tracker mutation without an approved apply observation.",
            )
        if result.sandbox_receipts != receipts:
            return VerificationResult(
                passed=False,
                reason_code="sandbox_receipt_mismatch",
                feedback="Copy sandbox receipts exactly and never claim a real external action.",
            )
        if receipts and not typed_goal.allow_sandbox_action:
            return VerificationResult(
                passed=False,
                reason_code="sandbox_action_not_requested",
                feedback="Do not create sandbox actions unless the goal explicitly allows them.",
            )
        return VerificationResult(
            passed=True,
            reason_code="application_ops_complete",
            feedback="Tracker proposal/mutation and sandbox receipts are grounded.",
            evidence_refs=[f"application:{typed_goal.application_id}"],
        )


class ApplicationOpsAgent:
    def __init__(
        self,
        *,
        model: AgentModel,
        tracker_store: ApplicationTrackerStore,
        user_id: str,
        budget: AgentBudget | None = None,
        trajectory=None,
        checkpoint_store: CheckpointStore | None = None,
        max_same_verifier_reason: int | None = None,
    ) -> None:
        self.registry = build_application_ops_registry(tracker_store, user_id=user_id)
        policy_context = PolicyContext(
            skill_allowed_tools=APPLICATION_OPS_TOOLS,
            agent_allowed_tools=APPLICATION_OPS_TOOLS,
            runtime_allowed_tools=APPLICATION_OPS_TOOLS,
            target_summary="local tracker or external sandbox action",
            arguments_preview_keys=("application_id", "new_status", "channel", "target"),
        )
        self.loop = AgentLoop(
            agent_id="application-ops",
            model=model,
            registry=self.registry,
            policy=PolicyEngine(),
            policy_context=policy_context,
            budget=BudgetManager(budget or AgentBudget.for_profile(BudgetProfile.STANDARD)),
            verifier=ApplicationOpsVerifier(),
            trajectory=trajectory,
            checkpoint_store=checkpoint_store,
            skill_version="application-ops-v1",
            prompt_version="application-ops-controller-v1",
            max_same_verifier_reason=max_same_verifier_reason,
        )

    def run(
        self,
        goal: ApplicationOpsGoal,
        *,
        session_id: str,
        run_id: str,
        workspace_root: Path | str,
        sandbox_root: Path | str,
        approvals: Mapping[str, ApprovalDecision] | None = None,
        resume: bool = False,
    ) -> AgentRunResult:
        return self.loop.run(
            session_id=session_id,
            run_id=run_id,
            goal=goal.model_dump(mode="json"),
            tool_context=ToolContext(
                session_id=session_id,
                run_id=run_id,
                agent_id="application-ops",
                workspace_root=str(Path(workspace_root).resolve()),
                sandbox_root=str(Path(sandbox_root).resolve()),
            ),
            approvals=approvals,
            resume=resume,
        )
