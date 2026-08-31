from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from pydantic import Field, ValidationError

from job_agent.agent_runtime.budgets import AgentBudget, BudgetManager, BudgetProfile
from job_agent.agent_runtime.checkpoint import CheckpointStore
from job_agent.agent_runtime.contracts import (
    AgentAction,
    AgentFinish,
    AgentModelContext,
    AgentModelResponse,
    AgentStepRecord,
    ToolEffect,
    ToolObservationStatus,
    VerificationResult,
)
from job_agent.agent_runtime.loop import AgentLoop
from job_agent.agent_runtime.policy import PolicyContext, PolicyEngine
from job_agent.agent_runtime.tool_executor import ToolExecutor
from job_agent.agent_runtime.tool_registry import ToolRegistry
from job_agent.backend_service.career_snapshot import CareerSnapshotRecord
from job_agent.backend_service.contracts import ApplicationAssistantInput, ClaimedRun, RunRecord
from job_agent.backend_service.tool_operations import LedgeredToolExecutor
from job_agent.schemas import StrictModel


READ_SNAPSHOT_TOOL = "career.read_snapshot"
WRITE_SANDBOX_DRAFT_TOOL = "application.write_sandbox_draft"
SEARCH_SNAPSHOT_TEXT_TOOL = "career.search_snapshot_text"
APPLICATION_ASSISTANT_TOOLS = (
    READ_SNAPSHOT_TOOL,
    SEARCH_SNAPSHOT_TEXT_TOOL,
)



class SearchSnapshotTextInput(StrictModel):
    snapshot_revision: str
    query: str = Field(min_length=1)

class SearchSnapshotTextOutput(StrictModel):
    snapshot_revision: str
    matches: list[str]



class ReadSnapshotInput(StrictModel):
    snapshot_revision: str


class ReadSnapshotOutput(StrictModel):
    snapshot_revision: str
    source_ref: str
    resume_text: str
    evidence_refs: list[str] = Field(default_factory=list)


class WriteSandboxDraftInput(StrictModel):
    channel: str = Field(pattern=r"^(email|form)$")
    target: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)


class SandboxDraftReceipt(StrictModel):
    sandbox: bool = True
    receipt_id: str
    path: str


class ApplicationAssistantResult(StrictModel):
    job_id: str
    snapshot_revision: str
    plan_summary: str = Field(min_length=1)
    draft_receipt: SandboxDraftReceipt


def build_application_assistant_registry(
    *,
    snapshot: CareerSnapshotRecord,
) -> ToolRegistry:
    registry = ToolRegistry()

    def search_snapshot_text(
        value:SearchSnapshotTextInput,
        _context
    )-> SearchSnapshotTextOutput:
        if value.snapshot_revision != snapshot.snapshot_revision:
            raise ValueError("career_snapshot_revision_mismatch")
        lines = snapshot.payload.resume_text.splitlines()
        matches = [line for line in lines if value.query.casefold() in line.casefold()]
        return SearchSnapshotTextOutput(
            snapshot_revision=snapshot.snapshot_revision,
            matches=matches
        )


    def read_snapshot(
        value: ReadSnapshotInput,
        _context,
    ) -> ReadSnapshotOutput:
        if value.snapshot_revision != snapshot.snapshot_revision:
            raise ValueError("career_snapshot_revision_mismatch")
        return ReadSnapshotOutput(
            snapshot_revision=snapshot.snapshot_revision,
            source_ref=snapshot.payload.source_ref,
            resume_text=snapshot.payload.resume_text,
            evidence_refs=snapshot.payload.evidence_refs,
        )

    def write_sandbox_draft(
        value: WriteSandboxDraftInput,
        context,
    ) -> SandboxDraftReceipt:
        if not context.sandbox_root:
            raise ValueError("sandbox_root_required")
        root = Path(context.sandbox_root).resolve()
        outbox = (root / "application_drafts").resolve()
        if outbox != root and not outbox.is_relative_to(root):
            raise ValueError("sandbox_path_escape")
        outbox.mkdir(parents=True, exist_ok=True)
        receipt_id = f"draft-{uuid4().hex}"
        path = (outbox / f"{receipt_id}.json").resolve()
        if path.parent != outbox:
            raise ValueError("sandbox_path_escape")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "sandbox": True,
                    "channel": value.channel,
                    "target": value.target,
                    "subject": value.subject,
                    "body": value.body,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
        return SandboxDraftReceipt(receipt_id=receipt_id, path=str(path))

    registry.register(
        name=READ_SNAPSHOT_TOOL,
        description="Read the immutable CareerSnapshot bound to this Run.",
        input_model=ReadSnapshotInput,
        output_model=ReadSnapshotOutput,
        handler=read_snapshot,
        effect=ToolEffect.READ_ONLY,
    )
    registry.register(
        name=SEARCH_SNAPSHOT_TEXT_TOOL,
        description="Search the CareerSnapshot resume text for lines matching a query.",
        input_model=SearchSnapshotTextInput,
        output_model=SearchSnapshotTextOutput,
        handler=search_snapshot_text,
        effect=ToolEffect.READ_ONLY,
    )
    registry.register(
        name=WRITE_SANDBOX_DRAFT_TOOL,
        description="Create one application draft in the disposable sandbox only.",
        input_model=WriteSandboxDraftInput,
        output_model=SandboxDraftReceipt,
        handler=write_sandbox_draft,
        effect=ToolEffect.EXTERNAL_DRAFT,
        idempotent=False,
        sandbox_only=True,
    )
    return registry


class ApplicationAssistantMockModel:
    """Deterministic controller used for recovery and approval tests."""

    def __init__(self, *, request: ApplicationAssistantInput) -> None:
        self.request = request

    def decide(self, context: AgentModelContext) -> AgentModelResponse:
        snapshot_step = next(
            (
                step
                for step in context.steps
                if step.action.tool_name == READ_SNAPSHOT_TOOL
                and step.observation.status == ToolObservationStatus.OK
            ),
            None,
        )
        draft_step = next(
            (
                step
                for step in context.steps
                if step.action.tool_name == WRITE_SANDBOX_DRAFT_TOOL
                and step.observation.status == ToolObservationStatus.OK
            ),
            None,
        )
        if snapshot_step is None:
            decision = AgentAction(
                action_id="read-bound-career-snapshot",
                tool_name=READ_SNAPSHOT_TOOL,
                tool_arguments={
                    "snapshot_revision": self.request.career_snapshot_revision,
                },
                expected_observation="bound immutable CareerSnapshot",
                progress_claim="read the Run-bound CareerSnapshot",
            )
        elif draft_step is None:
            job = self.request.selected_job
            decision = AgentAction(
                action_id="write-approved-application-draft",
                tool_name=WRITE_SANDBOX_DRAFT_TOOL,
                tool_arguments={
                    "channel": self.request.draft_channel,
                    "target": self.request.draft_target,
                    "subject": f"Application for {job.title}",
                    "body": (
                        f"Sandbox draft for {job.company} / {job.title}; "
                        f"grounded in snapshot {self.request.career_snapshot_revision}."
                    ),
                },
                expected_observation="sandbox draft receipt",
                progress_claim="prepare an approval-gated sandbox application draft",
            )
        else:
            receipt = SandboxDraftReceipt.model_validate(draft_step.observation.data)
            decision = AgentFinish(
                result=ApplicationAssistantResult(
                    job_id=self.request.selected_job.job_id,
                    snapshot_revision=self.request.career_snapshot_revision,
                    plan_summary=(
                        "Prepared one sandbox-only application draft from the "
                        "Run-bound CareerSnapshot."
                    ),
                    draft_receipt=receipt,
                ).model_dump(mode="json"),
                completion_evidence=[f"tool:{draft_step.action.action_id}"],
                confidence=1.0,
            )
        return AgentModelResponse(
            decision=decision,
            provider="mock",
            model="application-assistant-mock-v1",
        )


class ApplicationAssistantVerifier:
    def verify(
        self,
        finish: AgentFinish,
        *,
        goal: dict,
        steps: list[AgentStepRecord],
    ) -> VerificationResult:
        try:
            request = ApplicationAssistantInput.model_validate(goal)
            result = ApplicationAssistantResult.model_validate(finish.result)
            snapshot = next(
                ReadSnapshotOutput.model_validate(step.observation.data)
                for step in reversed(steps)
                if step.action.tool_name == READ_SNAPSHOT_TOOL
                and step.observation.status == ToolObservationStatus.OK
            )
            receipt = next(
                SandboxDraftReceipt.model_validate(step.observation.data)
                for step in reversed(steps)
                if step.action.tool_name == WRITE_SANDBOX_DRAFT_TOOL
                and step.observation.status == ToolObservationStatus.OK
            )
        except (ValidationError, StopIteration, ValueError):
            return VerificationResult(
                passed=False,
                reason_code="application_assistant_evidence_missing",
                feedback="Read the bound snapshot and return the observed sandbox receipt.",
            )
        passed = (
            snapshot.snapshot_revision == request.career_snapshot_revision
            and result.snapshot_revision == request.career_snapshot_revision
            and result.job_id == request.selected_job.job_id
            and result.draft_receipt == receipt
            and receipt.sandbox
        )
        return VerificationResult(
            passed=passed,
            reason_code=(
                "application_assistant_complete"
                if passed
                else "application_assistant_result_mismatch"
            ),
            feedback=(
                "Application draft is grounded and sandbox-only."
                if passed
                else "Return only the bound job, snapshot revision, and observed receipt."
            ),
            evidence_refs=[
                f"snapshot:{request.career_snapshot_revision}",
                f"job:{request.selected_job.job_id}",
            ],
        )


def _build_application_assistant_runtime(
    *,
    request: ApplicationAssistantInput,
    snapshot: CareerSnapshotRecord,
    checkpoint_store: CheckpointStore,
    executor: ToolExecutor | None,
    registry: ToolRegistry | None = None,
) -> AgentLoop:
    registry = registry or build_application_assistant_registry(snapshot=snapshot)
    return AgentLoop(
        agent_id="application-assistant",
        model=ApplicationAssistantMockModel(request=request),
        registry=registry,
        policy=PolicyEngine(),
        policy_context=PolicyContext(
            skill_allowed_tools=APPLICATION_ASSISTANT_TOOLS,
            agent_allowed_tools=APPLICATION_ASSISTANT_TOOLS,
            runtime_allowed_tools=APPLICATION_ASSISTANT_TOOLS,
            target_summary="Run-owned application sandbox",
            arguments_preview_keys=("channel", "target", "subject"),
        ),
        budget=BudgetManager(AgentBudget.for_profile(BudgetProfile.QUICK)),
        verifier=ApplicationAssistantVerifier(),
        checkpoint_store=checkpoint_store,
        skill_version="application-assistant-v1",
        prompt_version="application-assistant-controller-v1",
        executor=executor,
    )


def build_application_assistant_loop(
    *,
    record: RunRecord,
    snapshot: CareerSnapshotRecord,
    checkpoint_store: CheckpointStore,
    repository,
    claim: ClaimedRun,
    tool_failpoint=None,
) -> AgentLoop:
    if not isinstance(record.request.input, ApplicationAssistantInput):
        raise ValueError("task_input_contract_mismatch")
    registry = build_application_assistant_registry(snapshot=snapshot)
    executor = LedgeredToolExecutor(
        registry,
        repository=repository,
        claim=claim,
        failpoint=tool_failpoint,
    )
    return _build_application_assistant_runtime(
        request=record.request.input,
        snapshot=snapshot,
        checkpoint_store=checkpoint_store,
        executor=executor,
        registry=registry,
    )


def build_application_assistant_eval_loop(
    *,
    request: ApplicationAssistantInput,
    snapshot: CareerSnapshotRecord,
    checkpoint_store: CheckpointStore,
    executor: ToolExecutor | None = None,
) -> AgentLoop:
    """Build the same AgentLoop with an evaluation-scoped Tool executor."""

    return _build_application_assistant_runtime(
        request=request,
        snapshot=snapshot,
        checkpoint_store=checkpoint_store,
        executor=executor,
    )
