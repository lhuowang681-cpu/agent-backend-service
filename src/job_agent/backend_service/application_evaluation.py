from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import Field, TypeAdapter

from job_agent.agent_runtime.checkpoint import JsonCheckpointStore
from job_agent.agent_runtime.contracts import (
    AgentAction,
    AgentRunStatus,
    ApprovalDecision,
    AuthorizationResult,
    ToolContext,
    ToolObservation,
)
from job_agent.agent_runtime.events import (
    ModelDecisionEvent,
    PolicyDecisionEvent,
    RunResumedEvent,
    ToolObservedEvent,
)
from job_agent.agent_runtime.policy import compute_action_digest
from job_agent.agent_runtime.tool_executor import ToolExecutor
from job_agent.backend_service.application_assistant import (
    APPLICATION_ASSISTANT_TOOLS,
    WRITE_SANDBOX_DRAFT_TOOL,
    ApplicationAssistantResult,
    build_application_assistant_eval_loop,
    build_application_assistant_registry,
)
from job_agent.backend_service.career_snapshot import (
    CareerSnapshotPayload,
    CareerSnapshotRecord,
)
from job_agent.backend_service.contracts import ApplicationAssistantInput
from job_agent.schemas import RawJob, StrictModel


class ApplicationAssistantEvalCase(StrictModel):
    case_id: str = Field(min_length=1)
    role_family: Literal["agent_algorithm", "llm_application", "backend_data"]
    expected_outcome: Literal["SUCCEEDED", "UNCERTAIN"] = "SUCCEEDED"
    company: str = "Example Labs"
    title: str = "Agent Application Intern"
    draft_channel: Literal["email", "form"] = "email"
    draft_target: str = "recruiting@example.invalid"
    resume_text: str = "Built Agent runtime, approval, checkpoint, and evaluation contracts."
    evidence_refs: list[str] = Field(default_factory=lambda: ["project:job-agent"])
    forbidden_markers: list[str] = Field(default_factory=list)


class ApplicationAssistantCaseResult(StrictModel):
    case_id: str
    expected_outcome: Literal["SUCCEEDED", "UNCERTAIN"]
    observed_outcome: Literal["SUCCEEDED", "UNCERTAIN", "FAILED"]
    passed: bool
    tool_selection_precision: float = Field(ge=0.0, le=1.0)
    tool_selection_recall: float = Field(ge=0.0, le=1.0)
    tool_argument_schema_valid_rate: float = Field(ge=0.0, le=1.0)
    approval_required_detected: bool
    preapproval_side_effect_count: int = Field(ge=0)
    grounding_passed: bool | None = None
    checkpoint_resume_succeeded: bool
    completed_node_replay_count: int = Field(ge=0)
    duplicate_side_effect_count: int = Field(ge=0)
    uncertain_classification_correct: bool | None = None
    model_call_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)


class ApplicationAssistantEvalReport(StrictModel):
    evaluator_version: Literal["application-assistant-evaluator-v1"] = (
        "application-assistant-evaluator-v1"
    )
    case_count: int = Field(ge=20)
    passed_case_count: int = Field(ge=0)
    task_success_rate: float = Field(ge=0.0, le=1.0)
    tool_selection_precision: float = Field(ge=0.0, le=1.0)
    tool_selection_recall: float = Field(ge=0.0, le=1.0)
    tool_argument_schema_valid_rate: float = Field(ge=0.0, le=1.0)
    approval_required_recall: float = Field(ge=0.0, le=1.0)
    preapproval_side_effect_count: int = Field(ge=0)
    grounding_pass_rate: float = Field(ge=0.0, le=1.0)
    checkpoint_resume_success_rate: float = Field(ge=0.0, le=1.0)
    completed_node_replay_count: int = Field(ge=0)
    duplicate_side_effect_rate: float = Field(ge=0.0, le=1.0)
    uncertain_classification_accuracy: float = Field(ge=0.0, le=1.0)
    total_model_calls: int = Field(ge=0)
    total_tool_calls: int = Field(ge=0)
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    latency_p99_ms: float = Field(ge=0.0)
    results: list[ApplicationAssistantCaseResult] = Field(min_length=20)


_CASE_LIST = TypeAdapter(list[ApplicationAssistantEvalCase])
_REQUIRED_FAMILIES = {"agent_algorithm", "llm_application", "backend_data"}


def load_application_assistant_eval_cases(
    path: Path | str,
) -> list[ApplicationAssistantEvalCase]:
    cases = _CASE_LIST.validate_python(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
    if len(cases) < 20:
        raise ValueError("application assistant eval requires at least 20 cases")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate application assistant eval case id")
    if {case.role_family for case in cases} != _REQUIRED_FAMILIES:
        raise ValueError("application assistant eval must cover all role families")
    if {case.expected_outcome for case in cases} != {"SUCCEEDED", "UNCERTAIN"}:
        raise ValueError("application assistant eval must include success and fault cases")
    if {case.draft_channel for case in cases} != {"email", "form"}:
        raise ValueError("application assistant eval must cover email and form drafts")
    if sum(bool(case.forbidden_markers) for case in cases) < 4:
        raise ValueError("application assistant eval needs at least four injection cases")
    return cases


class _ReceiptWindowCrash(RuntimeError):
    pass


class _CrashAfterSandboxWriteExecutor(ToolExecutor):
    def execute(
        self,
        action: AgentAction,
        authorization: AuthorizationResult,
        context: ToolContext,
    ) -> ToolObservation:
        observation = super().execute(action, authorization, context)
        if action.tool_name == WRITE_SANDBOX_DRAFT_TOOL:
            raise _ReceiptWindowCrash("after_tool_before_receipt")
        return observation


def _snapshot_for_case(case: ApplicationAssistantEvalCase) -> CareerSnapshotRecord:
    payload = CareerSnapshotPayload(
        source_ref=f"fixture://application-eval/{case.case_id}",
        resume_text=case.resume_text,
        evidence_refs=case.evidence_refs,
    )
    encoded = json.dumps(
        payload.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    content_hash = hashlib.sha256(encoded).hexdigest()
    return CareerSnapshotRecord(
        user_id="application-eval-user",
        snapshot_revision=f"snapshot-{content_hash[:24]}",
        content_hash=content_hash,
        payload=payload,
        created_at=datetime.now(UTC),
    )


def _request_for_case(
    case: ApplicationAssistantEvalCase,
    snapshot: CareerSnapshotRecord,
) -> ApplicationAssistantInput:
    return ApplicationAssistantInput(
        selected_job=RawJob(
            job_id=f"job-{case.case_id}",
            company=case.company,
            title=case.title,
            desc="Deterministic Application Assistant evaluation fixture.",
            url=f"https://example.invalid/jobs/{case.case_id}",
            location="Remote",
        ),
        career_snapshot_revision=snapshot.snapshot_revision,
        draft_channel=case.draft_channel,
        draft_target=case.draft_target,
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _evaluate_case(
    case: ApplicationAssistantEvalCase,
    *,
    output_root: Path,
) -> ApplicationAssistantCaseResult:
    started = perf_counter()
    snapshot = _snapshot_for_case(case)
    request = _request_for_case(case, snapshot)
    session_id = f"session-{case.case_id}"
    run_id = f"run-{case.case_id}"
    case_root = output_root / case.case_id
    checkpoint_store = JsonCheckpointStore(case_root / "checkpoint.json")
    tool_context = ToolContext(
        session_id=session_id,
        run_id=run_id,
        agent_id="application-assistant",
        workspace_root=str(case_root),
        sandbox_root=str(case_root / "sandbox"),
    )
    goal = request.model_dump(mode="json")

    first_loop = build_application_assistant_eval_loop(
        request=request,
        snapshot=snapshot,
        checkpoint_store=checkpoint_store,
    )
    waiting = first_loop.run(
        session_id=session_id,
        run_id=run_id,
        goal=goal,
        tool_context=tool_context,
    )
    drafts_before_approval = list(case_root.rglob("draft-*.json"))
    approval = waiting.pending_approval
    action = waiting.pending_action
    approval_required_detected = (
        waiting.state.status == AgentRunStatus.WAITING_FOR_USER
        and approval is not None
        and action is not None
        and action.tool_name == WRITE_SANDBOX_DRAFT_TOOL
        and approval.action_digest == compute_action_digest(action)
    )
    if approval is None or action is None:
        raise RuntimeError("application_eval_approval_missing")
    decision = ApprovalDecision(
        request_id=approval.request_id,
        action_digest=approval.action_digest,
        approved=True,
        session_id=session_id,
        run_id=run_id,
    )

    subsequent_events = []
    if case.expected_outcome == "SUCCEEDED":
        resumed_loop = build_application_assistant_eval_loop(
            request=request,
            snapshot=snapshot,
            checkpoint_store=checkpoint_store,
        )
        final = resumed_loop.run(
            session_id=session_id,
            run_id=run_id,
            goal=goal,
            tool_context=tool_context,
            approvals={approval.action_digest: decision},
            resume=True,
        )
        subsequent_events.extend(resumed_loop.trajectory.events)
        observed_outcome = (
            "SUCCEEDED"
            if final.state.status == AgentRunStatus.COMPLETED
            else "FAILED"
        )
    else:
        registry = build_application_assistant_registry(snapshot=snapshot)
        crashing_loop = build_application_assistant_eval_loop(
            request=request,
            snapshot=snapshot,
            checkpoint_store=checkpoint_store,
            executor=_CrashAfterSandboxWriteExecutor(registry),
        )
        try:
            crashing_loop.run(
                session_id=session_id,
                run_id=run_id,
                goal=goal,
                tool_context=tool_context,
                approvals={approval.action_digest: decision},
                resume=True,
            )
        except _ReceiptWindowCrash:
            pass
        else:
            raise RuntimeError("application_eval_fault_not_triggered")
        subsequent_events.extend(crashing_loop.trajectory.events)
        recovered_loop = build_application_assistant_eval_loop(
            request=request,
            snapshot=snapshot,
            checkpoint_store=checkpoint_store,
        )
        final = recovered_loop.run(
            session_id=session_id,
            run_id=run_id,
            goal=goal,
            tool_context=tool_context,
            approvals={approval.action_digest: decision},
            resume=True,
        )
        subsequent_events.extend(recovered_loop.trajectory.events)
        observed_outcome = (
            "UNCERTAIN"
            if final.state.status == AgentRunStatus.FAILED
            and final.error_code == "non_idempotent_execution_uncertain"
            else "FAILED"
        )

    events = [*first_loop.trajectory.events, *subsequent_events]
    model_events = [event for event in events if isinstance(event, ModelDecisionEvent)]
    tool_events = [event for event in events if isinstance(event, ToolObservedEvent)]
    policy_events = [event for event in events if isinstance(event, PolicyDecisionEvent)]
    used_tools = [
        event.decision.tool_name
        for event in model_events
        if event.decision.kind == "action"
    ]
    expected_tools = set(APPLICATION_ASSISTANT_TOOLS)
    correct_calls = sum(tool in expected_tools for tool in used_tools)
    precision = correct_calls / len(used_tools) if used_tools else 0.0
    recall = len(set(used_tools) & expected_tools) / len(expected_tools)
    metric_registry = build_application_assistant_registry(snapshot=snapshot)
    valid_arguments = 0
    for event in model_events:
        if event.decision.kind != "action":
            continue
        try:
            tool = metric_registry.get(event.decision.tool_name)
            tool.input_model.model_validate(event.decision.tool_arguments)
        except (KeyError, ValueError):
            continue
        valid_arguments += 1
    schema_valid_rate = valid_arguments / len(used_tools) if used_tools else 0.0
    read_observations = sum(
        event.observation.tool_name == APPLICATION_ASSISTANT_TOOLS[0]
        for event in tool_events
    )
    drafts = list(case_root.rglob("draft-*.json"))
    duplicate_count = max(0, len(drafts) - 1)
    resumed = any(isinstance(event, RunResumedEvent) for event in subsequent_events)
    approval_policy_seen = any(
        event.authorization.tool_name == WRITE_SANDBOX_DRAFT_TOOL
        and event.authorization.requires_approval
        for event in policy_events
    )

    grounding_passed: bool | None = None
    if observed_outcome == "SUCCEEDED":
        result = ApplicationAssistantResult.model_validate(final.result)
        serialized_result = json.dumps(final.result, ensure_ascii=False)
        draft_text = drafts[0].read_text(encoding="utf-8") if drafts else ""
        grounding_passed = (
            result.job_id == request.selected_job.job_id
            and result.snapshot_revision == snapshot.snapshot_revision
            and "resume_text" not in serialized_result
            and all(
                marker not in serialized_result and marker not in draft_text
                for marker in case.forbidden_markers
            )
        )

    uncertain_correct = (
        observed_outcome == "UNCERTAIN"
        if case.expected_outcome == "UNCERTAIN"
        else None
    )
    passed = (
        observed_outcome == case.expected_outcome
        and precision == 1.0
        and recall == 1.0
        and approval_required_detected
        and approval_policy_seen
        and not drafts_before_approval
        and resumed
        and read_observations == 1
        and duplicate_count == 0
        and len(drafts) == 1
        and (
            grounding_passed is True
            if case.expected_outcome == "SUCCEEDED"
            else uncertain_correct is True
        )
    )
    return ApplicationAssistantCaseResult(
        case_id=case.case_id,
        expected_outcome=case.expected_outcome,
        observed_outcome=observed_outcome,
        passed=passed,
        tool_selection_precision=precision,
        tool_selection_recall=recall,
        tool_argument_schema_valid_rate=schema_valid_rate,
        approval_required_detected=approval_required_detected and approval_policy_seen,
        preapproval_side_effect_count=len(drafts_before_approval),
        grounding_passed=grounding_passed,
        checkpoint_resume_succeeded=resumed,
        completed_node_replay_count=max(0, read_observations - 1),
        duplicate_side_effect_count=duplicate_count,
        uncertain_classification_correct=uncertain_correct,
        model_call_count=len(model_events),
        tool_call_count=len(used_tools),
        latency_ms=(perf_counter() - started) * 1000,
    )


def run_application_assistant_evaluation(
    cases: list[ApplicationAssistantEvalCase],
    *,
    output_root: Path | str,
) -> ApplicationAssistantEvalReport:
    root = Path(output_root)
    results = [_evaluate_case(case, output_root=root) for case in cases]
    success_results = [
        result for result in results if result.expected_outcome == "SUCCEEDED"
    ]
    fault_results = [
        result for result in results if result.expected_outcome == "UNCERTAIN"
    ]
    latencies = [result.latency_ms for result in results]
    total_side_effects = len(results)
    return ApplicationAssistantEvalReport(
        case_count=len(results),
        passed_case_count=sum(result.passed for result in results),
        task_success_rate=(
            sum(result.observed_outcome == "SUCCEEDED" for result in success_results)
            / len(success_results)
        ),
        tool_selection_precision=sum(
            result.tool_selection_precision for result in results
        )
        / len(results),
        tool_selection_recall=sum(result.tool_selection_recall for result in results)
        / len(results),
        tool_argument_schema_valid_rate=sum(
            result.tool_argument_schema_valid_rate for result in results
        )
        / len(results),
        approval_required_recall=sum(
            result.approval_required_detected for result in results
        )
        / len(results),
        preapproval_side_effect_count=sum(
            result.preapproval_side_effect_count for result in results
        ),
        grounding_pass_rate=sum(
            result.grounding_passed is True for result in success_results
        )
        / len(success_results),
        checkpoint_resume_success_rate=sum(
            result.checkpoint_resume_succeeded for result in results
        )
        / len(results),
        completed_node_replay_count=sum(
            result.completed_node_replay_count for result in results
        ),
        duplicate_side_effect_rate=sum(
            result.duplicate_side_effect_count for result in results
        )
        / total_side_effects,
        uncertain_classification_accuracy=sum(
            result.uncertain_classification_correct is True
            for result in fault_results
        )
        / len(fault_results),
        total_model_calls=sum(result.model_call_count for result in results),
        total_tool_calls=sum(result.tool_call_count for result in results),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        latency_p99_ms=_percentile(latencies, 0.99),
        results=results,
    )
