from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from typing import Any

from pydantic import ValidationError

from job_agent.agent_runtime.checkpoint import JsonCheckpointStore
from job_agent.agent_runtime.contracts import (
    AgentAction,
    AgentFinish,
    AgentModelContext,
    AgentModelResponse,
    AgentRunStatus,
    ApprovalDecision,
    ApprovalRequest,
    ToolObservationStatus,
)
from job_agent.agent_runtime.live_agent_model import ToolUseDecisionModel, build_live_agent_model
from job_agent.domain_agents.application_ops import (
    APPLICATION_OPS_TOOLS,
    ApplicationOpsAgent,
    ApplicationOpsGoal,
    ApplyTransitionOutput,
    TrackerTransitionProposal,
    build_application_ops_registry,
)
from job_agent.schemas import ApplicationRecord, ApplicationStatus, TrackerEvent
from job_agent.ui.modes import WorkbenchMode
from job_agent.tools.application_tracker import (
    ApplicationTrackerStore,
)


@dataclass(frozen=True)
class OpsState:
    application_id: str | None
    current_status: ApplicationStatus | None
    desired_status: ApplicationStatus | None
    pending_approval_request: ApprovalRequest | None
    action_digest: str | None
    finished: bool
    final_result: dict | None
    error: str | None


def _load_tracker_store(output_dir: Path) -> ApplicationTrackerStore:
    """载入跨 session 的 tracker store（output_dir/tracker.json）。"""
    return ApplicationTrackerStore(output_dir / "tracker.json")


def _find_application(
    store: ApplicationTrackerStore,
    user_id: str,
    company: str,
    title: str,
    job_id: str | None = None,
) -> ApplicationRecord | None:
    """优先按当前 session 的 job_id 定位，兼容旧记录的 company+title 匹配。"""
    tracker = store.load(user_id=user_id)
    company_key = company.casefold().strip()
    title_key = title.casefold().strip()
    for app in tracker.applications:
        if job_id and app.job_id == job_id:
            return app
        if (
            app.company.casefold().strip() == company_key
            and app.title.casefold().strip() == title_key
        ):
            return app
    return None


class OfflineOpsModel:
    """offline/mock 的确定性 model：read→validate→apply→finish。

    finish 阶段从 context.steps 读 observations 构造 ApplicationOpsResult（verifier
    强制 proposal/applied_record 与观察一致）。
    """

    def __init__(
        self,
        application_id: str,
        current_status: ApplicationStatus,
        desired_status: ApplicationStatus,
        expected_revision: int = 0,
    ) -> None:
        self.application_id = application_id
        self.current_status = current_status
        self.desired_status = desired_status
        self.expected_revision = expected_revision
        self._phase = "read"

    def decide(self, context: AgentModelContext) -> AgentModelResponse:
        # resume-aware：context 里已有 apply 完成记录则直接 finish
        if context.steps:
            has_apply = any(
                s.action.tool_name == "ops.apply_transition"
                and s.observation.status == ToolObservationStatus.OK
                for s in context.steps
            )
            if has_apply:
                self._phase = "finish"
        if self._phase == "read":
            self._phase = "validate"
            return AgentModelResponse(
                decision=AgentAction(
                    action_id="read-tracker",
                    tool_name="ops.read_tracker",
                    tool_arguments={"application_id": self.application_id},
                    expected_observation="tracker state loaded",
                    progress_claim="Read approved local tracker.",
                ),
                provider="offline", model="offline-ops",
            )
        if self._phase == "validate":
            self._phase = "apply"
            return AgentModelResponse(
                decision=AgentAction(
                    action_id="validate-transition",
                    tool_name="ops.validate_transition",
                    tool_arguments={
                        "application_id": self.application_id,
                        "new_status": self.desired_status.value,
                        "expected_revision": self.expected_revision,
                    },
                    expected_observation="transition validated",
                    progress_claim="Validate transition.",
                ),
                provider="offline", model="offline-ops",
            )
        if self._phase == "apply":
            self._phase = "finish"
            return AgentModelResponse(
                decision=AgentAction(
                    action_id="apply-transition",
                    tool_name="ops.apply_transition",
                    tool_arguments={
                        "application_id": self.application_id,
                        "new_status": self.desired_status.value,
                        "expected_revision": self.expected_revision,
                    },
                    expected_observation="transition applied",
                    progress_claim="Apply transition.",
                ),
                provider="offline", model="offline-ops",
            )
        # finish：从 steps 读 observations 构造 result
        proposal = self._collect_proposal(context.steps)
        applied = self._collect_applied(context.steps)
        result = {
            "proposal": proposal.model_dump(mode="json") if proposal else {
                "application_id": self.application_id,
                "from_status": self.current_status.value,
                "to_status": self.desired_status.value,
                "expected_revision": 0, "notes": "", "valid": True,
                "reason_code": "valid_transition",
            },
            "applied_record": applied.application.model_dump(mode="json") if applied else None,
            "tracker_revision": applied.tracker_revision if applied else None,
            "sandbox_receipts": [],
        }
        return AgentModelResponse(
            decision=AgentFinish(
                result=result,
                completion_evidence=[f"application:{self.application_id}"],
                confidence=0.8,
            ),
            provider="offline", model="offline-ops",
        )

    @staticmethod
    def _collect_proposal(steps: Sequence) -> TrackerTransitionProposal | None:
        for step in reversed(steps):
            if (
                step.action.tool_name == "ops.validate_transition"
                and step.observation.status == ToolObservationStatus.OK
            ):
                try:
                    return TrackerTransitionProposal.model_validate(step.observation.data)
                except ValidationError:
                    continue
        return None

    @staticmethod
    def _collect_applied(steps: Sequence) -> ApplyTransitionOutput | None:
        for step in reversed(steps):
            if (
                step.action.tool_name == "ops.apply_transition"
                and step.observation.status == ToolObservationStatus.OK
            ):
                try:
                    return ApplyTransitionOutput.model_validate(step.observation.data)
                except ValidationError:
                    continue
        return None


def _build_offline_ops_model(
    application_id: str,
    current_status: ApplicationStatus,
    desired_status: ApplicationStatus,
    expected_revision: int = 0,
) -> OfflineOpsModel:
    return OfflineOpsModel(
        application_id=application_id,
        current_status=current_status,
        desired_status=desired_status,
        expected_revision=expected_revision,
    )


def _build_live_model(*, runtime, store, user_id, application_id, session_id, run_id):
    registry = build_application_ops_registry(store, user_id=user_id)
    return build_live_agent_model(
        provider=runtime.provider,
        registry=registry,
        allowed_tools=APPLICATION_OPS_TOOLS,
        agent_id="application-ops",
        session_id=session_id,
        run_id=run_id,
    )


def _build_model(
    *,
    mode: str,
    runtime: Any,
    store: ApplicationTrackerStore,
    user_id: str,
    application_id: str,
    current_status: ApplicationStatus,
    desired_status: ApplicationStatus,
    session_id: str,
    expected_revision: int = 0,
) -> OfflineOpsModel | ToolUseDecisionModel:
    if mode == WorkbenchMode.AGENT_API_LIVE.value:
        if runtime is None:
            raise ValueError("runtime is required for live mode")
        return _build_live_model(
            runtime=runtime, store=store, user_id=user_id,
            application_id=application_id,
            session_id=session_id, run_id="ops-run-1",
        )
    return _build_offline_ops_model(
        application_id, current_status, desired_status,
        expected_revision=expected_revision,
    )


def add_application_to_tracker(
    session_dir: Path,
    output_dir: Path,
    *,
    user_id: str = "workbench",
) -> ApplicationRecord:
    """从 session_dir 读 selected_job + fit_verdict，创建 ApplicationRecord 到 tracker。

    tracker 跨 session（output_dir/tracker.json）。重复调用（同 company+title）不重复创建，
    改为更新已有 record（session_dir / fit_score / verdict 等）。
    """
    store = _load_tracker_store(output_dir)
    selected = json.loads((session_dir / "selected_job.json").read_text(encoding="utf-8"))
    jd = json.loads((session_dir / "01_jd_structured.json").read_text(encoding="utf-8"))
    fit = json.loads((session_dir / "03_fit_verdict.json").read_text(encoding="utf-8"))

    tracker = store.load(user_id=user_id)
    next_id = f"app-{len(tracker.applications) + 1:03d}"

    record = ApplicationRecord(
        id=next_id,
        job_id=selected.get("job_id", ""),
        company=jd.get("company", selected.get("company", "")),
        title=jd.get("title", selected.get("title", "")),
        city=selected.get("location", ""),
        url=selected.get("url", ""),
        source=selected.get("source", ""),
        current_state=ApplicationStatus.TO_APPLY,
        verdict=fit.get("verdict", "strong fit"),
        fit_score=fit.get("score"),
        session_dir=str(session_dir),
        notes="",
        state_history=[
            TrackerEvent(
                state=ApplicationStatus.TO_APPLY,
                timestamp=store.current_timestamp(),
                notes="加入投递管理",
            )
        ],
    )
    return store.add_application(user_id=user_id, application=record, user_confirmed=True)


def start_ops_action(
    session_dir: Path,
    output_dir: Path,
    desired_status: ApplicationStatus,
    *,
    user_id: str = "workbench",
    mode: str = "offline_rule",
    runtime=None,
) -> OpsState:
    """offline/live：执行一次 ApplicationOpsAgent run，推进到 approval gate 或 finish。

    返回 OpsState：若 approval 触发 → pending_approval_request + action_digest；
    若直接 finish（合法但 abnormal）→ finished=True + final_result。
    """
    store = _load_tracker_store(output_dir)
    jd = json.loads((session_dir / "01_jd_structured.json").read_text(encoding="utf-8"))
    selected = json.loads((session_dir / "selected_job.json").read_text(encoding="utf-8"))
    app = _find_application(
        store,
        user_id,
        jd.get("company", ""),
        jd.get("title", ""),
        selected.get("job_id"),
    )
    if app is None:
        return OpsState(
            application_id=None, current_status=None, desired_status=None,
            pending_approval_request=None, action_digest=None,
            finished=True, final_result=None,
            error="ApplicationRecord not found — use '添加追踪' first.",
        )

    tracker_revision = store.load(user_id=user_id).revision
    model = _build_model(
        mode=mode, runtime=runtime, store=store, user_id=user_id,
        application_id=app.id, current_status=app.current_state,
        desired_status=desired_status,
        session_id=f"workbench-ops:{session_dir.name}",
        expected_revision=tracker_revision,
    )
    ops_checkpoint = JsonCheckpointStore(session_dir / "ops-checkpoint.json")
    agent = ApplicationOpsAgent(
        model=model, tracker_store=store, user_id=user_id,
        checkpoint_store=ops_checkpoint,
    )
    goal = ApplicationOpsGoal(
        application_id=app.id, desired_status=desired_status,
    )
    run_result = agent.run(
        goal=goal,
        session_id=f"workbench-ops:{session_dir.name}",
        run_id="ops-run-1",
        workspace_root=session_dir,
        sandbox_root=output_dir / "sandbox",
    )
    if run_result.state.status == AgentRunStatus.WAITING_FOR_USER and run_result.pending_approval is not None:
        return OpsState(
            application_id=app.id,
            current_status=app.current_state,
            desired_status=desired_status,
            pending_approval_request=run_result.pending_approval,
            action_digest=run_result.pending_approval.action_digest,
            finished=False,
            final_result=None,
            error=None,
        )
    if run_result.state.status == AgentRunStatus.COMPLETED:
        return OpsState(
            application_id=app.id,
            current_status=app.current_state,
            desired_status=desired_status,
            pending_approval_request=None,
            action_digest=None,
            finished=True,
            final_result=run_result.result,
            error=None,
        )
    return OpsState(
        application_id=app.id,
        current_status=app.current_state,
        desired_status=desired_status,
        pending_approval_request=None,
        action_digest=None,
        finished=True,
        final_result=None,
        error=f"agent ended with {run_result.state.status.value}: {run_result.error_code}",
    )


def approve_ops_action(
    session_dir: Path,
    output_dir: Path,
    state: OpsState,
    *,
    approved: bool,
    reason: str = "",
    user_id: str = "workbench",
    mode: str = "offline_rule",
    runtime=None,
) -> OpsState:
    """用户批准/拒绝后 resume ApplicationOpsAgent run。"""
    if state.pending_approval_request is None or state.action_digest is None:
        return OpsState(
            application_id=state.application_id,
            current_status=state.current_status,
            desired_status=state.desired_status,
            pending_approval_request=None,
            action_digest=None,
            finished=True,
            final_result=None,
            error="No pending approval to resolve.",
        )
    store = _load_tracker_store(output_dir)
    jd = json.loads((session_dir / "01_jd_structured.json").read_text(encoding="utf-8"))
    selected = json.loads((session_dir / "selected_job.json").read_text(encoding="utf-8"))
    app = _find_application(
        store,
        user_id,
        jd.get("company", ""),
        jd.get("title", ""),
        selected.get("job_id"),
    )
    if app is None:
        return OpsState(
            application_id=state.application_id,
            current_status=state.current_status,
            desired_status=state.desired_status,
            pending_approval_request=None,
            action_digest=None,
            finished=True,
            final_result=None,
            error="ApplicationRecord not found.",
        )

    tracker_revision = store.load(user_id=user_id).revision
    model = _build_model(
        mode=mode, runtime=runtime, store=store, user_id=user_id,
        application_id=app.id, current_status=app.current_state,
        desired_status=state.desired_status,
        session_id=f"workbench-ops:{session_dir.name}",
        expected_revision=tracker_revision,
    )
    ops_checkpoint = JsonCheckpointStore(session_dir / "ops-checkpoint.json")
    agent = ApplicationOpsAgent(
        model=model, tracker_store=store, user_id=user_id,
        checkpoint_store=ops_checkpoint,
    )
    goal = ApplicationOpsGoal(
        application_id=app.id,
        desired_status=state.desired_status,
    )
    approval = ApprovalDecision(
        request_id=state.pending_approval_request.request_id,
        action_digest=state.action_digest,
        approved=approved,
        reason=reason,
        session_id=f"workbench-ops:{session_dir.name}",
        run_id="ops-run-1",
    )
    run_result = agent.run(
        goal=goal,
        session_id=f"workbench-ops:{session_dir.name}",
        run_id="ops-run-1",
        workspace_root=session_dir,
        sandbox_root=output_dir / "sandbox",
        approvals={state.action_digest: approval},
        resume=True,
    )
    if run_result.state.status == AgentRunStatus.COMPLETED:
        return OpsState(
            application_id=app.id,
            current_status=app.current_state,
            desired_status=state.desired_status,
            pending_approval_request=None,
            action_digest=None,
            finished=True,
            final_result=run_result.result,
            error=None,
        )
    return OpsState(
        application_id=app.id,
        current_status=app.current_state,
        desired_status=state.desired_status,
        pending_approval_request=None,
        action_digest=None,
        finished=True,
        final_result=run_result.result if run_result.result else None,
        error=run_result.error_code or f"agent ended with {run_result.state.status.value}",
    )
