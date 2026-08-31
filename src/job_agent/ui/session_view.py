from __future__ import annotations

import json
import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from job_agent.atomic_io import atomic_write_json
from job_agent.tools.application_tracker import ApplicationTrackerStore
from job_agent.ui.application_status import status_label
from job_agent.ui.real_interview_rounds import list_real_interview_rounds


class WorkbenchStep(str, Enum):
    RESUME = "resume"
    INTERVIEW_PREP = "interview_prep"
    MOCK_INTERVIEW = "mock_interview"
    APPLICATION = "application"
    REVIEW = "review"


class StepState(str, Enum):
    BLOCKED = "blocked"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    STALE = "stale"


@dataclass(frozen=True)
class WorkbenchStepView:
    step: WorkbenchStep
    label: str
    state: StepState
    summary: str


@dataclass(frozen=True)
class RecommendedAction:
    action_id: str
    title: str
    description: str
    button_label: str
    target_step: WorkbenchStep


@dataclass(frozen=True)
class SessionView:
    session_dir: Path
    output_root: Path
    session_id: str
    ui_scope: str
    company: str
    title: str
    mode: str
    application_status: str | None
    steps: tuple[WorkbenchStepView, ...]
    recommended_action: RecommendedAction

    def get_step(self, step: WorkbenchStep) -> WorkbenchStepView:
        return next(item for item in self.steps if item.step == step)


_STEP_LABELS = {
    WorkbenchStep.RESUME: "岗位匹配",
    WorkbenchStep.INTERVIEW_PREP: "面试准备",
    WorkbenchStep.MOCK_INTERVIEW: "模拟面试",
    WorkbenchStep.APPLICATION: "投递进度",
    WorkbenchStep.REVIEW: "面试复盘",
}


def infer_output_root(session_dir: Path) -> Path:
    """从标准的 <output_root>/sessions/<session_id> 推导 output root。"""
    if session_dir.parent.name == "sessions":
        return session_dir.parent.parent
    return session_dir.parent


def list_sessions(output_root: Path) -> list[Path]:
    sessions_dir = Path(output_root) / "sessions"
    if not sessions_dir.exists():
        return []
    return sorted(
        (path for path in sessions_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def archive_session(session_dir: Path, output_root: Path) -> Path:
    """Move one valid workbench session to a recoverable local trash folder."""
    session_dir = Path(session_dir).resolve()
    output_root = Path(output_root).resolve()
    sessions_root = (output_root / "sessions").resolve()
    if session_dir.parent != sessions_root or not session_dir.is_dir():
        raise ValueError("只能删除当前工作台 sessions 目录中的有效岗位。")
    trash_root = output_root / ".trash" / "sessions"
    trash_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = trash_root / f"{session_dir.name}__{timestamp}"
    suffix = 1
    while destination.exists():
        destination = trash_root / f"{session_dir.name}__{timestamp}_{suffix}"
        suffix += 1
    shutil.move(str(session_dir), str(destination))
    return destination


def load_ui_runtime_settings(session_dir: Path) -> dict[str, str]:
    payload = _read_json(Path(session_dir) / ".workbench_ui.json")
    return {
        "mode": str(payload.get("mode") or "offline_rule"),
        "skill_root": str(payload.get("skill_root") or ""),
        "base_url": str(payload.get("base_url") or "https://open.bigmodel.cn/api/anthropic"),
        "model": str(payload.get("model") or "glm-4.7"),
    }


def write_ui_runtime_settings(session_dir: Path, settings: dict[str, str]) -> None:
    allowed = {"mode", "skill_root", "base_url", "model"}
    payload = {key: str(value) for key, value in settings.items() if key in allowed}
    atomic_write_json(Path(session_dir) / ".workbench_ui.json", payload)


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _freshness_status(session_state: dict, artifact_name: str) -> str | None:
    freshness = session_state.get("artifact_freshness", {})
    if not isinstance(freshness, dict):
        return None
    record = freshness.get(artifact_name)
    if not isinstance(record, dict):
        return None
    status = record.get("status")
    return status if isinstance(status, str) else None


def _artifact_state(
    session_dir: Path,
    session_state: dict,
    artifact_name: str,
    filename: str,
) -> tuple[bool, bool]:
    """返回 (exists_and_usable, explicitly_stale)。

    旧 session 可能没有 freshness 记录；此时保持文件存在即有效的兼容行为。
    """
    exists = (session_dir / filename).exists()
    freshness = _freshness_status(session_state, artifact_name)
    stale = freshness == "invalidated"
    return exists and not stale, stale


def _find_application_status(
    output_root: Path,
    *,
    job_id: str | None,
    company: str,
    title: str,
    user_id: str,
) -> str | None:
    if not company or not title:
        return None
    try:
        tracker = ApplicationTrackerStore(output_root / "tracker.json").load(user_id=user_id)
    except Exception:
        return None
    company_key = company.casefold().strip()
    title_key = title.casefold().strip()
    for application in tracker.applications:
        if job_id and application.job_id == job_id:
            return application.current_state.value
        if (
            application.company.casefold().strip() == company_key
            and application.title.casefold().strip() == title_key
        ):
            return application.current_state.value
    return None


def _build_steps(
    session_dir: Path,
    session_state: dict,
    *,
    application_status: str | None,
) -> tuple[WorkbenchStepView, ...]:
    jd_ready = (session_dir / "01_jd_structured.json").exists()
    evidence_ready = (session_dir / "02_evidence_mapping.json").exists()

    resume_ready, resume_stale = _artifact_state(
        session_dir, session_state, "targeted_resume", "06_targeted_resume.md"
    )
    if resume_stale:
        resume_state = StepState.STALE
        resume_summary = "目标简历已失效，需要重新生成。"
    elif resume_ready:
        resume_state = StepState.COMPLETED
        resume_summary = "目标简历已生成，可以继续修改。"
    elif jd_ready and evidence_ready:
        resume_state = StepState.READY
        resume_summary = "岗位和证据已准备，可以生成目标简历。"
    else:
        resume_state = StepState.BLOCKED
        resume_summary = "需要先完成岗位结构化和证据映射。"

    prep_artifacts = (
        ("interview_grilling", "07_interview_grilling.md"),
        ("answer_cards", "08_answer_cards.md"),
        ("mock_interview_plan", "09_mock_interview_plan.md"),
    )
    prep_states = [
        _artifact_state(session_dir, session_state, artifact_name, filename)
        for artifact_name, filename in prep_artifacts
    ]
    prep_has_stale = any(stale for _, stale in prep_states)
    prep_complete = all(usable for usable, _ in prep_states)
    if prep_has_stale:
        prep_state = StepState.STALE
        prep_summary = "简历或上游材料发生变化，需要重新生成面试材料。"
    elif prep_complete:
        prep_state = StepState.COMPLETED
        prep_summary = "重点题目、作答提示和模拟面试题库已准备。"
    elif resume_state == StepState.COMPLETED:
        prep_state = StepState.READY
        prep_summary = "目标简历已就绪，可以生成面试准备材料。"
    else:
        prep_state = StepState.BLOCKED
        prep_summary = "需要先完成有效的目标简历。"

    checkpoint_exists = (session_dir / "interview-checkpoint.json").exists()
    run_exists = any(session_dir.glob("10_mock_interview_run_*.json"))
    if prep_state == StepState.STALE:
        mock_state = StepState.STALE
        mock_summary = "面试材料已失效，重新准备后才能继续。"
    elif prep_state != StepState.COMPLETED:
        mock_state = StepState.BLOCKED
        mock_summary = "需要先完成面试准备。"
    elif checkpoint_exists:
        mock_state = StepState.IN_PROGRESS
        mock_summary = "存在未完成的模拟面试，可以从已保存的答题进度继续。"
    elif run_exists:
        mock_state = StepState.COMPLETED
        mock_summary = "已完成至少一轮模拟面试，可以查看复盘或再练一轮。"
    else:
        mock_state = StepState.READY
        mock_summary = "重点题目和作答提示已准备，可以开始第一轮模拟面试。"

    if not jd_ready:
        application_state = StepState.BLOCKED
        application_summary = "需要先建立岗位 session。"
    elif (session_dir / "ops-checkpoint.json").exists():
        application_state = StepState.IN_PROGRESS
        application_summary = "投递状态更新正在等待确认或恢复。"
    elif application_status is not None:
        terminal_states = {"accepted", "rejected", "abandoned", "closed"}
        application_state = (
            StepState.COMPLETED
            if application_status in terminal_states
            else StepState.IN_PROGRESS
        )
        application_summary = f"当前投递状态：{status_label(application_status)}。"
    else:
        application_state = StepState.READY
        application_summary = "尚未加入投递追踪。"

    review_ready, review_stale = _artifact_state(
        session_dir,
        session_state,
        "post_interview_review",
        "13_post_interview_review.md",
    )
    real_rounds = list_real_interview_rounds(session_dir)
    if real_rounds:
        review_state = StepState.COMPLETED
        review_summary = (
            f"已保存 {len(real_rounds)} 轮真实面试记录，"
            f"最近一轮：{real_rounds[0].stage}。"
        )
    elif review_stale:
        review_state = StepState.STALE
        review_summary = "真实面试复盘已失效，需要重新记录。"
    elif review_ready:
        review_state = StepState.COMPLETED
        review_summary = "已经保存真实面试复盘。"
    elif run_exists:
        review_state = StepState.READY
        review_summary = "模拟面试已完成，可以查看本轮表现或记录真实面试。"
    elif application_status in {
        "first_interview",
        "second_interview",
        "other_interview",
        "hr_interview",
        "offer",
        "accepted",
        "rejected",
    }:
        review_state = StepState.READY
        review_summary = "真实面试后可在这里记录问题、表现和反馈。"
    else:
        review_state = StepState.BLOCKED
        review_summary = "投递进入面试阶段后，可以在这里记录真实面试。"

    states = (
        (WorkbenchStep.RESUME, resume_state, resume_summary),
        (WorkbenchStep.INTERVIEW_PREP, prep_state, prep_summary),
        (WorkbenchStep.MOCK_INTERVIEW, mock_state, mock_summary),
        (WorkbenchStep.APPLICATION, application_state, application_summary),
        (WorkbenchStep.REVIEW, review_state, review_summary),
    )
    return tuple(
        WorkbenchStepView(
            step=step,
            label=_STEP_LABELS[step],
            state=state,
            summary=summary,
        )
        for step, state, summary in states
    )


def _recommend(steps: tuple[WorkbenchStepView, ...]) -> RecommendedAction:
    by_step = {item.step: item for item in steps}

    if by_step[WorkbenchStep.MOCK_INTERVIEW].state == StepState.IN_PROGRESS:
        return RecommendedAction(
            action_id="continue_mock_interview",
            title="继续模拟面试",
            description="上次答题进度已经保存，可以直接继续。",
            button_label="继续面试",
            target_step=WorkbenchStep.MOCK_INTERVIEW,
        )
    if by_step[WorkbenchStep.INTERVIEW_PREP].state == StepState.STALE:
        return RecommendedAction(
            action_id="refresh_interview_prep",
            title="重新生成面试材料",
            description="目标简历发生了变化，原有面试材料已失效。",
            button_label="前往重新准备",
            target_step=WorkbenchStep.INTERVIEW_PREP,
        )
    if by_step[WorkbenchStep.RESUME].state in {StepState.READY, StepState.STALE}:
        return RecommendedAction(
            action_id="prepare_resume",
            title="完善目标简历",
            description=by_step[WorkbenchStep.RESUME].summary,
            button_label="前往简历",
            target_step=WorkbenchStep.RESUME,
        )
    if by_step[WorkbenchStep.INTERVIEW_PREP].state == StepState.READY:
        return RecommendedAction(
            action_id="prepare_interview",
            title="生成面试准备材料",
            description=by_step[WorkbenchStep.INTERVIEW_PREP].summary,
            button_label="开始准备",
            target_step=WorkbenchStep.INTERVIEW_PREP,
        )
    if by_step[WorkbenchStep.MOCK_INTERVIEW].state == StepState.READY:
        return RecommendedAction(
            action_id="start_mock_interview",
            title="开始第一轮模拟面试",
            description=by_step[WorkbenchStep.MOCK_INTERVIEW].summary,
            button_label="开始模拟面试",
            target_step=WorkbenchStep.MOCK_INTERVIEW,
        )
    if by_step[WorkbenchStep.REVIEW].state == StepState.READY:
        return RecommendedAction(
            action_id="record_interview_review",
            title="记录真实面试",
            description=by_step[WorkbenchStep.REVIEW].summary,
            button_label="记录面试",
            target_step=WorkbenchStep.REVIEW,
        )
    return RecommendedAction(
        action_id="repeat_mock_interview",
        title="再练一轮模拟面试",
        description="主要材料已经齐备，可以通过新一轮练习继续提高。",
        button_label="再练一轮",
        target_step=WorkbenchStep.MOCK_INTERVIEW,
    )


def build_session_view(
    session_dir: Path,
    *,
    output_root: Path | None = None,
    mode: str = "offline_rule",
    user_id: str = "workbench",
) -> SessionView:
    session_dir = Path(session_dir)
    resolved_output_root = Path(output_root) if output_root is not None else infer_output_root(session_dir)
    session_state = _read_json(session_dir / "session_state.json")
    jd = _read_json(session_dir / "01_jd_structured.json")

    session_id = str(session_state.get("session_id") or session_dir.name)
    scope_hash = hashlib.sha256(
        str(session_dir.resolve()).encode("utf-8")
    ).hexdigest()[:12]
    ui_scope = f"{session_id}:{scope_hash}"
    company = str(session_state.get("company") or jd.get("company") or "未知公司")
    title = str(session_state.get("title") or jd.get("title") or "未知岗位")
    application_status = _find_application_status(
        resolved_output_root,
        job_id=str(session_state.get("job_id") or "") or None,
        company=company,
        title=title,
        user_id=user_id,
    )
    steps = _build_steps(
        session_dir,
        session_state,
        application_status=application_status,
    )
    return SessionView(
        session_dir=session_dir,
        output_root=resolved_output_root,
        session_id=session_id,
        ui_scope=ui_scope,
        company=company,
        title=title,
        mode=mode,
        application_status=application_status,
        steps=steps,
        recommended_action=_recommend(steps),
    )
