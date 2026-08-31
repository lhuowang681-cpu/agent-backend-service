from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from job_agent.schemas import ApplicationStatus
from job_agent.tools.application_tracker import (
    ApplicationTrackerStore,
)
from job_agent.ui.application_status import status_label, workbench_next_statuses
from job_agent.ui.latex_preview import compile_latex_preview
from job_agent.ui.interview_prep_view import load_interview_prep_questions
from job_agent.ui.live_action import execute_live_action
from job_agent.ui.local_config import load_local_runtime_defaults
from job_agent.ui.modes import (
    WorkbenchMode,
    build_live_runtime,
    check_live_runtime,
)
from job_agent.ui.new_session_wizard import render_new_session_wizard
from job_agent.ui.refill import run_refill
from job_agent.ui.resume_match import (
    clean_proof,
    match_counts,
    presentation_for,
    user_facing_resume_markdown,
)
from job_agent.ui.resume_source import (
    build_targeted_resume_draft,
    load_resume_source_metadata,
)
from job_agent.ui.session_view import (
    StepState,
    WorkbenchStep,
    archive_session,
    build_session_view,
    infer_output_root,
    list_sessions,
    load_ui_runtime_settings,
)
from job_agent.ui.run_diagnostics import (
    load_run_diagnostic,
)


_STATE_ICONS = {
    StepState.BLOCKED: "🔒",
    StepState.READY: "→",
    StepState.IN_PROGRESS: "●",
    StepState.COMPLETED: "✓",
    StepState.STALE: "!",
}

_MODE_LABELS = {
    WorkbenchMode.AGENT_API_LIVE.value: "真实模型模式",
}


def _set_active_workbench_step(step: str) -> None:
    st.session_state["active_workbench_step"] = step


def _render_queued_action_error(key: str) -> None:
    payload = st.session_state.pop(key, None)
    if not isinstance(payload, dict):
        return
    st.error(str(payload.get("message") or "操作暂时失败，请重试。"))
    st.caption(
        f"错误类型：{payload.get('error_code') or 'unexpected_error'}。"
        f"{payload.get('caption_suffix') or '可在页面顶部“运行诊断”查看脱敏记录。'}"
    )


def _queue_action_error(
    key: str,
    *,
    message: str | None,
    error_code: str | None,
    caption_suffix: str = "",
) -> None:
    st.session_state[key] = {
        "message": message or "操作暂时失败，请重试。",
        "error_code": error_code or "unexpected_error",
        "caption_suffix": caption_suffix,
    }
    st.rerun()


def _latest_session(output_root: Path) -> Path | None:
    sessions = list_sessions(output_root)
    return sessions[0] if sessions else None


def _active_session() -> Path | None:
    raw = st.session_state.get("active_session_dir")
    if not raw and st.session_state.get("session_dir"):
        raw = st.session_state["session_dir"]
        st.session_state["active_session_dir"] = raw
    if raw:
        path = Path(raw)
        if path.exists():
            st.session_state["active_output_root"] = str(infer_output_root(path))
            return path
    output_root = Path(st.session_state.get("active_output_root", "output/workbench"))
    latest = _latest_session(output_root)
    if latest is not None:
        st.session_state["active_session_dir"] = str(latest)
        st.session_state["session_dir"] = str(latest)
    return latest


def _session_label(session_dir: Path) -> str:
    try:
        state = json.loads((session_dir / "session_state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {}
    company = state.get("company") or "未知公司"
    title = state.get("title") or session_dir.name
    location = _session_location(session_dir)
    return f"{company} · {title}" + (f" · {location}" if location else "")


def _session_location(session_dir: Path) -> str | None:
    try:
        selected_job = json.loads(
            (session_dir / "selected_job.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    location = str(selected_job.get("location", "")).strip()
    return location if location and location != "unknown" else None


def _select_session(session_dir: Path, output_root: Path) -> Path:
    sessions = list_sessions(output_root)
    if session_dir not in sessions:
        sessions.insert(0, session_dir)
    options = [str(path) for path in sessions]
    current = str(session_dir)
    selected = st.selectbox(
        "当前岗位",
        options,
        index=options.index(current),
        format_func=lambda value: _session_label(Path(value)),
        key="workbench_session_selector",
        label_visibility="collapsed",
    )
    selected_path = Path(selected)
    if selected_path != session_dir:
        st.session_state["active_session_dir"] = str(selected_path)
        st.session_state["session_dir"] = str(selected_path)
        st.session_state["active_output_root"] = str(infer_output_root(selected_path))
        st.session_state.pop("active_workbench_step", None)
        st.rerun()
    return selected_path


def _build_runtime_for_action(settings: dict[str, str]):
    defaults = load_local_runtime_defaults(
        skill_root_override=settings.get("skill_root"),
    )
    if not defaults.skill_root:
        raise RuntimeError("Live runtime not ready; missing skill_root")
    return build_live_runtime(
        skill_root=defaults.skill_root,
        base_url=settings["base_url"],
        model=settings["model"],
    )


def _render_live_key_control(settings: dict[str, str]) -> None:
    defaults = load_local_runtime_defaults(
        skill_root_override=settings.get("skill_root"),
    )
    readiness = check_live_runtime(
        skill_root=defaults.skill_root,
        base_url=settings["base_url"],
        model=settings["model"],
    )
    if readiness.ready:
        st.success(
            f"实时 API 已就绪（{defaults.api_key_source}），后续生成环节自动复用。"
        )
    else:
        st.warning(
            readiness.message
            + "。生成操作已阻止；请在启动应用前设置环境变量，系统不会回退到离线或 mock。"
        )


def _render_run_diagnostics(view) -> None:
    diagnostic_path = view.session_dir / ".internal" / "ui_run_diagnostics.json"
    if not diagnostic_path.is_file():
        diagnostic_path = (
            view.session_dir / ".internal" / "agent_runtime_audit.json"
        )
    payload = load_run_diagnostic(diagnostic_path)
    with st.expander("运行诊断"):
        if not payload:
            st.info("这个岗位由旧版本创建，暂时没有可读取的运行诊断。")
            return
        status = str(payload.get("status") or "completed")
        readable_status = {
            "completed": "操作成功",
            "failed": "操作失败",
            "running": "处理中",
        }.get(status, status)
        duration_ms = int(payload.get("duration_ms") or 0)
        traces = payload.get("traces", [])
        traces = traces if isinstance(traces, list) else []
        metrics = st.columns(4)
        metrics[0].metric("最近状态", readable_status)
        metrics[1].metric(
            "总耗时",
            f"{duration_ms / 1000:.1f} 秒" if duration_ms else "旧记录未统计",
        )
        metrics[2].metric(
            "调用尝试",
            int(
                payload.get("attempt_count")
                or payload.get("llm_call_count")
                or len(traces)
            ),
        )
        metrics[3].metric(
            "结构修复",
            int(payload.get("schema_repair_count") or 0),
        )
        action_labels = {
            "create_session": "创建岗位流程",
            "regenerate_resume": "重新生成简历建议",
            "regenerate_interview_prep": "重新生成面试材料",
            "start_ops_transition": "验证投递状态迁移",
            "approve_ops_transition": "处理投递状态审批",
        }
        action_id = str(payload.get("action_id") or "")
        if action_id:
            st.caption(
                f"最近动作：{action_labels.get(action_id, action_id)}"
            )
        if payload.get("error_code"):
            st.error(f"失败类型：{payload['error_code']}")
        if payload.get("previous_version_preserved"):
            st.info("本次更新失败，当前仍展示上一成功版本。")
        if traces:
            stage_labels = {
                "jd-structurer": "解析岗位要求",
                "jd-analysis": "解析岗位要求",
                "evidence-mapping": "匹配简历依据",
                "evidence-contract": "匹配简历依据",
                "resume-tailoring": "生成岗位版简历",
                "interview-prep": "生成重点题目",
                "interview-grilling": "生成面试材料",
                "answer-cards": "生成作答提示",
                "mock-interview": "生成模拟面试题库",
            }
            rows = []
            for trace in traces:
                if not isinstance(trace, dict):
                    continue
                skill_id = str(trace.get("skill_id") or "")
                rows.append(
                    {
                        "环节": stage_labels.get(skill_id, skill_id or "未知环节"),
                        "耗时（毫秒）": trace.get("latency_ms", 0),
                        "结构校验": "通过" if trace.get("schema_valid") else "失败",
                        "错误类型": trace.get("error_code") or "",
                        "输入 Token": trace.get("input_tokens"),
                        "输出 Token": trace.get("output_tokens"),
                    }
                )
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            "诊断只记录阶段、耗时、Token 数和错误类型，不记录 API Key、完整简历、"
            "原始提示词或模型原始输出。"
        )
        st.download_button(
            "下载脱敏诊断 JSON",
            data=diagnostic_path.read_bytes(),
            file_name=f"{view.session_id}_diagnostics.json",
            mime="application/json",
            key=f"download_diagnostics::{view.ui_scope}",
        )


def _render_session_management(view) -> None:
    with st.expander("岗位管理"):
        st.caption(
            "删除后会移入项目的 .trash/sessions 回收区，不会永久清除；"
            "投递追踪中的独立记录也不会被删除。"
        )
        confirmed = st.checkbox(
            "我确认移除当前岗位及其本地准备材料",
            key=f"archive_confirm::{view.ui_scope}",
        )
        if st.button(
            "移至回收区",
            disabled=not confirmed,
            key=f"archive_session::{view.ui_scope}",
        ):
            archived_path = archive_session(view.session_dir, view.output_root)
            remaining = list_sessions(view.output_root)
            if remaining:
                st.session_state["active_session_dir"] = str(remaining[0])
                st.session_state["session_dir"] = str(remaining[0])
            else:
                st.session_state.pop("active_session_dir", None)
                st.session_state.pop("session_dir", None)
            st.session_state.pop("active_workbench_step", None)
            st.session_state["session_archive_notice"] = (
                f"岗位已移至回收区：{archived_path.name}"
            )
            st.rerun()


def _render_next_action(view, active_step: WorkbenchStep) -> None:
    action = view.recommended_action
    with st.container(border=True):
        col_text, col_action = st.columns([3, 1])
        with col_text:
            st.caption("推荐下一步")
            st.markdown(f"### {action.title}")
            st.write(action.description)
        with col_action:
            st.write("")
            st.write("")
            if (
                active_step != action.target_step
                and st.button(
                    f"前往{view.get_step(action.target_step).label}",
                    type="primary",
                    use_container_width=True,
                )
            ):
                st.session_state["active_workbench_step"] = action.target_step.value
                st.rerun()


def _render_resume(view, settings: dict[str, str]) -> None:
    st.subheader("岗位匹配与简历修改")
    if (view.session_dir / "evidence_current.json").is_file():
        from job_agent.ui.views.evidence_v2_view import render_evidence_v2_view

        render_evidence_v2_view(view.session_dir)
    resume_error_key = f"resume_error::{view.ui_scope}"
    _render_queued_action_error(resume_error_key)
    notice = st.session_state.pop(f"resume_notice::{view.ui_scope}", None)
    if notice:
        st.success(notice)

    jd_path = view.session_dir / "01_jd_structured.json"
    evidence_path = view.session_dir / "02_evidence_mapping.json"
    resume_path = view.session_dir / "06_targeted_resume.md"
    draft_path = view.session_dir / "06_targeted_resume_draft.md"
    normalized_path = view.session_dir / "00_resume_normalized.md"
    metadata = load_resume_source_metadata(view.session_dir)

    try:
        jd = json.loads(jd_path.read_text(encoding="utf-8"))
        evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        jd = {}
        evidence_payload = []

    evidence_items = (
        evidence_payload.get("items", [])
        if isinstance(evidence_payload, dict)
        else evidence_payload
    )
    evidence_items = evidence_items if isinstance(evidence_items, list) else []
    evidence_by_requirement = {
        str(item.get("requirement_id", "")): item
        for item in evidence_items
        if isinstance(item, dict)
    }
    counts = match_counts(
        item.get("level", "None")
        for item in evidence_items
        if isinstance(item, dict)
    )

    original_path: Path | None = None
    metadata_path = metadata.get("original_path")
    if isinstance(metadata_path, str):
        candidate = view.session_dir / Path(metadata_path).name
        if candidate.is_file():
            original_path = candidate
    if original_path is None:
        legacy_path = view.session_dir / "00_original_resume.md"
        if legacy_path.is_file():
            original_path = legacy_path
    if not normalized_path.is_file() and original_path is not None:
        normalized_path = original_path

    original_tab, match_tab, revised_tab = st.tabs(
        ["原始简历", "岗位匹配", "修改后简历"]
    )

    with original_tab:
        st.markdown("#### 创建该岗位时使用的简历")
        if original_path is None or not normalized_path.is_file():
            st.info(
                "这是旧版创建的岗位记录，当时没有在该岗位目录中保存原始简历全文。"
                "匹配页仍会展示创建时保存的依据；如需完整对照，请重新创建岗位流程。"
            )
        else:
            display_name = str(metadata.get("filename") or original_path.name)
            st.caption(
                f"文件：{display_name} · 格式：{original_path.suffix.lstrip('.').upper()}"
            )
            normalized_text = normalized_path.read_text(encoding="utf-8")
            if original_path.suffix.casefold() == ".tex":
                preview = compile_latex_preview(original_path)
                if preview.success:
                    st.success("LaTeX 已在本机安全编译，下面是第一页预览。")
                    if preview.png_path is not None:
                        st.image(str(preview.png_path), use_column_width=True)
                else:
                    st.warning(
                        f"{preview.message} 文字版简历仍可用于岗位匹配，不影响后续流程。"
                    )
                    if preview.diagnostic:
                        st.error(f"编译失败原因：{preview.diagnostic}")
                    if preview.log_path is not None and preview.log_path.is_file():
                        st.download_button(
                            "下载 LaTeX 编译日志",
                            data=preview.log_path.read_bytes(),
                            file_name=f"{original_path.stem}.log",
                            mime="text/plain",
                            key=f"resume_latex_log::{view.ui_scope}",
                        )

                download_columns = st.columns(2)
                with download_columns[0]:
                    st.download_button(
                        "下载 LaTeX 源文件",
                        data=original_path.read_bytes(),
                        file_name=Path(display_name).name,
                        mime="text/x-tex",
                        use_container_width=True,
                        key=f"resume_source_download::{view.ui_scope}",
                    )
                with download_columns[1]:
                    if preview.pdf_path is not None:
                        st.download_button(
                            "下载 PDF",
                            data=preview.pdf_path.read_bytes(),
                            file_name=f"{original_path.stem}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"resume_pdf_download::{view.ui_scope}",
                        )
                with st.expander("查看用于匹配的文字版简历", expanded=not preview.success):
                    st.markdown(normalized_text)
            else:
                with st.container(border=True):
                    st.markdown(normalized_text)
                mime = (
                    "text/markdown"
                    if original_path.suffix.casefold() == ".md"
                    else "text/plain"
                )
                st.download_button(
                    "下载原始简历",
                    data=original_path.read_bytes(),
                    file_name=Path(display_name).name,
                    mime=mime,
                    key=f"resume_source_download::{view.ui_scope}",
                )

    with match_tab:
        st.markdown("#### JD 要求与简历依据")
        if not jd or not evidence_items:
            st.warning("当前岗位缺少可读取的 JD 或简历匹配数据。")
        col_supported, col_partial, col_missing = st.columns(3)
        col_supported.metric("可以直接使用", counts["supported"])
        col_partial.metric("需要谨慎表达", counts["partial"])
        col_missing.metric("简历中未找到", counts["missing"])

        with st.expander("系统是怎么匹配的？", expanded=True):
            st.markdown(
                "1. 先把 JD 拆成一条条必须要求。\n"
                "2. 再为每条要求寻找简历中的对应依据。\n"
                "3. 根据原文是否包含实际行动、项目产物和量化结果，判断证据是否充分。\n\n"
                "页面只展示中文结论。内部等级编码不会写进你的简历，也不会直接决定录用结果。"
            )
            if settings["mode"] == WorkbenchMode.OFFLINE_RULE.value:
                st.caption(
                    "当前为本地快速模式：主要按关键词、行动词、项目产物和量化结果匹配，"
                    "可能漏掉同义表达，最终请以你对真实经历的确认作为准则。"
                )
            else:
                st.caption(
                    "当前为 Agent 模式：模型负责结构化理解并返回依据摘要；"
                    "摘要不等于逐字引用，最终需要与“原始简历”页签人工核对。"
                )

        requirements = jd.get("must_have", []) if isinstance(jd, dict) else []
        if not requirements:
            st.info("当前 JD 没有解析出明确的必须要求。")
        for index, requirement in enumerate(requirements, start=1):
            requirement_id = str(requirement.get("id", ""))
            item = evidence_by_requirement.get(requirement_id, {})
            level = item.get("level", "None")
            presentation = presentation_for(level)
            with st.container(border=True):
                col_requirement, col_status = st.columns([4, 1])
                with col_requirement:
                    st.markdown(
                        f"**{index}. {requirement.get('text', '未命名要求')}**"
                    )
                with col_status:
                    st.markdown(f"**{presentation.label}**")
                st.markdown(
                    f"**简历依据：** {clean_proof(str(item.get('proof', '')))}"
                )
                st.caption(f"判断说明：{presentation.explanation}")
                st.write(f"修改建议：{presentation.advice}")

    with revised_tab:
        st.markdown("#### 完整岗位版简历草稿")
        st.caption(
            "草稿保留原始简历正文，只突出已有证据支持的岗位重点；"
            "投递前请逐条确认，不会把证据不足的要求伪装成真实经历。"
        )
        draft_text = (
            draft_path.read_text(encoding="utf-8")
            if draft_path.is_file()
            else build_targeted_resume_draft(view.session_dir)
        )
        if draft_text:
            with st.container(border=True):
                st.markdown(draft_text)
            st.download_button(
                "下载修改后简历（Markdown）",
                data=draft_text.encode("utf-8"),
                file_name=f"{view.company}_{view.title}_定向简历.md",
                mime="text/markdown",
                key=f"targeted_resume_download::{view.ui_scope}",
            )
        else:
            st.info(
                "当前岗位没有保存可用于生成完整草稿的原始简历。"
                "请重新创建岗位流程并导入简历。"
            )

        st.markdown("#### 修改依据与建议")
        if resume_path.exists():
            with st.expander("查看本次修改建议", expanded=not bool(draft_text)):
                st.markdown(
                    user_facing_resume_markdown(
                        resume_path.read_text(encoding="utf-8")
                    )
                )
        else:
            st.info("简历修改建议尚未生成。")
            if st.button(
                "生成简历修改建议",
                type="primary",
                key=f"resume_generate::{view.ui_scope}",
            ):
                with st.spinner("正在生成简历修改建议…"):
                    run_refill(view.session_dir, "继续改简历")
                st.rerun()

        feedback = st.text_area(
            "补充修改要求（可选）",
            placeholder="例如：优先突出模型评测；没有量化结果的内容保持谨慎。",
            key=f"resume_feedback::{view.ui_scope}",
        )
        if resume_path.exists() and st.button(
            "重新生成修改建议",
            type="primary",
            key=f"resume_regenerate::{view.ui_scope}",
        ):
            from job_agent.ui.resume_loop import revise_resume_live, revise_resume_offline

            mode = WorkbenchMode.AGENT_API_LIVE
            if mode == WorkbenchMode.AGENT_API_LIVE:
                runtime_box: dict[str, object] = {}

                def regenerate_resume():
                    runtime = _build_runtime_for_action(settings)
                    runtime_box["runtime"] = runtime
                    return revise_resume_live(
                        view.session_dir,
                        feedback,
                        runtime,
                    )

                protected_artifacts = [
                    resume_path,
                    draft_path,
                    view.session_dir / "07_interview_grilling.md",
                    view.session_dir / "08_answer_cards.md",
                    view.session_dir / "09_mock_interview_plan.md",
                    view.session_dir / "10_mock_interview_live.md",
                    view.session_dir / "mock_answers.json",
                    view.session_dir / "15_mock_interview_debrief.md",
                    view.session_dir / "session_state.json",
                ]
                with st.spinner("简历修改 Agent 正在生成建议…"):
                    action_result = execute_live_action(
                        action_id="regenerate_resume",
                        operation=regenerate_resume,
                        session_dir=view.session_dir,
                        harness=lambda: getattr(
                            runtime_box.get("runtime"),
                            "harness",
                            None,
                        ),
                        current_stage="resume_tailoring",
                        previous_artifacts=protected_artifacts,
                    )
                if not action_result.ok:
                    _queue_action_error(
                        resume_error_key,
                        message=action_result.user_message,
                        error_code=action_result.error_code,
                    )
            else:
                with st.spinner("正在重新生成简历修改建议…"):
                    revise_resume_offline(view.session_dir, feedback)
            st.session_state[f"resume_notice::{view.ui_scope}"] = (
                "简历修改建议和完整草稿已更新；原有面试材料需要重新生成，"
                "避免继续使用旧版本。"
            )
            st.rerun()


def _render_interview_prep(view) -> None:
    step = view.get_step(WorkbenchStep.INTERVIEW_PREP)
    st.subheader("面试准备")
    prep_error_key = f"interview_prep_error::{view.ui_scope}"
    _render_queued_action_error(prep_error_key)
    st.write(step.summary)
    if step.state in {
        StepState.READY,
        StepState.STALE,
        StepState.COMPLETED,
    }:
        label = (
            "生成面试材料"
            if step.state == StepState.READY
            else "重新生成面试材料"
        )
        if step.state == StepState.COMPLETED:
            st.caption(
                "需要更新题目或把旧版规则作答卡升级为模型生成内容时，可重新生成。"
                "如果题目发生变化，当前模拟面试进度会失效。"
            )
        if st.button(label, type="primary"):
            settings = load_ui_runtime_settings(view.session_dir)
            mode = WorkbenchMode.AGENT_API_LIVE
            if mode == WorkbenchMode.AGENT_API_LIVE:
                from job_agent.ui.interview_prep_loop import (
                    regenerate_interview_prep_live,
                )

                runtime_box: dict[str, object] = {}
                stage_box = {"value": "interview_prep"}
                status = st.status("正在生成面试准备材料", expanded=True)

                def update_progress(stage: str) -> None:
                    stage_box["value"] = stage
                    labels = {
                        "interview_prep": "1/2 正在生成重点题目",
                        "answer_cards": "2/2 正在生成中文作答提示",
                    }
                    status.update(label=labels.get(stage, stage))

                def regenerate_prep():
                    runtime = _build_runtime_for_action(settings)
                    runtime_box["runtime"] = runtime
                    return regenerate_interview_prep_live(
                        view.session_dir,
                        runtime,
                        progress_callback=update_progress,
                    )

                protected_artifacts = [
                    view.session_dir / filename
                    for filename in (
                        "07_interview_grilling.md",
                        "08_answer_cards.md",
                        "09_mock_interview_plan.md",
                        "10_mock_interview_live.md",
                        "mock_answers.json",
                        "15_mock_interview_debrief.md",
                        "session_state.json",
                    )
                ]
                action_result = execute_live_action(
                    action_id="regenerate_interview_prep",
                    operation=regenerate_prep,
                    session_dir=view.session_dir,
                    harness=lambda: getattr(
                        runtime_box.get("runtime"),
                        "harness",
                        None,
                    ),
                    current_stage=lambda: stage_box["value"],
                    previous_artifacts=protected_artifacts,
                )
                if not action_result.ok:
                    status.update(
                        label="面试准备材料生成失败",
                        state="error",
                    )
                    _queue_action_error(
                        prep_error_key,
                        message=(
                            "旧版面试材料没有被覆盖。"
                            f"{action_result.user_message}"
                        ),
                        error_code=action_result.error_code,
                    )
                status.update(
                    label="面试准备材料已更新",
                    state="complete",
                    expanded=False,
                )
            else:
                with st.spinner("正在生成重点题目、作答提示和模拟面试流程…"):
                    result = run_refill(view.session_dir, "面试准备")
                if result.executed:
                    st.success("面试准备材料已更新。")
                else:
                    st.warning(result.message)
            st.rerun()

    questions = load_interview_prep_questions(view.session_dir)
    if not questions:
        st.info("还没有可展示的面试题。请先生成面试准备材料。")
        return

    guide_tab, questions_tab, answers_tab, mock_tab = st.tabs(
        ["怎么使用", "重点题目", "作答提示", "模拟面试怎么进行"]
    )
    with guide_tab:
        st.markdown(
            """
#### 建议按这个顺序使用

1. 先看“重点题目”，理解面试官为什么会问，以及最可能继续追问什么。
2. 再看“作答提示”，只使用简历中确实存在的经历组织 60 秒回答。
3. 最后进入“模拟面试”，系统会一次只问一道题，全部回答后统一做规则证据检查。

这里展示的是面试训练材料，不是新的求职流程，也不会修改你的简历。
"""
        )
        st.info(
            f"当前岗位已准备 {len(questions)} 道重点题。"
            "建议先完整看一遍，再开始模拟面试。"
        )

    with questions_tab:
        for index, item in enumerate(questions, start=1):
            with st.expander(f"{index}. {item.question}", expanded=index == 1):
                if item.intent:
                    st.markdown("**面试官想考什么**")
                    st.write(item.intent)
                if item.risk:
                    st.markdown("**你需要提前补强的地方**")
                    st.write(item.risk)
                if item.follow_ups:
                    st.markdown("**可能继续追问**")
                    for follow_up in item.follow_ups:
                        st.write(f"- {follow_up}")

    with answers_tab:
        st.caption(
            "这些内容是回答框架和事实边界，不是可直接背诵的标准答案。"
        )
        for index, item in enumerate(questions, start=1):
            with st.expander(f"{index}. {item.question}", expanded=index == 1):
                if item.answer_lead:
                    st.markdown("**回答切入点**")
                    st.write(item.answer_lead)
                if item.supporting_evidence:
                    st.markdown("**简历中可以使用的真实依据**")
                    st.write(item.supporting_evidence)
                if item.truth_boundary:
                    st.markdown("**不要越过的事实边界**")
                    st.warning(item.truth_boundary)
                if item.practice_actions:
                    st.markdown("**本题练习动作**")
                    for action in item.practice_actions:
                        st.write(f"- {action}")

    with mock_tab:
        st.markdown(
            """
#### 模拟面试是什么？

它只是当前岗位的一轮正式答题练习：系统从上面的重点题中依次出题，你输入回答，
Live 模式在整轮结束后调用一次 LLM 做五维语义评价；确定性规则只做辅助审计。
完成后可在“面试复盘”查看整轮结果。

离线模式不会伪装成 AI 评分，只保存 transcript 和规则证据审计。

你不需要理解内部题目编号或证据代码，也不需要重新输入 JD 和简历。
"""
        )
        st.button(
            "开始一轮模拟面试",
            type="primary",
            key=f"prep_to_mock::{view.ui_scope}",
            on_click=_set_active_workbench_step,
            args=(WorkbenchStep.MOCK_INTERVIEW.value,),
        )


def _render_mock_interview(view, settings: dict[str, str]) -> None:
    from job_agent.ui.mock_interview_loop import (
        MockInterviewRuntimeError,
        answer_question,
        restore_mock_interview_state,
        start_mock_interview,
    )

    st.subheader("模拟面试")
    if view.get_step(WorkbenchStep.MOCK_INTERVIEW).state in {
        StepState.BLOCKED,
        StepState.STALE,
    }:
        st.warning(view.get_step(WorkbenchStep.MOCK_INTERVIEW).summary)
        return

    state_key = f"mock_state::{view.ui_scope}"
    error_key = f"mock_error::{view.ui_scope}"
    interview_state = st.session_state.get(state_key)
    checkpoint_path = view.session_dir / "interview-checkpoint.json"
    if (
        interview_state is not None
        and not interview_state.finished
        and not checkpoint_path.exists()
    ):
        st.session_state.pop(state_key, None)
        interview_state = None
        st.warning(
            "上一轮模拟面试异常中断，答题断点已经失效。"
            "请重新开始一轮；已完成的历史轮次不会受影响。"
        )
    if interview_state is None and checkpoint_path.exists():
        interview_state = restore_mock_interview_state(view.session_dir)
        if interview_state is not None:
            st.session_state[state_key] = interview_state
        else:
            st.warning(
                "检测到无法恢复的旧答题断点。直接点击“开始模拟面试”"
                "即可建立一个新的答题轮次。"
            )
    mock_error = st.session_state.pop(error_key, None)
    if mock_error:
        st.error(mock_error)

    st.caption(
        "“面试准备”保存题库和答题方向；“模拟面试”一次只展示一道题。"
        "Live 模式会把当前回答作为 observation，由面试官 Agent 决定追问"
        "还是切换主题；全部回答完成后才统一评价。离线模式按已批准题库"
        "逐题推进，断点和历史同样会保存。"
    )
    st.info(
        "Live 整轮完成后只调用一次 LLM 做相关性、技术深度、推理表达、"
        "证据充分度和事实边界评价；规则层只保留可复现审计信号。"
    )

    max_questions = st.number_input(
        "题目数量",
        min_value=1,
        max_value=10,
        value=3,
        key=f"mock_max_questions::{view.ui_scope}",
    )
    start_label = "重新开始一轮" if interview_state is not None else "开始模拟面试"
    restart_confirmed = True
    restart_confirm_key = f"mock_restart_confirm::{view.ui_scope}"
    restart_reset_key = f"mock_restart_reset::{view.ui_scope}"
    if st.session_state.pop(restart_reset_key, False):
        st.session_state[restart_confirm_key] = False
    if interview_state is not None and not interview_state.finished:
        restart_confirmed = st.checkbox(
            "我确认放弃当前未完成的答题进度",
            key=restart_confirm_key,
        )
    if st.button(
        start_label,
        type="primary",
        disabled=not restart_confirmed,
    ):
        try:
            with st.spinner("正在启动模拟面试…"):
                runtime = _build_runtime_for_action(settings)
                interview_state = start_mock_interview(
                    view.session_dir,
                    max_questions=int(max_questions),
                    mode=WorkbenchMode.AGENT_API_LIVE.value,
                    runtime=runtime,
                )
        except MockInterviewRuntimeError as exc:
            st.session_state[error_key] = (
                "模拟面试启动失败，系统已保留现有材料。"
                f"错误代码：{exc.error_code}。请重新开始；"
                "如果仍然失败，可在“运行诊断”中查看详情。"
            )
            st.rerun()
        except Exception:
            st.session_state[error_key] = (
                "模拟面试启动失败，系统已保留现有材料。"
                "请检查模型凭据和服务配置后重试；"
                "页面不会展示本机路径或模型原始响应。"
            )
            st.rerun()
        else:
            st.session_state[state_key] = interview_state
            st.session_state[restart_reset_key] = True
            st.rerun()

    if (
        interview_state is not None
        and not interview_state.finished
        and interview_state.current_question is not None
    ):
        question = interview_state.current_question
        total_questions = (
            getattr(interview_state, "total_questions", 0)
            or int(max_questions)
        )
        is_last_question = (
            interview_state.question_index + 1 >= total_questions
        )
        with st.container(border=True):
            st.caption(
                f"第 {interview_state.question_index + 1} / "
                f"{total_questions} 题"
            )
            st.markdown(f"### {question.prompt}")
            st.caption(
                f"重点：{question.focus}"
                + (
                    f" · 风险：{', '.join(question.risk_flags)}"
                    if question.risk_flags
                    else ""
                )
            )
            with st.form(
                f"mock_answer_form::{view.ui_scope}::{question.question_id}",
                clear_on_submit=False,
            ):
                answer = st.text_area(
                    "你的答案",
                    height=180,
                    key=f"mock_answer::{view.ui_scope}::{question.question_id}",
                )
                elapsed = st.number_input(
                    "用时（秒）",
                    min_value=0,
                    value=0,
                    key=f"mock_elapsed::{view.ui_scope}::{question.question_id}",
                )
                submitted = st.form_submit_button(
                    (
                        "提交最后回答并生成整轮评价"
                        if is_last_question
                        else (
                            "提交回答，让面试官决定下一题"
                            if interview_state.adaptive
                            else "保存答案，下一题"
                        )
                    ),
                    type="primary",
                )
            if submitted:
                if not answer.strip():
                    st.error("请先填写答案。")
                else:
                    try:
                        spinner_text = (
                            "正在保存完整记录并生成整轮 AI 评价…"
                            if is_last_question
                            else (
                                "面试官正在根据当前回答生成下一题…"
                                if interview_state.adaptive
                                else "正在保存答案并进入下一题…"
                            )
                        )
                        with st.spinner(spinner_text):
                            runtime = _build_runtime_for_action(settings)
                            interview_state = answer_question(
                                view.session_dir,
                                interview_state,
                                answer=answer,
                                elapsed_seconds=int(elapsed),
                                max_questions=int(max_questions),
                                mode=WorkbenchMode.AGENT_API_LIVE.value,
                                runtime=runtime,
                            )
                    except MockInterviewRuntimeError as exc:
                        st.session_state[error_key] = (
                            "本次回答或整轮评价未能提交，系统仍停在当前题，"
                            "已完成的答题进度不变。"
                            f"错误代码：{exc.error_code}。请重试提交；"
                            "如果仍然失败，可重新开始一轮。"
                        )
                        st.rerun()
                    except Exception:
                        st.session_state[error_key] = (
                            "本次回答后的下一题生成失败，系统仍停在当前题。"
                            "请检查模型服务后重新提交；不会丢失此前答案，"
                            "也不会生成假分数。"
                        )
                        st.rerun()
                    else:
                        st.session_state[state_key] = interview_state
                        st.rerun()

    if interview_state is not None and interview_state.history:
        st.markdown("#### 本轮已答")
        for item in interview_state.history:
            st.write(
                f"**{item['question_id']} · 已保存**  "
                + (
                    "整轮已完成。"
                    if interview_state.finished
                    else "完成全部题目后统一评价。"
                )
            )
    if interview_state is not None and interview_state.finished:
        from job_agent.schemas import (
            MockInterviewEvaluationResult,
            RuleEvidenceAudit,
        )
        from job_agent.ui.interview_review import (
            list_run_rounds,
            load_run_payload,
        )
        from job_agent.ui.mock_interview_evaluation_view import (
            render_mock_ai_evaluation,
            render_rule_audits,
        )

        result = interview_state.final_result
        st.success("本轮完成，答题记录已保存。")
        raw_evaluation = result.get("ai_evaluation")
        if isinstance(raw_evaluation, dict):
            render_mock_ai_evaluation(
                MockInterviewEvaluationResult.model_validate(raw_evaluation)
            )
        elif result.get("evaluation_error_code"):
            st.warning(
                "本轮 transcript 和规则审计已保存，但 AI 评价生成失败。"
                "请检查模型服务后在面试日志中重新生成；页面不会展示 traceback。"
            )
        else:
            st.info("当前为离线模式，本轮尚无 AI 语义评价。")
        rounds = list_run_rounds(view.session_dir)
        payload = load_run_payload(rounds[0].path) if rounds else None
        if payload:
            audits = [
                RuleEvidenceAudit.model_validate(item)
                for item in payload.get("rule_evidence_audit", [])
            ]
            if audits:
                render_rule_audits(audits)


def _render_application(view, settings: dict[str, str]) -> None:
    from job_agent.ui.application_ops_loop import (
        add_application_to_tracker,
        approve_ops_action,
        start_ops_action,
    )

    st.subheader("投递进度")
    ops_error_key = f"ops_error::{view.ui_scope}"
    _render_queued_action_error(ops_error_key)
    output_root = view.output_root
    store = ApplicationTrackerStore(output_root / "tracker.json")
    jd_path = view.session_dir / "01_jd_structured.json"
    if not jd_path.exists():
        st.warning("当前 session 缺少岗位结构化数据。")
        return
    jd = json.loads(jd_path.read_text(encoding="utf-8"))
    selected_job_path = view.session_dir / "selected_job.json"
    selected_job = (
        json.loads(selected_job_path.read_text(encoding="utf-8"))
        if selected_job_path.exists()
        else {}
    )
    job_id = str(selected_job.get("job_id") or "")
    company = jd.get("company", "")
    title = jd.get("title", "")
    existing = None
    try:
        tracker = store.load(user_id="workbench")
        for application in tracker.applications:
            if job_id and application.job_id == job_id:
                existing = application
                break
            if (
                application.company.casefold().strip() == company.casefold().strip()
                and application.title.casefold().strip() == title.casefold().strip()
            ):
                existing = application
                break
    except Exception:
        pass

    state_key = f"ops_state::{view.ui_scope}"
    ops_state = st.session_state.get(state_key)
    if existing is None:
        st.info("当前岗位还没有加入投递追踪。")
        if st.button("添加到投递追踪", type="primary"):
            record = add_application_to_tracker(view.session_dir, output_root)
            st.success(f"已添加：{record.id}")
            st.rerun()
        return

    st.metric("当前状态", status_label(existing.current_state))
    st.caption(
        "状态只允许按当前阶段向前推进；选择“已结束”后，该岗位不再继续流转。"
    )
    if existing.state_history:
        with st.expander("查看状态记录"):
            for event in existing.state_history:
                timestamp = event.timestamp.replace("T", " ")[:16]
                note = f" · {event.notes}" if event.notes else ""
                st.markdown(
                    f"- {timestamp} · {status_label(event.state)}{note}"
                )
    if ops_state is None or not ops_state.pending_approval_request:
        allowed = workbench_next_statuses(existing.current_state)
        if not allowed:
            st.info("当前申请已到终态。")
        else:
            next_status = st.selectbox(
                "更新为",
                [None, *allowed],
                format_func=lambda status: (
                    "请选择下一状态" if status is None else status_label(status)
                ),
                key=f"ops_next_status::{view.ui_scope}",
            )
            if st.button(
                "提交状态更新",
                type="primary",
                disabled=next_status is None,
            ):
                if settings["mode"] == WorkbenchMode.AGENT_API_LIVE.value:
                    runtime_box: dict[str, object] = {}

                    def start_transition():
                        runtime = _build_runtime_for_action(settings)
                        runtime_box["runtime"] = runtime
                        candidate = start_ops_action(
                            view.session_dir,
                            output_root,
                            next_status,
                            mode=settings["mode"],
                            runtime=runtime,
                        )
                        if candidate.error:
                            raise RuntimeError(candidate.error)
                        return candidate

                    with st.spinner("正在验证状态迁移…"):
                        action_result = execute_live_action(
                            action_id="start_ops_transition",
                            operation=start_transition,
                            session_dir=view.session_dir,
                            harness=lambda: getattr(
                                runtime_box.get("runtime"),
                                "harness",
                                None,
                            ),
                            current_stage="application_ops",
                            previous_artifacts=[
                                view.session_dir / "ops-checkpoint.json",
                                output_root / "tracker.json",
                            ],
                        )
                    if not action_result.ok:
                        _queue_action_error(
                            ops_error_key,
                            message=action_result.user_message,
                            error_code=action_result.error_code,
                            caption_suffix="当前投递状态没有更新。",
                        )
                    ops_state = action_result.value
                else:
                    with st.spinner("正在验证状态迁移…"):
                        ops_state = start_ops_action(
                            view.session_dir,
                            output_root,
                            next_status,
                            mode=settings["mode"],
                            runtime=None,
                        )
                st.session_state[state_key] = ops_state
                st.rerun()
    elif ops_state.pending_approval_request is not None:
        request = ops_state.pending_approval_request
        target_status = ApplicationStatus(
            str(request.arguments_preview.get("new_status", ""))
        )
        st.warning("请确认投递状态更新")
        st.markdown(
            f"将 **{existing.company} · {existing.title}** "
            f"从 **{status_label(existing.current_state)}** "
            f"更新为 **{status_label(target_status)}**。"
        )
        st.caption("此操作只更新本地求职记录，不会替你向公司发送申请。")
        col_approve, col_reject = st.columns(2)
        with col_approve:
            if st.button("批准", type="primary", use_container_width=True):
                if settings["mode"] == WorkbenchMode.AGENT_API_LIVE.value:
                    runtime_box: dict[str, object] = {}

                    def approve_transition():
                        runtime = _build_runtime_for_action(settings)
                        runtime_box["runtime"] = runtime
                        candidate = approve_ops_action(
                            view.session_dir,
                            output_root,
                            ops_state,
                            approved=True,
                            mode=settings["mode"],
                            runtime=runtime,
                        )
                        if candidate.error:
                            raise RuntimeError(candidate.error)
                        return candidate

                    action_result = execute_live_action(
                        action_id="approve_ops_transition",
                        operation=approve_transition,
                        session_dir=view.session_dir,
                        harness=lambda: getattr(
                            runtime_box.get("runtime"),
                            "harness",
                            None,
                        ),
                        current_stage="application_ops_approval",
                        previous_artifacts=[
                            view.session_dir / "ops-checkpoint.json",
                            output_root / "tracker.json",
                        ],
                    )
                    if not action_result.ok:
                        _queue_action_error(
                            ops_error_key,
                            message=action_result.user_message,
                            error_code=action_result.error_code,
                            caption_suffix="审批仍保留，可直接重试。",
                        )
                    st.session_state[state_key] = action_result.value
                else:
                    st.session_state[state_key] = approve_ops_action(
                        view.session_dir,
                        output_root,
                        ops_state,
                        approved=True,
                        mode=settings["mode"],
                        runtime=None,
                    )
                st.rerun()
        with col_reject:
            if st.button("拒绝", use_container_width=True):
                if settings["mode"] == WorkbenchMode.AGENT_API_LIVE.value:
                    runtime_box: dict[str, object] = {}

                    def reject_transition():
                        runtime = _build_runtime_for_action(settings)
                        runtime_box["runtime"] = runtime
                        candidate = approve_ops_action(
                            view.session_dir,
                            output_root,
                            ops_state,
                            approved=False,
                            reason="User denied.",
                            mode=settings["mode"],
                            runtime=runtime,
                        )
                        if candidate.error:
                            raise RuntimeError(candidate.error)
                        return candidate

                    action_result = execute_live_action(
                        action_id="approve_ops_transition",
                        operation=reject_transition,
                        session_dir=view.session_dir,
                        harness=lambda: getattr(
                            runtime_box.get("runtime"),
                            "harness",
                            None,
                        ),
                        current_stage="application_ops_approval",
                        previous_artifacts=[
                            view.session_dir / "ops-checkpoint.json",
                            output_root / "tracker.json",
                        ],
                    )
                    if not action_result.ok:
                        _queue_action_error(
                            ops_error_key,
                            message=action_result.user_message,
                            error_code=action_result.error_code,
                            caption_suffix="审批仍保留，可直接重试。",
                        )
                    st.session_state[state_key] = action_result.value
                else:
                    st.session_state[state_key] = approve_ops_action(
                        view.session_dir,
                        output_root,
                        ops_state,
                        approved=False,
                        reason="User denied.",
                        mode=settings["mode"],
                        runtime=None,
                    )
                st.rerun()
    if ops_state is not None and ops_state.finished:
        if ops_state.error:
            st.error(ops_state.error)
        elif ops_state.final_result:
            st.success("投递状态已更新。")
        st.session_state.pop(state_key, None)


def _render_review(view) -> None:
    from job_agent.ui.interview_review import (
        build_question_model_answer_map,
        list_run_rounds,
        load_run_payload,
    )
    from job_agent.ui.mock_interview_evaluation_loop import (
        load_mock_interview_evaluation,
    )
    from job_agent.ui.mock_interview_evaluation_view import (
        render_mock_ai_evaluation,
        render_rule_audits,
    )
    from job_agent.ui.post_interview_review_view import (
        render_ai_review_result,
        render_post_interview_review,
    )
    from job_agent.ui.post_interview_review_loop import (
        generate_post_interview_review_live,
        load_post_interview_review_result,
        review_paths,
        write_next_interview_focus,
    )
    from job_agent.ui.real_interview_editor import (
        render_real_interview_question_editor,
        reset_real_interview_editor,
    )
    from job_agent.ui.real_interview_rounds import (
        REAL_INTERVIEW_STAGES,
        archive_legacy_real_interview,
        list_real_interview_rounds,
        save_real_interview_round,
        suggested_real_interview_stage,
    )

    st.subheader("面试复盘")
    settings = load_ui_runtime_settings(view.session_dir)
    review_error_key = f"post_interview_review_error::{view.ui_scope}"
    _render_queued_action_error(review_error_key)
    review_notice = st.session_state.pop(
        f"review_notice::{view.ui_scope}", None
    )
    if review_notice:
        st.success(review_notice)
    rounds = list_run_rounds(view.session_dir)
    post_review_path = view.session_dir / "13_post_interview_review.md"

    if rounds:
        labels = [
            f"模拟第 {item.round_number} 轮 · "
            f"{item.timestamp[:10]} {item.timestamp[11:13]}:{item.timestamp[13:15]}"
            + (
                f" · AI 五维总分 {item.average_score:.1f}"
                if item.has_ai_evaluation
                else (
                    f" · 历史规则基线 {item.average_score:.1f}"
                    if item.average_score > 0
                    else " · 尚无 AI 评价"
                )
            )
            for item in rounds
        ]
        selected_label = st.selectbox(
            "模拟面试轮次",
            labels,
            key=f"review_round::{view.ui_scope}",
        )
        selected = rounds[labels.index(selected_label)]
        payload = load_run_payload(selected.path)
        if payload:
            from job_agent.schemas import RuleEvidenceAudit

            evaluation = load_mock_interview_evaluation(selected.path)
            if evaluation is not None:
                render_mock_ai_evaluation(evaluation)
            elif payload.get("schema_version") == 2:
                if payload.get("evaluation_status") == "ai_failed":
                    st.warning(
                        "本轮 transcript 和规则证据审计已保存，"
                        "但 AI 评价生成失败。"
                    )
                else:
                    st.info(
                        "本轮只有 transcript 和规则证据审计，尚无 AI 语义评价。"
                    )
            else:
                result = payload.get("result", {})
                st.metric(
                    "历史规则基线",
                    f"{result.get('average_score', 0):.1f}/5",
                )
                st.caption(
                    "这是旧版本规则结果，不判断技术答案是否正确。"
                )
            model_answers = build_question_model_answer_map(view.session_dir)
            for index, item in enumerate(payload.get("history", []), start=1):
                qid = item.get("question_id", f"Q{index}")
                with st.expander(
                    f"{qid} · {item.get('prompt', '')[:60]}"
                ):
                    st.markdown(f"**你的答案**\n\n{item.get('answer', '')}")
                    if model_answers.get(qid):
                        st.markdown(f"**参考答案**\n\n{model_answers[qid]}")
                    if payload.get("schema_version") != 2:
                        st.markdown(
                            f"**旧版规则建议**\n\n"
                            f"{item.get('improvement', '—')}"
                        )
            audits = [
                RuleEvidenceAudit.model_validate(item)
                for item in payload.get("rule_evidence_audit", [])
            ]
            if audits:
                render_rule_audits(audits)
    else:
        st.info("当前岗位还没有已完成的模拟面试。")

    st.divider()
    st.markdown("#### 已记录的真实面试轮次")
    real_rounds = list_real_interview_rounds(view.session_dir)
    if real_rounds:
        real_labels = {
            item.round_id: (
                f"{item.stage} · {item.created_at[:10]} "
                f"{item.created_at[11:16]}"
            )
            for item in real_rounds
        }
        selected_real_id = st.selectbox(
            "选择真实面试轮次",
            list(real_labels),
            format_func=lambda round_id: real_labels[round_id],
            key=f"real_review_round::{view.ui_scope}",
        )
        selected_real = next(
            item for item in real_rounds if item.round_id == selected_real_id
        )
        render_post_interview_review(selected_real.path)
        ai_review = load_post_interview_review_result(
            view.session_dir,
            selected_real.round_id,
        )
        if ai_review is not None:
            render_ai_review_result(ai_review)
            if ai_review.next_mock_topics and st.button(
                "作为下一轮模拟面试重点",
                key=f"use_review_focus::{view.ui_scope}::{selected_real.round_id}",
            ):
                write_next_interview_focus(
                    view.session_dir,
                    round_id=selected_real.round_id,
                    topics=ai_review.next_mock_topics,
                )
                st.success("已保存；下一轮模拟面试会优先覆盖这些主题。")
        else:
            st.info("本轮目前只有面试记录，尚未生成 AI 复盘。")
        if settings["mode"] == WorkbenchMode.AGENT_API_LIVE.value:
            review_label = (
                "重新生成 AI 复盘"
                if ai_review is not None
                else "生成 AI 复盘"
            )
            if st.button(
                review_label,
                type="primary",
                key=f"generate_real_review::{view.ui_scope}::{selected_real.round_id}",
            ):
                runtime_box: dict[str, object] = {}

                def generate_review():
                    runtime = _build_runtime_for_action(settings)
                    runtime_box["runtime"] = runtime
                    return generate_post_interview_review_live(
                        view.session_dir,
                        selected_real.round_id,
                        runtime,
                    )

                json_review_path, markdown_review_path = review_paths(
                    view.session_dir,
                    selected_real.round_id,
                )
                with st.spinner("AI 正在对照岗位要求和简历证据生成复盘…"):
                    action_result = execute_live_action(
                        action_id="post_interview_review",
                        operation=generate_review,
                        session_dir=view.session_dir,
                        harness=lambda: getattr(
                            runtime_box.get("runtime"),
                            "harness",
                            None,
                        ),
                        current_stage="post_interview_review",
                        previous_artifacts=[
                            json_review_path,
                            markdown_review_path,
                            post_review_path,
                            view.session_dir / "session_state.json",
                        ],
                    )
                if not action_result.ok:
                    _queue_action_error(
                        review_error_key,
                        message=action_result.user_message,
                        error_code=action_result.error_code,
                        caption_suffix="原始面试记录和上一版 AI 复盘仍然保留。",
                    )
                st.session_state[f"review_notice::{view.ui_scope}"] = (
                    f"{selected_real.stage}的 AI 复盘已生成。"
                )
                st.rerun()
        else:
            st.caption(
                "当前 session 不是“真实模型模式”，可以保存和回看记录，"
                "但不会用本地规则冒充 AI 复盘。"
            )
    elif post_review_path.exists():
        st.caption("这是旧版保存的历史记录；新增下一轮时会自动归档。")
        render_post_interview_review(post_review_path)
    else:
        st.info("当前岗位还没有真实面试记录。")

    st.markdown("#### 新增一轮真实面试")
    st.caption("一面、二面、HR 面等会分别保存，不会覆盖之前的记录。")
    notes_key = f"real_interview_notes::{view.ui_scope}"
    stage_key = f"real_interview_stage::{view.ui_scope}"
    reset_key = f"real_interview_reset::{view.ui_scope}"
    if st.session_state.pop(reset_key, False):
        st.session_state.pop(notes_key, None)
        st.session_state.pop(stage_key, None)
        reset_real_interview_editor(view.ui_scope)
    suggested_stage = suggested_real_interview_stage(view.session_dir)
    stage = st.selectbox(
        "面试阶段",
        REAL_INTERVIEW_STAGES,
        index=REAL_INTERVIEW_STAGES.index(suggested_stage),
        key=stage_key,
    )
    notes = st.text_area(
        "记录问题、表现和反馈",
        placeholder="例如：一面重点追问 LoRA 评测、上线风险和个人负责范围。",
        key=notes_key,
    )
    question_records = render_real_interview_question_editor(view.ui_scope)
    if st.button("保存面试记录", type="primary"):
        if not notes.strip() and not question_records:
            st.error("请至少填写一条面试问题或一段面试记录。")
        elif notes.strip() and len(notes.strip()) < 10 and not question_records:
            st.error(
                "记录太短。请至少写清一道面试题、你的回答表现，"
                "以及面试官的反馈或追问。"
            )
        else:
            try:
                archive_legacy_real_interview(view.session_dir)
                combined_notes = notes.strip()
                if question_records:
                    question_summary = "\n".join(
                        (
                            f"问题：{item.question}\n"
                            f"我的回答：{item.answer or '未记录'}\n"
                            f"面试官追问：{item.interviewer_follow_up or '未记录'}\n"
                            f"面试官反馈：{item.interviewer_feedback or '未记录'}"
                        )
                        for item in question_records
                    )
                    combined_notes = "\n\n".join(
                        item
                        for item in (combined_notes, question_summary)
                        if item
                    )
                with st.spinner("正在保存本轮面试记录…"):
                    result = run_refill(
                        view.session_dir,
                        f"interview notes: {combined_notes}",
                    )
                if result.executed:
                    save_real_interview_round(
                        view.session_dir,
                        stage=stage,
                        notes=notes,
                        questions=question_records,
                    )
                    st.session_state[f"review_notice::{view.ui_scope}"] = (
                        f"{stage}已保存为独立轮次；可选择该轮生成 AI 复盘。"
                    )
                    st.session_state[reset_key] = True
                    st.rerun()
                else:
                    st.warning(result.message)
            except (OSError, ValueError) as exc:
                st.error(f"本轮面试记录保存失败：{exc}")


def render_workbench_page() -> None:
    archive_notice = st.session_state.pop("session_archive_notice", None)
    if archive_notice:
        st.success(archive_notice)
    if st.session_state.get("show_new_session_wizard"):
        render_new_session_wizard()
        if st.button("返回工作台"):
            st.session_state["show_new_session_wizard"] = False
            st.rerun()
        return

    session_dir = _active_session()
    if session_dir is None:
        render_new_session_wizard()
        return

    output_root = infer_output_root(session_dir)
    settings = load_ui_runtime_settings(session_dir)
    view = build_session_view(
        session_dir,
        output_root=output_root,
        mode=settings["mode"],
    )

    col_title, col_session, col_new = st.columns([3, 2, 1.4])
    with col_title:
        st.title("求职工作台")
        st.markdown(f"### {view.company} · {view.title}")
        location = _session_location(session_dir)
        if location:
            st.caption(f"地点：{location}")
    with col_session:
        st.caption("当前岗位")
        session_dir = _select_session(session_dir, output_root)
        st.caption("运行方式：实时模型（API）")
    with col_new:
        st.write("")
        if st.button("＋ 开始新流程", use_container_width=True):
            st.session_state["show_new_session_wizard"] = True
            st.rerun()

    _render_live_key_control(settings)
    _render_run_diagnostics(view)
    _render_session_management(view)

    visible_steps = view.steps
    step_values = [item.step.value for item in visible_steps]
    if st.session_state.get("active_workbench_step") not in step_values:
        st.session_state["active_workbench_step"] = view.recommended_action.target_step.value
    current_step = WorkbenchStep(st.session_state["active_workbench_step"])
    _render_next_action(view, current_step)

    st.caption(
        "流程状态："
        + " · ".join(
            f"{_STATE_ICONS[item.state]} {item.label}" for item in visible_steps
        )
    )
    label_by_value = {item.step.value: item.label for item in visible_steps}
    selected = st.radio(
        "工作流",
        step_values,
        format_func=lambda value: label_by_value[value],
        horizontal=True,
        key="active_workbench_step",
        label_visibility="collapsed",
    )
    selected_step = WorkbenchStep(selected)

    st.caption(view.get_step(selected_step).summary)
    if selected_step == WorkbenchStep.RESUME:
        _render_resume(view, settings)
    elif selected_step == WorkbenchStep.INTERVIEW_PREP:
        _render_interview_prep(view)
    elif selected_step == WorkbenchStep.MOCK_INTERVIEW:
        _render_mock_interview(view, settings)
    elif selected_step == WorkbenchStep.APPLICATION:
        _render_application(view, settings)
    elif selected_step == WorkbenchStep.REVIEW:
        _render_review(view)
