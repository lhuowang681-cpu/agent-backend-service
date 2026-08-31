from __future__ import annotations

from pathlib import Path

import streamlit as st

from job_agent.career import CareerStore, CompanyDraft
from job_agent.career import JobDraft as CareerJobDraft
from job_agent.domain_agents.opportunity_research import OpportunityResearchGoal
from job_agent.ui.jd_pool import (
    EXPLICIT_JD_DELIMITER,
    JobDraft,
    build_jobs_from_drafts,
    parse_explicit_jd_import,
)
from job_agent.ui.live_action import execute_live_action
from job_agent.ui.local_config import load_local_runtime_defaults
from job_agent.ui.modes import WorkbenchMode, build_runtime, check_live_runtime
from job_agent.ui.research import run_research
from job_agent.ui.resume_source import (
    build_pasted_resume_source,
    build_resume_source,
)
from job_agent.ui.run_diagnostics import (
    write_run_diagnostic,
)
from job_agent.ui.runner import run_initial_v2_live
from job_agent.ui.session_view import write_ui_runtime_settings

_DRAFT_IDS_KEY = "wizard_job_draft_ids"
_DRAFT_COUNTER_KEY = "wizard_job_draft_counter"


def _draft_widget_key(field: str, draft_id: str) -> str:
    return f"wizard_job_{field}::{draft_id}"


def _ensure_job_drafts() -> list[str]:
    if _DRAFT_IDS_KEY not in st.session_state:
        st.session_state[_DRAFT_IDS_KEY] = ["1"]
        st.session_state[_DRAFT_COUNTER_KEY] = 1
    first_id = str(st.session_state[_DRAFT_IDS_KEY][0])
    prefill = {
        "company": st.session_state.get("career_company_name", ""),
        "title": st.session_state.get("career_job_title", ""),
        "text": st.session_state.get("career_prefill_jd", ""),
    }
    for field, value in prefill.items():
        key = _draft_widget_key(field, first_id)
        if value and key not in st.session_state:
            st.session_state[key] = value
    return list(st.session_state[_DRAFT_IDS_KEY])


def _add_job_draft() -> None:
    counter = int(st.session_state.get(_DRAFT_COUNTER_KEY, 1)) + 1
    st.session_state[_DRAFT_COUNTER_KEY] = counter
    ids = list(st.session_state.get(_DRAFT_IDS_KEY, ["1"]))
    ids.append(str(counter))
    st.session_state[_DRAFT_IDS_KEY] = ids


def _delete_job_draft(draft_id: str) -> None:
    ids = list(st.session_state.get(_DRAFT_IDS_KEY, ["1"]))
    if len(ids) <= 1:
        return
    st.session_state[_DRAFT_IDS_KEY] = [item for item in ids if item != draft_id]
    for field in ("company", "title", "location", "text"):
        st.session_state.pop(_draft_widget_key(field, draft_id), None)


def _import_job_drafts() -> None:
    segments = parse_explicit_jd_import(
        str(st.session_state.get("wizard_bulk_jd_import", ""))
    )
    if not segments:
        return
    ids: list[str] = []
    counter = int(st.session_state.get(_DRAFT_COUNTER_KEY, 0))
    for segment in segments:
        counter += 1
        draft_id = str(counter)
        ids.append(draft_id)
        st.session_state[_draft_widget_key("text", draft_id)] = segment
    st.session_state[_DRAFT_COUNTER_KEY] = counter
    st.session_state[_DRAFT_IDS_KEY] = ids


def _render_job_draft_editor() -> list[JobDraft]:
    draft_ids = _ensure_job_drafts()
    drafts: list[JobDraft] = []
    for index, draft_id in enumerate(draft_ids, start=1):
        with st.container(border=True):
            header, remove = st.columns([5, 1])
            with header:
                st.markdown(f"#### 岗位 {index}")
            with remove:
                st.button(
                    "删除",
                    key=f"wizard_delete_job::{draft_id}",
                    disabled=len(draft_ids) <= 1,
                    on_click=_delete_job_draft,
                    args=(draft_id,),
                    use_container_width=True,
                )
            company_col, title_col, location_col = st.columns(3)
            with company_col:
                company = st.text_input(
                    "公司名补充（可选）",
                    key=_draft_widget_key("company", draft_id),
                )
            with title_col:
                title = st.text_input(
                    "岗位名称（可选）",
                    key=_draft_widget_key("title", draft_id),
                )
            with location_col:
                location = st.text_input(
                    "地点补充（可选）",
                    placeholder="例如：北京、上海、远程",
                    key=_draft_widget_key("location", draft_id),
                )
            jd_text = st.text_area(
                "岗位 JD",
                height=260,
                placeholder="粘贴这个岗位的完整 JD；正文中的空行会被完整保留。",
                key=_draft_widget_key("text", draft_id),
            )
            drafts.append(
                JobDraft(
                    draft_id=draft_id,
                    company=company,
                    title=title,
                    location=location,
                    jd_text=jd_text,
                )
            )
    st.button(
        "添加另一个岗位",
        on_click=_add_job_draft,
        use_container_width=True,
    )
    with st.expander("高级：按明确分隔符批量导入"):
        st.caption(
            f"每两个岗位之间使用 `{EXPLICIT_JD_DELIMITER}`。普通空行属于 JD 正文。"
        )
        st.text_area(
            "批量导入 JD",
            key="wizard_bulk_jd_import",
            height=180,
        )
        st.button(
            "导入为岗位卡片",
            on_click=_import_job_drafts,
            use_container_width=True,
        )
    return drafts


def _activate_session(session_dir: Path, output_root: Path, settings: dict[str, str]) -> None:
    st.session_state["active_session_dir"] = str(session_dir)
    st.session_state["active_output_root"] = str(output_root)
    st.session_state["session_dir"] = str(session_dir)  # 兼容旧 AppTest 与已有 session
    st.session_state["active_workbench_step"] = "resume"
    st.session_state["show_new_session_wizard"] = False
    st.session_state.pop("wizard_pending", None)
    st.session_state.pop("mock_interview_state", None)
    st.session_state.pop("ops_state", None)
    write_ui_runtime_settings(session_dir, settings)


def _create_session(payload: dict, selected_job_id: str | None) -> None:
    mode = WorkbenchMode(payload["mode"])
    if mode != WorkbenchMode.AGENT_API_LIVE:
        raise ValueError("Product session creation requires AGENT_API_LIVE")
    output_root = Path(payload["output_root"])
    latest_diagnostic_path = (
        output_root / ".internal" / "latest_ui_run_diagnostics.json"
    )
    local_defaults = load_local_runtime_defaults(
        skill_root_override=payload.get("skill_root"),
    )
    resolved_skill_root = local_defaults.skill_root or payload.get("skill_root") or ""
    status = st.status("正在创建岗位流程", expanded=True)
    status.write("1/5 解析岗位要求：识别职责、硬性要求和面试追问点。")
    status.write("2/5 匹配简历依据：判断哪些内容能直接讲、哪些需要谨慎表达。")
    status.write("3/5 生成岗位版简历：整理修改建议和完整草稿。")
    status.write("4/5 生成重点题目：围绕岗位要求设计追问。")
    status.write("5/5 生成作答提示：把真实依据组织成可练习的回答框架。")
    status.caption(
        "真实模型会依次完成多个结构化调用，通常需要几十秒。"
        "当前页面会在整条链路完成或失败后更新。"
    )
    stage_labels = {
        "offline_processing": "正在本地分析岗位和简历",
        "jd_structurer": "1/5 正在解析岗位要求",
        "evidence_mapping": "2/5 正在匹配简历依据",
        "resume_tailoring": "3/5 正在生成岗位版简历",
        "interview_prep": "4/5 正在生成重点题目",
        "answer_cards": "5/5 正在生成作答提示",
    }
    stage_box = {"value": "offline_processing"}
    runtime_box: dict[str, object] = {}

    def update_progress(stage: str) -> None:
        stage_box["value"] = stage
        status.update(label=stage_labels.get(stage, f"正在处理：{stage}"))

    def create_operation():
        runtime = build_runtime(
            mode,
            skill_root=resolved_skill_root or None,
            base_url=payload["base_url"],
            model=payload["model"],
        )
        runtime_box["runtime"] = runtime
        return run_initial_v2_live(
            payload["jobs"],
            payload["resume_text"],
            output_root,
            runtime=runtime,
            selected_job_id=selected_job_id,
            resume_source_name=payload.get("resume_source_name"),
            resume_source_text=payload.get("resume_source_text"),
            progress_callback=update_progress,
        )

    action_result = execute_live_action(
        action_id="create_session",
        operation=create_operation,
        session_dir=output_root,
        harness=lambda: getattr(
            runtime_box.get("runtime"),
            "harness",
            None,
        ),
        current_stage=lambda: stage_box["value"],
        mode=mode.value,
        diagnostic_path=latest_diagnostic_path,
    )
    if not action_result.ok:
        status.update(label="岗位流程创建失败", state="error")
        st.error(
            f"{action_result.user_message}"
            "你的 JD 和简历仍保留在页面中，可以直接再次点击创建。"
        )
        st.caption(
            f"错误类型：{action_result.error_code}。"
            "页面不会展示模型原始输出或本机 API Key。"
        )
        if action_result.diagnostic_path is not None:
            st.download_button(
                "下载本次脱敏诊断",
                data=action_result.diagnostic_path.read_bytes(),
                file_name="job_agent_run_diagnostics.json",
                mime="application/json",
            )
        return

    session_dir = Path(action_result.value)
    selected_job = next(
        job
        for job in payload["jobs"]
        if selected_job_id is None or job.job_id == selected_job_id
    )
    company_id = st.session_state.get("career_company_id")
    career_store = CareerStore.open(output_root)
    if not company_id and selected_job.company.strip():
        company_id = career_store.save_company(
            CompanyDraft(selected_job.company)
        ).id
    career_store.attach_job(
        CareerJobDraft(
            selected_job.title,
            selected_job.location,
            selected_job.url,
            selected_job.source,
            company_id=company_id,
            legacy_job_id=selected_job.job_id,
        ),
        session_dir=session_dir,
    )
    session_diagnostic_path = (
        session_dir / ".internal" / "ui_run_diagnostics.json"
    )
    if action_result.diagnostic is not None:
        write_run_diagnostic(
            session_diagnostic_path,
            action_result.diagnostic,
        )
    status.update(label="岗位流程创建完成", state="complete", expanded=False)
    settings = {
        "mode": mode.value,
        "skill_root": resolved_skill_root,
        "base_url": payload["base_url"],
        "model": payload["model"],
    }
    _activate_session(session_dir, output_root, settings)
    st.rerun()


def _render_candidate_selection(payload: dict) -> None:
    st.subheader("选择目标岗位")
    st.caption(
        f"已识别 {len(payload['candidates'])} 个岗位。"
        "当前不会假装自动排序，请选择你要准备的目标岗位。"
    )
    candidates = payload["candidates"]
    labels = [
        f"{candidate.title} · {candidate.company} [{candidate.job_id}]"
        for candidate in candidates
    ]
    selected_label = st.radio(
        "候选岗位",
        labels,
        key="wizard_selected_candidate",
    )
    selected = candidates[labels.index(selected_label)]
    selected_job = next(job for job in payload["jobs"] if job.job_id == selected.job_id)
    with st.container(border=True):
        st.markdown(f"#### {selected_job.title}")
        st.write(
            f"{selected_job.company or '未知公司'} · "
            f"{selected_job.location if selected_job.location != 'unknown' else '地点待确认'}"
        )

    col_back, col_create = st.columns([1, 3])
    with col_back:
        if st.button("返回修改", use_container_width=True):
            st.session_state.pop("wizard_pending", None)
            st.rerun()
    with col_create:
        if st.button("创建此岗位流程", type="primary", use_container_width=True):
            _create_session(payload, selected.job_id)


def render_new_session_wizard() -> None:
    st.title("开始新的求职流程")
    st.caption("创建一个独立岗位流程。创建完成后，后续操作都在工作台中继续。")

    pending = st.session_state.get("wizard_pending")
    if pending is not None:
        _render_candidate_selection(pending)
        return

    local_defaults = load_local_runtime_defaults()
    mode = WorkbenchMode.AGENT_API_LIVE
    if local_defaults.api_key:
        st.success(
            f"实时 API 已就绪（{local_defaults.api_key_source}）。"
        )
    else:
        st.warning(
            "实时 API 未就绪：请在启动应用前设置环境变量 "
            "JOB_AGENT_LIVE_API_KEY。"
        )
    col_jd, col_resume = st.columns([3, 2])
    with col_jd:
        job_drafts = _render_job_draft_editor()
    with col_resume:
        uploaded_resume = st.file_uploader(
            "导入简历文件",
            type=["txt", "md", "tex"],
            help="支持纯文本、Markdown 和 LaTeX；LaTeX 会保留源码并转换为匹配文本。",
            key="wizard_resume_file",
        )
        pasted_resume = st.text_area(
            "或粘贴简历文字",
            height=220,
            placeholder="没有文件时，可直接粘贴 Markdown 或纯文本简历。",
            key="wizard_resume",
        )
        uploaded_source = None
        if uploaded_resume is not None:
            try:
                uploaded_source = build_resume_source(
                    uploaded_resume.name,
                    uploaded_resume.getvalue(),
                )
                st.success(
                    f"已读取 {uploaded_source.filename}，"
                    f"解析出 {len(uploaded_source.normalized_markdown)} 个字符。"
                )
                with st.expander("查看解析后的简历文字"):
                    st.markdown(uploaded_source.normalized_markdown)
            except ValueError as exc:
                st.error(str(exc))

    with st.expander("高级设置"):
        output_root = st.text_input(
            "输出目录",
            value=st.session_state.get("active_output_root", "output/workbench"),
            key="wizard_output_root",
        )
        skill_root = st.text_input(
            "技能包目录（模型模式必填）",
            value=local_defaults.skill_root or "",
            key="wizard_skill_root",
        )
        if local_defaults.skill_root:
            st.caption(
                f"已自动找到技能包（{local_defaults.skill_root_source}）。"
            )
        base_url = st.text_input(
            "模型服务地址",
            value="https://open.bigmodel.cn/api/anthropic",
            key="wizard_base_url",
        )
        model = st.text_input("模型名称", value="glm-4.7", key="wizard_model")

    preview_jobs = build_jobs_from_drafts(job_drafts)
    if preview_jobs:
        st.caption(f"当前已添加 {len(preview_jobs)} 个有效岗位。")
        for preview in preview_jobs:
            preview_location = (
                preview.location
                if preview.location != "unknown"
                else "地点待确认"
            )
            st.caption(
                f"识别结果：{preview.company} · {preview.title} · {preview_location}"
            )

    readiness = check_live_runtime(
        skill_root=skill_root,
        base_url=base_url,
        model=model,
    )
    if not readiness.ready:
        st.caption(readiness.message)
    if st.button(
        "分析材料并创建流程",
        type="primary",
        use_container_width=True,
        disabled=not readiness.ready,
    ):
        if not preview_jobs and uploaded_resume is None and not pasted_resume.strip():
            st.error("JD 和简历都不能为空。")
            return
        if not preview_jobs:
            st.error("岗位 JD 不能为空。")
            return
        try:
            resume_source = (
                uploaded_source
                if uploaded_source is not None
                else build_pasted_resume_source(pasted_resume)
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        resolved_defaults = load_local_runtime_defaults(
            skill_root_override=skill_root,
        )
        resolved_skill_root = resolved_defaults.skill_root or ""
        if not resolved_skill_root:
            st.error("模型模式需要填写技能包目录。")
            return
        if not resolved_defaults.api_key:
            st.error(
                "实时 API 未就绪：请设置 JOB_AGENT_LIVE_API_KEY 环境变量。"
            )
            return
        jobs = preview_jobs
        if not jobs:
            st.error("没有解析出有效 JD。")
            return
        payload = {
            "mode": mode.value,
            "jobs": jobs,
            "resume_text": resume_source.normalized_markdown,
            "resume_source_name": resume_source.filename,
            "resume_source_text": resume_source.raw_text,
            "output_root": output_root,
            "skill_root": resolved_skill_root,
            "base_url": base_url,
            "model": model,
        }
        if len(jobs) == 1:
            _create_session(payload, jobs[0].job_id)
            return
        with st.spinner("正在分析多个岗位…"):
            candidates = run_research(
                jobs,
                OpportunityResearchGoal(
                    target_role="用户选择的目标岗位",
                    cities=["用户提供的岗位地点"],
                    keywords=[job.title for job in jobs],
                ),
                workspace_root=Path(output_root) / "_research",
                use_live=False,
            )
        payload["candidates"] = candidates
        st.session_state["wizard_pending"] = payload
        st.rerun()
