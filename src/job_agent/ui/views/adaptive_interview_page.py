from __future__ import annotations

from pathlib import Path

import streamlit as st

from job_agent.career import CareerStore
from job_agent.interview.debrief import AdaptiveDebriefService
from job_agent.interview.dossier import FactConfirmation, ProjectDossierService
from job_agent.interview.run_store import AdaptiveInterviewRunStore
from job_agent.ui.adaptive_interview_loop import index_finished_interview, start_full_mock, start_project_grill, submit_adaptive_answer
from job_agent.ui.local_config import RuntimeDefaults
from job_agent.ui.modes import build_live_runtime, check_live_runtime


def _runtime(defaults: RuntimeDefaults):
    if not defaults.skill_root:
        raise RuntimeError("Live runtime not ready; missing skill_root")
    return build_live_runtime(skill_root=defaults.skill_root, base_url=defaults.base_url, model=defaults.model)


def _render_active_run(
    *,
    root: Path,
    run_id: str,
    repo_root: Path,
    defaults: RuntimeDefaults,
    career_store: CareerStore,
    live_ready: bool,
) -> None:
    store = AdaptiveInterviewRunStore(root)
    run = store.load(run_id)
    st.subheader("面试进行中" if run.status in {"waiting", "paused"} else "面试已结束")
    for turn in run.transcript:
        with st.chat_message("assistant"):
            st.write(turn.question)
        with st.chat_message("user"):
            st.write(turn.answer or "")
    if run.status == "paused":
        if st.button("继续面试", key="adaptive_resume", disabled=not live_ready):
            store.set_status(run_id, "waiting")
            st.rerun()
        return
    if run.status == "waiting" and run.pending_turn is not None:
        with st.chat_message("assistant"):
            st.write(run.pending_turn.public_message)
        answer = st.text_area("你的回答", key=f"adaptive_answer_{run.revision}", height=180)
        elapsed = st.number_input("本题用时（秒）", min_value=0, value=0, key=f"adaptive_elapsed_{run.revision}")
        col1, col2, col3 = st.columns(3)
        if col1.button("提交回答", key=f"adaptive_submit_{run.revision}", disabled=not live_ready):
            try:
                updated = submit_adaptive_answer(root=root, repo_root=repo_root, run_id=run_id, answer=answer, elapsed_seconds=int(elapsed), runtime=_runtime(defaults))
            except Exception as exc:
                st.error(f"本轮未提交：{exc}")
            else:
                st.session_state["adaptive_run_id"] = updated.run_id
                st.rerun()
        if col2.button("暂停", key=f"adaptive_pause_{run.revision}"):
            store.set_status(run_id, "paused")
            st.rerun()
        if col3.button("提前结束", key=f"adaptive_end_{run.revision}"):
            store.set_status(run_id, "ended_by_user")
            st.rerun()
    if run.status in {"completed", "ended_by_user"}:
        try:
            index_finished_interview(career_store=career_store, root=root, run=run)
        except Exception as exc:
            st.warning(f"面试记录暂未写入公司库：{exc}")
        debrief_path = root / "interview_runs" / run_id / "debrief.json"
        if not debrief_path.exists():
            if st.button("生成最终复盘", key="adaptive_debrief", disabled=not live_ready):
                try:
                    AdaptiveDebriefService(root).generate(run, runtime=_runtime(defaults))
                except Exception as exc:
                    st.error(f"复盘生成失败，可重试：{exc}")
                else:
                    index_finished_interview(career_store=career_store, root=root, run=run)
                    st.rerun()
        else:
            from job_agent.interview.contracts import AdaptiveInterviewDebrief
            debrief = AdaptiveInterviewDebrief.model_validate_json(debrief_path.read_text(encoding="utf-8"))
            st.subheader("最终复盘")
            st.write(f"本场结论：{debrief.conclusion}")
            for item in debrief.dimensions:
                st.write(f"{item.dimension}: {item.score}/5 · {item.rationale}")
            for card in debrief.improvement_cards:
                with st.expander(f"回答改进卡 · {card.turn_id}"):
                    st.write(card.issue)
                    st.write(" → ".join(card.recommended_structure))
                    st.caption(card.honesty_boundary)
                    st.write(card.retry_question)
        if st.button("重新开始", key="adaptive_restart"):
            st.session_state.pop("adaptive_run_id", None)
            st.session_state.pop("adaptive_run_root", None)
            st.rerun()


def render_adaptive_interview_page(
    *,
    output_root: Path,
    repo_root: Path,
    store: CareerStore,
    defaults: RuntimeDefaults,
) -> None:
    st.header("面试训练")
    if st.button("返回工作台", key="adaptive_back"):
        workbench_page = st.session_state.get("_workbench_page")
        if workbench_page is not None:
            st.switch_page(workbench_page)
        st.session_state.pop("career_page", None)
        st.rerun()
    readiness = check_live_runtime(skill_root=defaults.skill_root, base_url=defaults.base_url, model=defaults.model)
    if not readiness.ready:
        st.warning(readiness.message)
    active_root = st.session_state.get("adaptive_run_root")
    active_run = st.session_state.get("adaptive_run_id")
    if active_root and active_run:
        _render_active_run(
            root=Path(active_root),
            run_id=active_run,
            repo_root=repo_root,
            defaults=defaults,
            career_store=store,
            live_ready=readiness.ready,
        )
        return

    dossier_service = ProjectDossierService(output_root)
    current = dossier_service.load_current(repo_root)
    tab_project, tab_full = st.tabs(["项目拷打", "完整模拟面试"])
    with tab_project:
        if current is None or not current.ready_for_interview:
            if current is not None:
                st.warning("项目代码已变化，需重新生成并确认 Project Dossier。")
            if st.button("读取当前仓库并生成 Dossier 草稿", disabled=not readiness.ready):
                try:
                    draft = dossier_service.generate_draft(repo_root, runtime=_runtime(defaults))
                except Exception as exc:
                    st.error(f"Dossier 生成失败：{exc}")
                else:
                    st.session_state["adaptive_dossier_draft"] = draft.model_dump(mode="json")
                    st.rerun()
            draft_payload = st.session_state.get("adaptive_dossier_draft")
            if draft_payload:
                from job_agent.interview.contracts import ProjectDossier
                draft = ProjectDossier.model_validate(draft_payload)
                st.caption("实现事实来自代码；ownership、指标和设计动机必须由你确认。")
                decisions = []
                for fact in draft.facts:
                    st.write(f"[{fact.kind}] {fact.statement}")
                    if fact.status == "needs_confirmation":
                        choice = st.selectbox("确认状态", ["待确认", "确认", "拒绝"], key=f"fact_choice_{fact.fact_id}")
                        edited = st.text_input("修正表述（可选）", value=fact.statement, key=f"fact_text_{fact.fact_id}")
                        if choice != "待确认":
                            decisions.append(FactConfirmation(fact_id=fact.fact_id, decision="confirmed" if choice == "确认" else "rejected", edited_statement=edited))
                if st.button("保存确认后的 Dossier"):
                    try:
                        dossier_service.confirm(draft, decisions)
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.session_state.pop("adaptive_dossier_draft", None)
                        st.rerun()
        else:
            st.success(f"Project Dossier revision {current.revision} 已就绪")
            scope = st.selectbox("拷打范围", ["全项目实战", "项目介绍与架构", "Agent Runtime 与 Tool Use", "状态、Checkpoint 与恢复", "安全、审批与幂等", "Evaluation 与测试", "失败案例与设计取舍"])
            duration = st.select_slider("时长（分钟）", options=[15, 30, 45, 60], value=30, key="project_duration")
            pressure = st.radio("压力等级", ["正常", "高压"], horizontal=True, key="project_pressure")
            if st.button("开始项目拷打", disabled=not readiness.ready):
                try:
                    root, run = start_project_grill(output_root=output_root, repo_root=repo_root, dossier=current, duration_minutes=duration, pressure="high" if pressure == "高压" else "normal", focus=None if scope == "全项目实战" else scope, runtime=_runtime(defaults))
                except Exception as exc:
                    st.error(f"无法开始项目拷打：{exc}")
                else:
                    st.session_state["adaptive_run_root"] = str(root)
                    st.session_state["adaptive_run_id"] = run.run_id
                    st.rerun()
    with tab_full:
        selected_job_id = st.session_state.get("adaptive_job_id")
        if not selected_job_id:
            st.info("请从公司详情中的岗位进入完整模拟面试。")
        else:
            job = store.get_job(selected_job_id)
            st.write(f"岗位：{job.title}")
            st.caption("本场只使用该岗位关联的 JD、定制简历和 Evidence 快照。")
            duration = st.select_slider("时长（分钟）", options=[15, 30, 45, 60], value=30, key="full_duration")
            pressure = st.radio("压力等级", ["正常", "高压"], horizontal=True, key="full_pressure")
            if st.button("开始完整模拟面试", disabled=not readiness.ready):
                try:
                    usable_dossier = current if current and current.ready_for_interview else None
                    root, run = start_full_mock(career_store=store, repo_root=repo_root, job_id=selected_job_id, dossier=usable_dossier, duration_minutes=duration, pressure="high" if pressure == "高压" else "normal", runtime=_runtime(defaults))
                except Exception as exc:
                    st.error(f"无法开始完整模拟面试：{exc}")
                else:
                    st.session_state["adaptive_run_root"] = str(root)
                    st.session_state["adaptive_run_id"] = run.run_id
                    st.rerun()
