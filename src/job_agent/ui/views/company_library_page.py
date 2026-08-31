from __future__ import annotations

from pathlib import Path

import streamlit as st

from job_agent.career import CareerStore, Company, CompanyDraft, JobDraft


def prefill_new_job(company: Company) -> None:
    st.session_state["career_company_id"] = company.id
    st.session_state["career_company_name"] = company.display_name
    st.session_state["career_open_new_job"] = True


def render_company_library_page(*, output_root: Path, store: CareerStore) -> None:
    st.header("公司库")
    if st.button("返回工作台", key="career_back_to_workbench"):
        workbench_page = st.session_state.get("_workbench_page")
        if workbench_page is not None:
            st.switch_page(workbench_page)
        st.session_state.pop("career_page", None)
        st.rerun()
    preview = st.session_state.get("career_migration_summary")
    if preview:
        st.info(f"迁移：已导入 {preview.get('imported', 0)}，已合并 {preview.get('merged', 0)}，待整理 {preview.get('pending', 0)}")
    with st.expander("新建公司", expanded=not bool(store.list_companies())):
        with st.form("new_company"):
            name = st.text_input("公司名称")
            priority = st.selectbox("优先级", ["high", "normal", "low"])
            tags = st.text_input("标签（逗号分隔）")
            notes = st.text_area("备注")
            if st.form_submit_button("保存公司"):
                if not name.strip(): st.error("公司名称不能为空")
                else:
                    store.save_company(CompanyDraft(name, priority, notes=notes, tags={t.strip() for t in tags.split(",") if t.strip()})); st.success("已保存公司"); st.rerun()
    query = st.text_input("搜索公司", key="career_company_query")
    priority = st.selectbox("筛选优先级", ["全部", "high", "normal", "low"], key="career_company_priority")
    tags = st.multiselect("筛选标签", store.list_tags(), key="career_company_tags")
    summaries = store.list_companies(query=query, priority=None if priority == "全部" else priority, tags=set(tags))
    if not summaries: st.caption("暂无公司")
    for summary in summaries:
        company = summary.company
        with st.container(border=True):
            st.subheader(company.display_name)
            st.caption(f"标签：{', '.join(company.tags) or '—'} · 岗位：{summary.job_count} · 阶段：{summary.latest_state or '待整理'}")
            if st.button("查看详情", key=f"company_detail_{company.id}"):
                st.session_state["career_selected_company"] = company.id; st.rerun()
    selected_id = st.session_state.get("career_selected_company")
    if selected_id:
        detail = store.get_company_detail(selected_id)
        st.divider(); st.subheader(f"公司详情 · {detail.company.display_name}")
        st.write(detail.company.notes or "暂无备注")
        with st.expander("编辑公司"):
            with st.form(f"edit_company_{detail.company.id}"):
                display_name = st.text_input("公司名称", value=detail.company.display_name)
                priority_value = st.selectbox("优先级", ["high", "normal", "low"], index=["high", "normal", "low"].index(detail.company.priority) if detail.company.priority in {"high", "normal", "low"} else 1)
                tag_value = st.text_input("标签（逗号分隔）", value=", ".join(detail.company.tags))
                notes_value = st.text_area("备注", value=detail.company.notes)
                if st.form_submit_button("保存修改"):
                    try:
                        store.save_company(CompanyDraft(display_name, priority_value, notes=notes_value, tags={tag.strip() for tag in tag_value.split(",") if tag.strip()}), company_id=detail.company.id)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.success("已保存修改"); st.rerun()
        if st.button("添加岗位", key="career_add_job"):
            prefill_new_job(detail.company)
        if st.session_state.get("career_open_new_job"):
            with st.form("new_job"):
                title = st.text_input("岗位标题")
                city = st.text_input("城市")
                url = st.text_input("JD 链接")
                jd = st.text_area("JD（可选，继续使用既有 JD+简历流程）")
                if st.form_submit_button("保存岗位"):
                    if not title.strip(): st.error("岗位标题不能为空")
                    else:
                        job = store.attach_job(
                            JobDraft(
                                title,
                                city,
                                url,
                                "manual",
                                company_id=detail.company.id,
                            ),
                            session_dir=None,
                        )
                        st.session_state["career_job_id"] = job.id
                        st.session_state["career_job_title"] = job.title
                        st.session_state["career_prefill_jd"] = jd
                        st.session_state["career_ready_for_workflow"] = True
                        st.success("岗位已保存，可继续既有工作流")
        if st.session_state.get("career_ready_for_workflow"):
            if st.button("使用此岗位开始 JD 工作流", key="career_start_jd_workflow"):
                st.session_state.pop("career_page", None)
                st.session_state.pop("career_ready_for_workflow", None)
                workbench_page = st.session_state.get("_workbench_page")
                if workbench_page is not None:
                    st.session_state["show_new_session_wizard"] = True
                    st.switch_page(workbench_page)
                st.rerun()
        if detail.jobs:
            st.markdown("**岗位**")
            for job in detail.jobs:
                col_job, col_interview = st.columns([3, 1])
                col_job.write(f"{job.title} · {job.city or '城市未填写'} · {job.current_state}")
                if col_interview.button("完整模拟面试", key=f"adaptive_full_mock_{job.id}"):
                    st.session_state["adaptive_job_id"] = job.id
                    st.session_state["career_page"] = "adaptive_interview"
                    adaptive_page = st.session_state.get("_adaptive_interview_page")
                    if adaptive_page is not None:
                        st.switch_page(adaptive_page)
                    st.rerun()
        if detail.interviews:
            st.markdown("**面试记录**")
            for interview in detail.interviews: st.write(f"{interview.round_label or interview.kind} · {interview.occurred_at or '未记录日期'}")
