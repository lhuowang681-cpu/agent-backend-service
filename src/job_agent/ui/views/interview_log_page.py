from __future__ import annotations

from pathlib import Path

import streamlit as st

from job_agent.schemas import RuleEvidenceAudit
from job_agent.ui.interview_review import build_timeline, load_run_payload
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
    load_post_interview_review_result,
)
from job_agent.ui.session_view import infer_output_root


def _item_key(item) -> str:
    source = item.run_path or item.review_path or item.session_dir
    return f"{item.item_type}::{source}"


def _activate_session(session_dir: Path) -> None:
    st.session_state["active_session_dir"] = str(session_dir)
    st.session_state["session_dir"] = str(session_dir)
    st.session_state["active_output_root"] = str(infer_output_root(session_dir))
    st.session_state.pop("active_workbench_step", None)


def _render_mock_detail(item) -> None:
    payload = load_run_payload(item.run_path) if item.run_path else None
    if payload is None:
        st.warning("该轮数据无法读取。")
        return
    result = payload.get("result", {})
    evaluation = load_mock_interview_evaluation(item.run_path)
    if evaluation is not None:
        render_mock_ai_evaluation(evaluation)
    elif payload.get("schema_version") == 2:
        if payload.get("evaluation_status") == "ai_failed":
            st.warning(
                "本轮 transcript 和规则证据审计已保存，但 AI 评价生成失败。"
            )
        else:
            st.info("本轮只有 transcript 和规则证据审计，尚无 AI 语义评价。")
    else:
        average = result.get("average_score")
        st.metric(
            "历史规则基线",
            f"{average:.1f}/5" if average is not None else "—",
        )

    for index, answer in enumerate(payload.get("history", []), start=1):
        st.divider()
        qid = answer.get("question_id", f"Q{index}")
        st.markdown(f"#### {qid}")
        st.write(answer.get("prompt", ""))
        st.caption("你的答案")
        st.write(answer.get("answer", ""))
        if payload.get("schema_version") != 2:
            st.caption("改进建议")
            st.write(answer.get("improvement", "—"))
    audits = [
        RuleEvidenceAudit.model_validate(raw)
        for raw in payload.get("rule_evidence_audit", [])
    ]
    if audits:
        render_rule_audits(audits)


def _render_real_detail(item) -> None:
    if item.review_path is None or not item.review_path.exists():
        st.warning("面经文件缺失。")
        return
    render_post_interview_review(item.review_path)
    if item.review_path.suffix.lower() == ".json":
        round_id = item.review_path.stem
        ai_review = load_post_interview_review_result(
            item.session_dir,
            round_id,
        )
        if ai_review is not None:
            render_ai_review_result(ai_review)
        else:
            st.info("本轮状态：仅记录，尚未生成 AI 复盘。")


def render_interview_log_page() -> None:
    st.title("面试日志")
    st.caption("跨岗位查看模拟面试和真实面经。这里不启动或修改当前工作流。")

    output_root = Path(st.session_state.get("active_output_root", "output/workbench"))
    with st.expander("数据范围"):
        output_root_value = st.text_input(
            "输出目录",
            value=str(output_root),
            key="interview_log_output_root",
        )
        if Path(output_root_value) != output_root:
            output_root = Path(output_root_value)
            st.session_state["active_output_root"] = str(output_root)

    try:
        timeline = build_timeline(output_root)
    except Exception:
        timeline = []
        st.warning("面试日志扫描失败，请检查输出目录和文件权限。")

    if not timeline:
        st.info("当前目录还没有面试记录。")
        return

    companies = sorted({item.company for item in timeline})
    col_search, col_company, col_type = st.columns([2, 1, 1])
    with col_search:
        search = st.text_input(
            "搜索",
            placeholder="公司或岗位",
            key="interview_log_search",
        ).casefold().strip()
    with col_company:
        company_filter = st.selectbox(
            "公司",
            ["全部", *companies],
            key="interview_log_company",
        )
    with col_type:
        type_filter = st.selectbox(
            "类型",
            ["全部", "模拟面试", "真实面经"],
            key="interview_log_type",
        )

    filtered = []
    for item in timeline:
        if company_filter != "全部" and item.company != company_filter:
            continue
        if type_filter == "模拟面试" and item.item_type != "mock":
            continue
        if type_filter == "真实面经" and item.item_type != "real":
            continue
        haystack = f"{item.company} {item.title}".casefold()
        if search and search not in haystack:
            continue
        filtered.append(item)

    if not filtered:
        st.info("没有符合当前筛选条件的记录。")
        return

    selected_key = st.session_state.get("selected_interview_log_item")
    selected_item = next(
        (item for item in filtered if _item_key(item) == selected_key),
        filtered[0],
    )
    st.session_state["selected_interview_log_item"] = _item_key(selected_item)

    col_list, col_detail = st.columns([1, 2], gap="large")
    with col_list:
        st.subheader(f"记录 · {len(filtered)}")
        for item in filtered:
            key = _item_key(item)
            is_mock = item.item_type == "mock"
            type_label = (
                item.round_label
                if is_mock
                else f"真实面试 · {item.round_label}"
            )
            if not is_mock:
                type_label += (
                    " · 已 AI 复盘"
                    if item.ai_review_path is not None
                    and item.ai_review_path.is_file()
                    else " · 仅记录"
                )
            if (
                is_mock
                and item.ai_review_path is not None
                and item.average_score is not None
            ):
                score = f" · AI 五维总分 {item.average_score:.1f}/5"
            elif (
                is_mock
                and item.average_score is not None
                and item.average_score > 0
            ):
                score = f" · 历史规则基线 {item.average_score:.1f}/5"
            else:
                score = ""
            with st.container(border=True):
                st.markdown(f"**{item.company} · {item.title}**")
                st.caption(f"{item.date or '日期未知'} · {type_label}{score}")
                if st.button(
                    "查看详情",
                    key=f"log_detail::{key}",
                    type="primary" if key == _item_key(selected_item) else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["selected_interview_log_item"] = key
                    st.rerun()

    with col_detail:
        st.caption(selected_item.date or "日期未知")
        st.subheader(f"{selected_item.company} · {selected_item.title}")
        st.write(
            selected_item.round_label
            if selected_item.item_type == "mock"
            else f"真实面试 · {selected_item.round_label}"
        )
        if st.button("设为当前岗位并回到工作台", type="primary"):
            _activate_session(selected_item.session_dir)
            workbench_page = st.session_state.get("_workbench_page")
            if workbench_page is not None:
                st.switch_page(workbench_page)
            st.success("已设为当前岗位。请点击左侧“工作台”返回。")
        if selected_item.item_type == "mock":
            _render_mock_detail(selected_item)
        else:
            _render_real_detail(selected_item)
