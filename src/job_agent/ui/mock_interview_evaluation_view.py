from __future__ import annotations

import streamlit as st

from job_agent.schemas import (
    MockInterviewEvaluationResult,
    RuleEvidenceAudit,
)


def render_mock_ai_evaluation(
    evaluation: MockInterviewEvaluationResult,
) -> None:
    st.markdown("#### AI 综合评价")
    st.metric("五维等权总分", f"{evaluation.overall_score:.2f} / 5")
    for item in evaluation.question_evaluations:
        with st.expander(
            f"{item.question_id} · {item.status}",
            expanded=item.status in {"weak", "insufficient_information"},
        ):
            cols = st.columns(5)
            labels = (
                ("相关性", item.relevance),
                ("技术深度", item.technical_depth),
                ("推理表达", item.reasoning_clarity),
                ("证据充分度", item.evidence_grounding),
                ("事实边界", item.truth_boundary),
            )
            for column, (label, score) in zip(cols, labels):
                column.metric(label, f"{score}/5")
            if item.strengths:
                st.markdown("**可以保留**")
                for strength in item.strengths:
                    st.write(f"- {strength}")
            if item.gaps:
                st.markdown("**需要补齐**")
                for gap in item.gaps:
                    st.write(f"- {gap}")
            if item.improved_answer:
                st.markdown(f"**改进版回答**\n\n{item.improved_answer}")
            if item.next_focus:
                st.markdown(f"**下一步练习**\n\n{item.next_focus}")
    if evaluation.priority_gaps:
        st.markdown("**整轮优先缺口**")
        for item in evaluation.priority_gaps:
            st.write(f"- {item}")
    if evaluation.next_mock_topics:
        st.markdown("**下一轮建议主题**")
        for item in evaluation.next_mock_topics:
            st.write(f"- {item}")
    if evaluation.limitations:
        st.markdown("**局限与事实边界**")
        for item in evaluation.limitations:
            st.write(f"- {item}")
    st.caption(
        "这是受确定性 Guard 约束的 LLM 语义评价，不是客观真值，"
        "不判断面试是否通过，也不预测录用概率。"
    )


def render_rule_audits(audits: list[RuleEvidenceAudit]) -> None:
    with st.expander("规则证据审计（辅助信号）", expanded=False):
        st.caption(
            "规则只检查是否作答、长度、证据词、个人贡献和越界风险；"
            "它不判断技术正确性，也不参与 AI 五维总分。"
        )
        for item in audits:
            st.markdown(f"**{item.question_id}**")
            st.write(
                f"- 作答：{'是' if item.answer_present else '否'}"
                f"；长度：{item.answer_length_band}"
                f"；个人贡献清晰：{'是' if item.ownership_clear else '否'}"
            )
            st.write(
                f"- 证据信号：{'、'.join(item.evidence_signals) or '无'}"
            )
            st.write(
                f"- 风险信号：{'、'.join(item.risk_flags) or '无'}"
            )
