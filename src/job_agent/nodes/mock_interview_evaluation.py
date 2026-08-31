from __future__ import annotations

from job_agent.schemas import (
    MockInterviewEvaluationResult,
    MockInterviewTranscriptItem,
    RuleEvidenceAudit,
)


_STATUS_LABELS = {
    "strong": "表现较强",
    "partial": "部分覆盖",
    "weak": "需要重点补齐",
    "insufficient_information": "信息不足",
}


def render_mock_interview_evaluation(
    evaluation: MockInterviewEvaluationResult,
    transcript: list[MockInterviewTranscriptItem],
) -> str:
    questions = {item.question_id: item for item in transcript}
    lines = [
        f"# AI 模拟面试评价：{evaluation.company} · {evaluation.title}",
        "",
        f"- 运行 ID：`{evaluation.run_id}`",
        f"- 五维等权总分：{evaluation.overall_score:.2f}/5",
        "",
        "## 逐题评价",
        "",
    ]
    for item in evaluation.question_evaluations:
        source = questions[item.question_id]
        lines.extend(
            [
                f"### {source.question}",
                "",
                f"- 状态：{_STATUS_LABELS.get(item.status, item.status)}",
                (
                    "- 五维："
                    f"相关性 {item.relevance} / 技术深度 {item.technical_depth} / "
                    f"推理表达 {item.reasoning_clarity} / 证据充分度 {item.evidence_grounding} / "
                    f"事实边界 {item.truth_boundary}"
                ),
                f"- 我的回答：{source.answer or '未回答'}",
            ]
        )
        if item.strengths:
            lines.append(f"- 可保留：{'；'.join(item.strengths)}")
        if item.gaps:
            lines.append(f"- 需要补齐：{'；'.join(item.gaps)}")
        if item.improved_answer:
            lines.append(f"- 改进版回答：{item.improved_answer}")
        if item.evidence_ids:
            lines.append(f"- 引用证据：{', '.join(item.evidence_ids)}")
        if item.next_focus:
            lines.append(f"- 下一步练习：{item.next_focus}")
        lines.append("")

    lines.extend(["## 整轮结论", ""])
    lines.extend(f"- 优势：{item}" for item in evaluation.strengths)
    lines.extend(f"- 优先缺口：{item}" for item in evaluation.priority_gaps)
    lines.extend(["", "## 下一轮主题", ""])
    lines.extend(f"- {item}" for item in evaluation.next_mock_topics)
    lines.extend(["", "## 局限与边界", ""])
    lines.extend(f"- {item}" for item in evaluation.limitations)
    lines.extend(
        [
            "",
            "说明：这是受 Guard 约束的 LLM 语义评价，不是客观真值，"
            "不判断面试是否通过，也不预测录用概率。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_rule_evidence_audit(
    audits: list[RuleEvidenceAudit],
) -> str:
    lines = [
        "# 规则证据审计",
        "",
        "这里只展示可复现的表面信号，不判断技术正确性、回答相关性或录用结果。",
        "",
        "| 题目 | 是否作答 | 长度 | 证据信号 | 风险信号 | 个人贡献是否清晰 |",
        "|---|---:|---|---|---|---:|",
    ]
    for item in audits:
        lines.append(
            "| "
            + " | ".join(
                [
                    item.question_id,
                    "是" if item.answer_present else "否",
                    item.answer_length_band,
                    "、".join(item.evidence_signals) or "无",
                    "、".join(item.risk_flags) or "无",
                    "是" if item.ownership_clear else "否",
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"
