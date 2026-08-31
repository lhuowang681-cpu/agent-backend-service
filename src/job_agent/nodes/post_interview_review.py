from __future__ import annotations

from job_agent.schemas import (
    PostInterviewGap,
    PostInterviewReviewResult,
    PostInterviewRoundInput,
)


_CATEGORY_LABELS = {
    "knowledge": "知识缺口",
    "expression": "表达结构",
    "project_evidence": "项目证据",
    "truth_boundary": "事实边界",
}
_STATUS_LABELS = {
    "strong": "回答较完整",
    "partial": "部分覆盖",
    "weak": "需要重点补齐",
    "insufficient_information": "信息不足",
}


def _render_gap(gap: PostInterviewGap) -> str:
    category = _CATEGORY_LABELS.get(gap.category, gap.category)
    requirements = (
        f"；关联要求：{', '.join(gap.requirement_ids)}"
        if gap.requirement_ids
        else ""
    )
    return (
        f"- **{category} / {gap.priority}**：{gap.finding}"
        f"{requirements}\n  - 下一步：{gap.action}"
    )


def render_post_interview_review(
    review: PostInterviewReviewResult,
    interview_round: PostInterviewRoundInput,
) -> str:
    lines = [
        f"# AI 面试复盘：{review.company} · {review.title}",
        "",
        f"- 面试阶段：{interview_round.stage}",
        f"- 轮次 ID：`{review.round_id}`",
        "",
        "## AI 总结",
        "",
        review.summary,
        "",
        "## 逐题诊断",
        "",
    ]
    question_map = {
        item.question_id: item for item in interview_round.questions
    }
    for diagnosis in review.question_diagnoses:
        source = question_map[diagnosis.question_id]
        lines.extend(
            [
                f"### {source.question}",
                "",
                f"- 状态：{_STATUS_LABELS.get(diagnosis.status, diagnosis.status)}",
                f"- 我的回答：{source.answer or '未记录'}",
                f"- 面试官追问：{source.interviewer_follow_up or '未记录'}",
            ]
        )
        if diagnosis.strengths:
            lines.append(f"- 可保留：{'；'.join(diagnosis.strengths)}")
        if diagnosis.improved_answer:
            lines.append(f"- 改进版回答：{diagnosis.improved_answer}")
        lines.append(f"- 练习动作：{diagnosis.practice_action}")
        if diagnosis.gaps:
            lines.append("")
            lines.extend(_render_gap(gap) for gap in diagnosis.gaps)
        lines.append("")

    lines.extend(["## 优先补齐项", ""])
    if review.priority_gaps:
        lines.extend(_render_gap(gap) for gap in review.priority_gaps)
    else:
        lines.append("- 当前记录没有形成可靠的优先级结论。")

    lines.extend(["", "## 学习与练习计划", ""])
    lines.extend(f"- {item}" for item in review.learning_plan)
    lines.extend(["", "## 下一轮模拟面试重点", ""])
    lines.extend(f"- {item}" for item in review.next_mock_topics)
    lines.extend(["", "## 简历调整建议", ""])
    lines.extend(
        f"- {item}" for item in review.resume_adjustments
    )
    lines.extend(["", "## 信息不足与边界", ""])
    lines.extend(f"- {item}" for item in review.limitations)
    lines.extend(
        [
            "",
            "说明：AI 复盘用于发现准备缺口，不判断面试是否通过，也不预测录用概率。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
