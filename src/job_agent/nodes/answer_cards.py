from __future__ import annotations

from job_agent.schemas import AnswerCard, AnswerCardDeck, EvidenceItem, EvidenceLevel, InterviewPrep, TargetedResume


def _evidence_by_requirement(evidence: list[EvidenceItem]) -> dict[str, EvidenceItem]:
    return {item.requirement_id: item for item in evidence}


def _answer_for(level: EvidenceLevel, evidence_item: EvidenceItem | None) -> tuple[str, str, str]:
    if evidence_item is None:
        return (
            "这项要求目前缺少直接经历。回答时先诚实说明没有做过，再连接到最接近的真实经验和补齐计划。",
            "简历中暂未找到可直接支撑这项要求的依据。",
            "不要声称自己完成过相关实现、上线、指标提升或生产环境交付。",
        )
    if level in {EvidenceLevel.NONE, EvidenceLevel.C0}:
        return (
            "这项要求目前缺少直接经历。回答时先诚实说明没有做过，再连接到最接近的真实经验和补齐计划。",
            evidence_item.proof,
            evidence_item.risk,
        )
    if level == EvidenceLevel.C1:
        return (
            f"先说明能力边界，再按“实际做了什么—留下什么产物—还缺什么”回答：{evidence_item.claim}。",
            evidence_item.proof,
            evidence_item.risk,
        )
    return (
        f"按“结论—个人动作—证据—复盘”组织回答：{evidence_item.claim}。",
        evidence_item.proof,
        evidence_item.risk,
    )


def build_answer_cards(
    interview_prep: InterviewPrep,
    evidence: list[EvidenceItem],
    targeted_resume: TargetedResume,
) -> AnswerCardDeck:
    evidence_map = _evidence_by_requirement(evidence)
    resume_claims = len(targeted_resume.conservative_bullets) + len(targeted_resume.standard_bullets)
    cards: list[AnswerCard] = []

    for question in interview_prep.questions:
        evidence_item = evidence_map.get(question.requirement_id)
        short_answer, supporting_evidence, boundary = _answer_for(question.evidence_level, evidence_item)
        cards.append(
            AnswerCard(
                requirement_id=question.requirement_id,
                question=question.question,
                short_answer=short_answer,
                evidence_level=question.evidence_level,
                supporting_evidence=supporting_evidence,
                boundary=boundary,
                practice_prompts=[
                    "控制在 60 秒内，不添加没有依据的指标或成果。",
                    "说出能够证明这段经历的具体代码、日志、文档或实验结果。",
                    f"回答只围绕当前简历中 {resume_claims} 条可安全使用的事实。",
                ],
            )
        )

    return AnswerCardDeck(company=interview_prep.company, title=interview_prep.title, cards=cards)


def render_answer_cards(deck: AnswerCardDeck) -> str:
    lines = [
        "# Answer Cards",
        "",
        f"- Company: {deck.company}",
        f"- Title: {deck.title}",
        "",
        "## Cards",
        "",
    ]
    for index, card in enumerate(deck.cards, start=1):
        lines.append(f"### {index}. {card.question}")
        lines.append("")
        lines.append(f"- Requirement ID: `{card.requirement_id}`")
        lines.append(f"- Evidence level: `{card.evidence_level.value}`")
        lines.append(f"- Short answer: {card.short_answer}")
        lines.append(f"- Supporting evidence: {card.supporting_evidence}")
        lines.append(f"- Truth boundary: {card.boundary}")
        if card.practice_prompts:
            lines.append("- Practice prompts:")
            lines.extend(f"  - {item}" for item in card.practice_prompts)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
