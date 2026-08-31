from __future__ import annotations

from statistics import mean

from job_agent.schemas import (
    AnswerCard,
    AnswerCardDeck,
    MockInterviewAnswer,
    MockInterviewAnswerSet,
    MockInterviewDebrief,
    MockInterviewPlan,
    MockInterviewQuestion,
    MockInterviewScore,
    RuleEvidenceAudit,
)


EVIDENCE_TERMS = {
    "config",
    "data",
    "format",
    "loss",
    "log",
    "logs",
    "metric",
    "metrics",
    "eval",
    "evaluation",
    "baseline",
    "checkpoint",
    "lora",
    "sft",
    "experiment",
    "配置",
    "数据",
    "日志",
    "指标",
    "评测",
    "测试",
    "实验",
    "负责",
    "实现",
}
HONEST_LIMIT_TERMS = {
    "not claim",
    "would not claim",
    "only read",
    "not sure",
    "don't know",
    "learning",
    "我没有",
    "没有实战",
    "未做过",
    "无实战",
    "不了解",
    "不熟悉",
    "只阅读",
}
OVERCLAIM_TERMS = {
    "production",
    "deployed",
    "online",
    "millions",
    "users",
    "sla",
    "served",
    "生产环境",
    "大规模上线",
    "百万用户",
}


def _by_question_id(plan: MockInterviewPlan) -> dict[str, MockInterviewQuestion]:
    return {question.question_id: question for question in plan.questions}


def _cards_by_requirement(cards: AnswerCardDeck) -> dict[str, AnswerCard]:
    return {card.requirement_id: card for card in cards.cards}


def _term_hits(answer: str, terms: set[str]) -> set[str]:
    lower = answer.lower()
    return {term for term in terms if term in lower}


def _summarize(answer: str) -> str:
    normalized = " ".join(answer.split())
    if len(normalized) <= 120:
        return normalized
    return normalized[:117].rstrip() + "..."


def audit_mock_interview(
    plan: MockInterviewPlan,
    answer_set: MockInterviewAnswerSet,
) -> list[RuleEvidenceAudit]:
    """提取可复现的表面信号；不判断技术正确性、相关性或录用结果。"""

    question_ids = {question.question_id for question in plan.questions}
    audits: list[RuleEvidenceAudit] = []
    for answer in answer_set.answers:
        if answer.question_id not in question_ids:
            continue
        text = answer.answer.strip()
        evidence_hits = sorted(_term_hits(text, EVIDENCE_TERMS))
        honest_hits = sorted(_term_hits(text, HONEST_LIMIT_TERMS))
        overclaim_hits = sorted(_term_hits(text, OVERCLAIM_TERMS))
        risk_flags: list[str] = []
        if not text:
            length_band = "missing"
            risk_flags.append("missing_answer")
        elif len(text) < 40:
            length_band = "thin"
            risk_flags.append("thin_answer")
        else:
            length_band = "adequate"
        if honest_hits:
            risk_flags.append("truth_boundary_signal")
        if overclaim_hits:
            risk_flags.append("overclaim_signal")
        lower = f" {text.casefold()} "
        ownership_clear = not (
            " we " in lower
            or "我们" in text
        )
        if not ownership_clear:
            risk_flags.append("vague_ownership")
        audits.append(
            RuleEvidenceAudit(
                question_id=answer.question_id,
                answer_present=bool(text),
                answer_length_band=length_band,
                evidence_signals=evidence_hits,
                risk_flags=risk_flags,
                ownership_clear=ownership_clear,
            )
        )
    return audits


def _score_answer(answer: MockInterviewAnswer, question: MockInterviewQuestion, card: AnswerCard | None) -> MockInterviewScore:
    text = answer.answer.strip()
    evidence_hits = _term_hits(text, EVIDENCE_TERMS)
    honest_hits = _term_hits(text, HONEST_LIMIT_TERMS)
    overclaim_hits = _term_hits(text, OVERCLAIM_TERMS)
    risk_flags: list[str] = []

    if not text:
        score = 1
        diagnosis = "没有提供可用于评分的答案。"
        improvement = "先给出简洁、真实的回答，再说明可以支撑它的证据。"
        risk_flags.append("missing answer")
    elif len(text) < 40:
        score = 2
        diagnosis = "答案过短，尚未体现深度或证据。"
        improvement = "补充具体数据、配置、指标、日志或个人负责范围。"
        risk_flags.append("thin answer")
    elif overclaim_hits and card and "production" not in card.supporting_evidence.lower():
        score = 1
        diagnosis = "答案包含答题卡证据无法支持的高风险表述。"
        improvement = "没有明确证据时，删除生产环境或上线规模相关表述。"
        risk_flags.append("overclaim")
    elif honest_hits:
        score = 3
        diagnosis = "答案诚实说明了能力边界，但还需要连接到相邻的真实经验。"
        improvement = "按“没有做过什么、实际做过什么、如何补齐差距”组织回答。"
        risk_flags.append("thin but honest")
    elif evidence_hits:
        score = 5 if len(evidence_hits) >= 4 else 4
        diagnosis = "答案具体，并且关联了可验证的证据。"
        improvement = "压缩成 60 秒结构，并明确说明你的个人负责范围。"
    else:
        score = 3
        diagnosis = "答案基本合理，但缺少具体支撑证据。"
        improvement = "补充明确的产物、指标、日志或实验。"
        risk_flags.append("missing evidence")

    if "we " in f" {text.lower()} ":
        risk_flags.append("vague ownership")

    return MockInterviewScore(
        question_id=answer.question_id,
        question=question.prompt,
        answer_summary=_summarize(text),
        score=score,
        diagnosis=diagnosis,
        risk_flags=risk_flags,
        improvement=improvement,
    )


def _readiness(average_score: float, scores: list[MockInterviewScore]) -> str:
    if any("overclaim" in score.risk_flags for score in scores):
        return "high risk"
    if average_score >= 4.2:
        return "ready"
    if average_score >= 3.0:
        return "needs targeted practice"
    return "not ready"


def score_mock_interview(
    plan: MockInterviewPlan,
    answer_set: MockInterviewAnswerSet,
    answer_cards: AnswerCardDeck,
) -> MockInterviewDebrief:
    question_map = _by_question_id(plan)
    card_map = _cards_by_requirement(answer_cards)
    scores: list[MockInterviewScore] = []

    for answer in answer_set.answers:
        question = question_map.get(answer.question_id)
        if question is None:
            continue
        card = card_map.get(question.requirement_id)
        scores.append(_score_answer(answer, question, card))

    if not scores:
        raise ValueError("cannot build debrief without answers matching the mock interview plan")

    average_score = round(mean(score.score for score in scores), 2)
    strengths = [score.diagnosis for score in scores if score.score >= 4]
    gaps = [score.improvement for score in scores if score.score < 4 or score.risk_flags]
    if not strengths:
        strengths = ["Honesty and recovery can still be built through targeted practice."]
    if not gaps:
        gaps = ["Keep practicing concise ownership and evidence-backed explanations."]

    return MockInterviewDebrief(
        company=plan.company,
        title=plan.title,
        mode=answer_set.mode,
        average_score=average_score,
        readiness=_readiness(average_score, scores),  # type: ignore[arg-type]
        scores=scores,
        strengths_to_keep=strengths[:3],
        gaps_to_fix=gaps[:3],
        action_checklist=[
            "Rewrite the weakest answer with one concrete artifact and one ownership sentence.",
            "Rehearse each high-risk question in 60 seconds.",
            "Re-run the mock interview on questions with score below 4.",
        ],
        final_note=(
            "Practice diagnosis only; rule-based evidence completeness baseline, "
            "not technical correctness and not a real interview pass/fail prediction."
        ),
    )


def render_mock_debrief(debrief: MockInterviewDebrief) -> str:
    lines = [
        "# Mock Interview Debrief",
        "",
        f"**Mode:** {debrief.mode}",
        f"**Company:** {debrief.company}",
        f"**Role:** {debrief.title}",
        f"**Average score:** {debrief.average_score}/5.0",
        f"**Readiness:** {debrief.readiness}",
        "",
        "## Per-Question Scores",
        "",
        "| # | Question | Answer Summary | Score | Diagnosis | Risk Flags |",
        "|---|----------|----------------|:--:|-----------|------------|",
    ]
    for index, score in enumerate(debrief.scores, start=1):
        risks = ", ".join(score.risk_flags) if score.risk_flags else "none"
        lines.append(
            f"| {index} | {score.question} | {score.answer_summary} | {score.score} | {score.diagnosis} | {risks} |"
        )
    lines.extend(["", "## Overall Assessment", ""])
    lines.extend(f"- Strength: {item}" for item in debrief.strengths_to_keep)
    lines.extend(f"- Gap: {item}" for item in debrief.gaps_to_fix)
    lines.extend(["", "## Action Checklist", ""])
    lines.extend(f"- {item}" for item in debrief.action_checklist)
    lines.extend(
        [
            "",
            "## Use These Resources",
            "",
            "- `07_interview_grilling.md`",
            "- `08_answer_cards.md`",
            "- `09_mock_interview_plan.md`",
            "",
            f"Final note: {debrief.final_note}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
