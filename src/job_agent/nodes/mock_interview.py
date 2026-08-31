from __future__ import annotations

from job_agent.schemas import AnswerCardDeck, EvidenceLevel, InterviewPrep, MockInterviewPlan, MockInterviewQuestion


SCORING_DIMENSIONS = [
    "technical_depth",
    "evidence_quality",
    "ownership_clarity",
    "communication",
    "composure",
    "role_fit",
]


LIVE_RULES = [
    "Ask one question at a time.",
    "Use at most 1-2 follow-ups per primary question.",
    "Score each answer on a 1-5 scale after the candidate answers.",
    "Give only one-line feedback during the mock.",
    "Do not provide the correct answer during the mock.",
    "Save teaching, model answers, and remediation for the debrief.",
]


def _persona_for(company: str) -> str:
    company_lower = company.lower()
    if "byte" in company_lower or "bytedance" in company_lower:
        return "fast-paced ownership press interviewer"
    if "moonshot" in company_lower or "zhipu" in company_lower or "baidu" in company_lower:
        return "research-depth technical interviewer"
    if "alibaba" in company_lower:
        return "business value and resilience interviewer"
    return "fair senior engineer"


def _risk_flags(level: EvidenceLevel) -> list[str]:
    if level in {EvidenceLevel.C2, EvidenceLevel.C3}:
        return []
    if level == EvidenceLevel.C1:
        return ["thin evidence", "ownership follow-up likely"]
    return ["missing evidence", "do not overclaim"]


def build_mock_interview_plan(
    interview_prep: InterviewPrep,
    answer_cards: AnswerCardDeck,
    mode: str = "technical",
    question_count: int = 4,
    difficulty: str = "realistic",
) -> MockInterviewPlan:
    answer_requirement_ids = {card.requirement_id for card in answer_cards.cards}
    selected_questions = [
        question for question in interview_prep.questions if question.requirement_id in answer_requirement_ids
    ][:question_count]
    if not selected_questions:
        selected_questions = interview_prep.questions[:question_count]

    questions: list[MockInterviewQuestion] = []
    for index, question in enumerate(selected_questions, start=1):
        questions.append(
            MockInterviewQuestion(
                question_id=f"Q{index}",
                requirement_id=question.requirement_id,
                prompt=question.question,
                follow_ups=question.follow_ups[:2],
                time_limit_seconds=180,
                focus=question.intent,
                risk_flags=_risk_flags(question.evidence_level),
            )
        )

    return MockInterviewPlan(
        company=interview_prep.company,
        title=interview_prep.title,
        mode=mode,  # type: ignore[arg-type]
        persona=_persona_for(interview_prep.company),
        difficulty=difficulty,  # type: ignore[arg-type]
        questions=questions,
        scoring_dimensions=SCORING_DIMENSIONS,
        live_rules=LIVE_RULES,
    )


def render_mock_interview_plan(plan: MockInterviewPlan) -> str:
    lines = [
        "# Mock Interview Plan",
        "",
        f"- Mode: {plan.mode}",
        f"- Company: {plan.company}",
        f"- Role: {plan.title}",
        f"- Persona: {plan.persona}",
        f"- Difficulty: {plan.difficulty}",
        "",
        "## Live Rules",
        "",
    ]
    lines.extend(f"- {rule}" for rule in plan.live_rules)
    lines.extend(
        [
            "",
            "## Scoring Dimensions",
            "",
        ]
    )
    lines.extend(f"- {dimension}" for dimension in plan.scoring_dimensions)
    lines.extend(["", "## Question Queue", ""])
    for question in plan.questions:
        lines.append(f"### {question.question_id}. {question.prompt}")
        lines.append("")
        lines.append(f"- Requirement ID: `{question.requirement_id}`")
        lines.append(f"- Focus: {question.focus}")
        lines.append(f"- Time limit: {question.time_limit_seconds} seconds")
        if question.risk_flags:
            lines.append(f"- Risk flags: {', '.join(question.risk_flags)}")
        if question.follow_ups:
            lines.append("- Follow-ups:")
            lines.extend(f"  - {item}" for item in question.follow_ups)
        lines.append("")
    lines.extend(
        [
            "## Debrief Boundary",
            "",
            "- Do not write `15_mock_interview_debrief.md` until candidate answers are collected and scored.",
            "- Use answer cards only after the live round, during debrief.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
