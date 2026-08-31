from __future__ import annotations

from job_agent.schemas import (
    EvidenceItem,
    EvidenceLevel,
    FitVerdictResult,
    InterviewPrep,
    InterviewQuestion,
    JobRequirement,
    StructuredJD,
    TargetedResume,
)


def _evidence_by_requirement(evidence: list[EvidenceItem]) -> dict[str, EvidenceItem]:
    return {item.requirement_id: item for item in evidence}


def _risk_note(evidence_item: EvidenceItem | None) -> tuple[EvidenceLevel, str]:
    if evidence_item is None or evidence_item.level == EvidenceLevel.NONE:
        return EvidenceLevel.NONE, "No evidence yet. Do not claim hands-on ownership."
    if evidence_item.level == EvidenceLevel.C0:
        return evidence_item.level, "Only keyword-level evidence. Use as learning exposure, not project ownership."
    if evidence_item.level == EvidenceLevel.C1:
        return evidence_item.level, f"Cautious claim only: {evidence_item.risk}"
    return evidence_item.level, evidence_item.risk


def _question_for(requirement: JobRequirement, level: EvidenceLevel) -> str:
    if level in {EvidenceLevel.C2, EvidenceLevel.C3}:
        return f"请结合一个真实项目说明你的经验：{requirement.text}"
    if level == EvidenceLevel.C1:
        return f"这项要求中你实际做过哪一部分：{requirement.text}"
    return f"在没有项目证据前，你会如何验证和补齐这项能力：{requirement.text}"


def prepare_interview(
    structured_jd: StructuredJD,
    evidence: list[EvidenceItem],
    fit_result: FitVerdictResult,
    targeted_resume: TargetedResume,
) -> InterviewPrep:
    evidence_map = _evidence_by_requirement(evidence)
    questions: list[InterviewQuestion] = []
    resume_claim_count = len(targeted_resume.conservative_bullets) + len(targeted_resume.standard_bullets)

    for requirement in structured_jd.must_have:
        evidence_item = evidence_map.get(requirement.id)
        level, risk_note = _risk_note(evidence_item)
        questions.append(
            InterviewQuestion(
                requirement_id=requirement.id,
                question=_question_for(requirement, level),
                intent=requirement.probe,
                evidence_level=level,
                risk_note=risk_note,
                follow_ups=[
                    "What exact data, config, metric, or log can support this answer?",
                    "Which part was independently done by you?",
                ],
            )
        )

    return InterviewPrep(
        company=structured_jd.company,
        title=structured_jd.title,
        strategy_summary=(
            f"Verdict is {fit_result.verdict.value}. Prepare from {resume_claim_count} resume-safe claims "
            "and keep unsupported requirements framed as learning gaps."
        ),
        questions=questions,
    )


def render_interview_prep(prep: InterviewPrep) -> str:
    lines = [
        "# Interview Grilling",
        "",
        f"- Company: {prep.company}",
        f"- Title: {prep.title}",
        f"- Strategy: {prep.strategy_summary}",
        "",
        "## Questions",
        "",
    ]
    for index, question in enumerate(prep.questions, start=1):
        lines.append(f"### {index}. {question.question}")
        lines.append("")
        lines.append(f"- Requirement ID: `{question.requirement_id}`")
        lines.append(f"- Evidence level: `{question.evidence_level.value}`")
        lines.append(f"- Intent: {question.intent}")
        lines.append(f"- Risk: {question.risk_note}")
        if question.follow_ups:
            lines.append("- Follow-ups:")
            lines.extend(f"  - {item}" for item in question.follow_ups)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
