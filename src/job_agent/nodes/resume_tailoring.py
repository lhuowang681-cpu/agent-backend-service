from __future__ import annotations

from job_agent.schemas import EvidenceItem, EvidenceLevel, ResumeBullet, StructuredJD, TargetedResume


STRONG_LEVELS = {EvidenceLevel.C2, EvidenceLevel.C3}
CAUTIOUS_LEVELS = {EvidenceLevel.C1}
UNSUPPORTED_LEVELS = {EvidenceLevel.C0, EvidenceLevel.NONE}


def _bullet_from_evidence(item: EvidenceItem, prefix: str) -> ResumeBullet:
    return ResumeBullet(
        requirement_id=item.requirement_id,
        text=f"{prefix}: {item.claim}",
        evidence_level=item.level,
        evidence_summary=item.proof,
        risk=item.risk,
    )


def tailor_resume(structured_jd: StructuredJD, evidence: list[EvidenceItem]) -> TargetedResume:
    conservative: list[ResumeBullet] = []
    standard: list[ResumeBullet] = []
    stronger: list[ResumeBullet] = []
    claims_to_remove: list[str] = []

    for item in evidence:
        if item.level in STRONG_LEVELS:
            conservative.append(_bullet_from_evidence(item, "Evidence-backed bullet"))
        elif item.level in CAUTIOUS_LEVELS:
            standard.append(_bullet_from_evidence(item, "Cautious bullet after 1-day evidence check"))
            stronger.append(_bullet_from_evidence(item, "Stronger bullet after adding concrete proof"))
        elif item.level in UNSUPPORTED_LEVELS:
            claims_to_remove.append(f"{item.claim} ({item.level.value}): {item.risk}")
            stronger.append(_bullet_from_evidence(item, "Only after evidence exists"))

    return TargetedResume(
        company=structured_jd.company,
        title=structured_jd.title,
        strategy_summary=(
            "Use C2/C3 evidence directly, keep C1 claims cautious, "
            "and remove or downgrade unsupported claims."
        ),
        conservative_bullets=conservative,
        standard_bullets=standard,
        stronger_after_evidence=stronger,
        claims_to_remove=claims_to_remove,
    )


def _render_bullets(bullets: list[ResumeBullet], empty_text: str) -> list[str]:
    if not bullets:
        return [empty_text]
    return [
        f"- {bullet.text} [evidence={bullet.evidence_level.value}; proof={bullet.evidence_summary}; risk={bullet.risk}]"
        for bullet in bullets
    ]


def render_targeted_resume(tailoring: TargetedResume) -> str:
    lines = [
        f"# Targeted Resume Tailoring: {tailoring.company} - {tailoring.title}",
        "",
        "## Resume Strategy",
        "",
        f"- {tailoring.strategy_summary}",
        "",
        "## Conservative Bullets",
        "",
        *_render_bullets(tailoring.conservative_bullets, "- No C2/C3 evidence-backed bullet is ready yet."),
        "",
        "## Standard Bullets",
        "",
        *_render_bullets(tailoring.standard_bullets, "- No C1 cautious bullet is available."),
        "",
        "## Stronger After Evidence",
        "",
        *_render_bullets(tailoring.stronger_after_evidence, "- No evidence-upgrade bullet is needed."),
        "",
        "## Claims To Remove Or Downgrade",
        "",
    ]
    if tailoring.claims_to_remove:
        lines.extend(f"- {claim}" for claim in tailoring.claims_to_remove)
    else:
        lines.append("- No unsupported claim detected.")
    return "\n".join(lines).rstrip() + "\n"
