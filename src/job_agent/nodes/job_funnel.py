from __future__ import annotations

from job_agent.schemas import JobLead, RawJob


STRONG_TERMS = [
    "post-training",
    "post training",
    "RLHF",
    "RLAIF",
    "SFT",
    "LoRA",
    "DPO",
    "GRPO",
    "reward model",
    "alignment",
    "instruction tuning",
    "preference data",
    "safety alignment",
]
RISK_TERMS = ["backend", "database", "cache", "Web"]


def _lead_score_adjustment(lead_score: int | None) -> int:
    if lead_score is None:
        return 0
    if lead_score >= 85:
        return 5
    if lead_score <= 40:
        return -5
    return 0


def _score_job(job: RawJob) -> tuple[int, list[str], list[str]]:
    text = f"{job.title} {job.desc}"
    lowered = text.lower()
    hits = [term for term in STRONG_TERMS if term.lower() in lowered]
    risks = [f"Non-posttraining signal: {term}" for term in RISK_TERMS if term.lower() in lowered]
    risks.extend(f"Crawler risk: {flag}" for flag in job.risk_flags)

    score = 35 + len(hits) * 8 - len(risks) * 15 + _lead_score_adjustment(job.lead_score)
    reasons = [f"Keyword hit: {term}" for term in hits[:5]]
    if job.lead_reason:
        reasons.append(f"Scout hint: {job.lead_reason}")
    if not reasons:
        reasons = ["No post-training/RLHF core keyword hit."]
    return max(0, min(100, score)), reasons, risks


def _bucket(score: int) -> str:
    if score >= 80:
        return "Strong Fit"
    if score >= 65:
        return "Good Fit"
    if score >= 45:
        return "Risky Fit"
    return "Not Recommended"


def score_jobs(jobs: list[RawJob]) -> list[JobLead]:
    leads = []
    for job in jobs:
        score, reasons, risks = _score_job(job)
        leads.append(JobLead(job=job, score=score, bucket=_bucket(score), reasons=reasons, risks=risks))
    return sorted(leads, key=lambda lead: lead.score, reverse=True)
