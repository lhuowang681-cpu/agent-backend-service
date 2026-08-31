from __future__ import annotations

from job_agent.schemas import JobLead


BUCKET_ORDER = ["Strong Fit", "Good Fit", "Risky Fit", "Not Recommended"]


def render_funnel_report(leads: list[JobLead]) -> str:
    lines = ["# Job Funnel Report", ""]
    for bucket in BUCKET_ORDER:
        bucket_leads = [lead for lead in leads if lead.bucket == bucket]
        lines.extend([f"## {bucket}", ""])
        if not bucket_leads:
            lines.extend(["_No jobs in this bucket._", ""])
            continue
        for index, lead in enumerate(bucket_leads, start=1):
            job = lead.job
            lines.append(f"{index}. **{job.company} - {job.title}** ({lead.score}%)")
            lines.append(f"   - Job ID: {job.job_id}")
            lines.append(f"   - Location: {job.location}")
            lines.append(f"   - Source: {job.source}")
            if job.posted_date:
                lines.append(f"   - Posted: {job.posted_date}")
            if job.fetched_at:
                lines.append(f"   - Fetched: {job.fetched_at}")
            lines.append(f"   - URL: {job.url}")
            lines.append(f"   - Reasons: {'; '.join(lead.reasons)}")
            risk_text = "; ".join(lead.risks) if lead.risks else "No obvious funnel risk."
            lines.append(f"   - Risks: {risk_text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
