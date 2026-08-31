from __future__ import annotations

import json
from pathlib import Path

from job_agent.schemas import RawJob


def load_fixture_jobs(path: Path) -> list[RawJob]:
    if not path.exists():
        raise FileNotFoundError(f"Job fixture not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return [RawJob.model_validate(item) for item in data]


def _pick(item: dict, *keys: str, default=None):
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return default


def load_raw_jobs(path: Path) -> list[RawJob]:
    if not path.exists():
        raise FileNotFoundError(f"raw_jobs file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    jobs = []
    for index, item in enumerate(data, start=1):
        normalized = {
            "job_id": str(_pick(item, "job_id", "id", "jobId", default=f"raw_{index:03d}")),
            "company": _pick(item, "company", "company_name", "companyName", default="Unknown"),
            "title": _pick(item, "title", "name", "position", default="Untitled"),
            "desc": _pick(item, "desc", "description", "jd", "job_description", default=""),
            "url": _pick(item, "url", "link", "job_url", default=""),
            "location": _pick(item, "location", "city", "work_city", default="unknown"),
            "salary": _pick(item, "salary", "salary_range", default=None),
            "source": _pick(item, "source", default="raw_jobs"),
            "job_type": _pick(item, "job_type", "jobType", "type", default="unknown"),
            "posted_date": _pick(item, "posted_date", "postedDate", "date", default=None),
            "fetched_at": _pick(item, "fetched_at", "fetchedAt", default=None),
            "lead_score": _pick(item, "lead_score", "leadScore", default=None),
            "lead_reason": _pick(item, "lead_reason", "leadReason", default=None),
            "risk_flags": _pick(item, "risk_flags", "riskFlags", default=[]),
        }
        jobs.append(RawJob.model_validate(normalized))
    return jobs


def build_manual_job(
    jd_text: str,
    company: str = "Manual",
    title: str = "Manual JD",
    location: str = "unknown",
) -> RawJob:
    return RawJob(
        job_id="manual_001",
        company=company,
        title=title,
        desc=jd_text,
        url="manual://pasted-jd",
        location=location,
        salary=None,
        source="manual",
        job_type="unknown",
    )
