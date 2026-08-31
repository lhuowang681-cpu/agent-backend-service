from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CompanyDraft:
    display_name: str
    priority: str = "normal"
    website_url: str | None = None
    notes: str = ""
    tags: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Company:
    id: str
    normalized_name: str
    display_name: str
    priority: str
    website_url: str | None
    notes: str
    tags: tuple[str, ...] = ()
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class JobDraft:
    title: str
    city: str = ""
    url: str = ""
    source: str = "manual"
    current_state: str = "to_apply"
    legacy_job_id: str | None = None
    company_id: str | None = None


@dataclass(frozen=True)
class Job:
    id: str
    company_id: str | None
    legacy_job_id: str | None
    title: str
    city: str
    url: str
    source: str
    current_state: str
    session_dir: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class StageEventDraft:
    state: str
    occurred_at: str | None = None
    notes: str = ""
    origin: str = "user"
    source_key: str | None = None


@dataclass(frozen=True)
class StageEvent:
    id: str
    job_id: str
    state: str
    occurred_at: str
    notes: str
    origin: str
    source_key: str


@dataclass(frozen=True)
class ArtifactLink:
    id: str
    job_id: str | None
    session_dir: str
    artifact_type: str
    relative_path: str
    content_hash: str | None


@dataclass(frozen=True)
class Interview:
    id: str
    job_id: str | None
    company_id: str | None
    kind: str
    round_label: str
    occurred_at: str | None
    session_dir: str
    artifact_path: str
    ai_review_path: str | None


@dataclass(frozen=True)
class CompanySummary:
    company: Company
    job_count: int
    latest_state: str | None
    latest_updated_at: str | None


@dataclass(frozen=True)
class CompanyDetail:
    company: Company
    jobs: tuple[Job, ...] = ()
    interviews: tuple[Interview, ...] = ()
    latest_stage: str | None = None
    latest_updated_at: str | None = None
