from __future__ import annotations

from pathlib import Path

from job_agent.schemas import JobScoutResult, JobSourceKind, SearchIntent
from job_agent.tools.job_sources import JobSourceAdapter, get_job_source_adapter


def scout_jobs(
    intent: SearchIntent,
    source_kind: JobSourceKind | str,
    source_path: Path | str,
    adapter: JobSourceAdapter | None = None,
) -> JobScoutResult:
    kind = JobSourceKind(source_kind)
    path = Path(source_path)
    source_adapter = adapter or get_job_source_adapter(kind)
    adapter_kind = JobSourceKind(source_adapter.source_kind)
    if adapter_kind != kind:
        raise ValueError(f"adapter source_kind mismatch: expected {kind}, got {adapter_kind}")
    jobs = source_adapter.load(intent=intent, source_path=path)

    warnings = []
    if not jobs:
        warnings.append("no_jobs_found")

    return JobScoutResult(
        intent=intent,
        source_kind=kind,
        source_path=str(path),
        jobs=jobs,
        warnings=warnings,
    )
