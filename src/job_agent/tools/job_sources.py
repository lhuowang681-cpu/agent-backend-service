from __future__ import annotations

from pathlib import Path
from typing import Protocol

from job_agent.schemas import JobSourceKind, RawJob, SearchIntent
from job_agent.tools.job_search import load_fixture_jobs, load_raw_jobs


class JobSourceAdapter(Protocol):
    source_kind: JobSourceKind

    def load(self, intent: SearchIntent, source_path: Path) -> list[RawJob]:
        ...


class FixtureJobSourceAdapter:
    source_kind = JobSourceKind.FIXTURE

    def load(self, intent: SearchIntent, source_path: Path) -> list[RawJob]:
        return load_fixture_jobs(source_path)


class RawJobsSourceAdapter:
    source_kind = JobSourceKind.RAW_JOBS

    def load(self, intent: SearchIntent, source_path: Path) -> list[RawJob]:
        return load_raw_jobs(source_path)


_ADAPTERS: dict[JobSourceKind, JobSourceAdapter] = {
    JobSourceKind.FIXTURE: FixtureJobSourceAdapter(),
    JobSourceKind.RAW_JOBS: RawJobsSourceAdapter(),
}


def get_job_source_adapter(source_kind: JobSourceKind | str) -> JobSourceAdapter:
    kind = JobSourceKind(source_kind)
    return _ADAPTERS[kind]
