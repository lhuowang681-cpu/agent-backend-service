from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from job_agent.evidence.contracts import EvidenceV2Error


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def require_safe_id(value: str, *, error_code: str = "unsafe_identifier") -> str:
    if not _SAFE_ID.fullmatch(value) or value in {".", ".."}:
        raise EvidenceV2Error(error_code)
    return value


@dataclass(frozen=True)
class EvidenceArtifactPaths:
    session_dir: Path
    run_id: str

    def __post_init__(self) -> None:
        require_safe_id(self.run_id, error_code="unsafe_run_id")

    @property
    def runs_root(self) -> Path:
        return self.session_dir / "evidence_runs"

    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.run_id

    @property
    def sources_dir(self) -> Path:
        return self.run_dir / "sources"

    @property
    def current(self) -> Path:
        return self.session_dir / "evidence_current.json"

    def source_path(self, source_id: str) -> Path:
        return self.sources_dir / f"{require_safe_id(source_id, error_code='unsafe_source_id')}.txt"


ARTIFACT_FILENAMES = ("sources.json", "requirements.json", "mapping.json", "assessment.json")
