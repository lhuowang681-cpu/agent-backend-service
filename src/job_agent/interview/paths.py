from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _safe_component(value: str) -> str:
    if not value or value in {".", ".."} or any(char in value for char in ("/", "\\", ":")):
        raise ValueError("path component is not safe")
    return value


@dataclass(frozen=True)
class InterviewArtifactPaths:
    root: Path
    run_id: str

    @classmethod
    def for_run(cls, root: Path, run_id: str) -> "InterviewArtifactPaths":
        return cls(Path(root).resolve(), _safe_component(run_id))

    @property
    def run_dir(self) -> Path:
        return self.root / "interview_runs" / self.run_id

    @property
    def context(self) -> Path:
        return self.run_dir / "context.json"

    @property
    def blueprint(self) -> Path:
        return self.run_dir / "blueprint.json"

    @property
    def run(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def transcript(self) -> Path:
        return self.run_dir / "transcript.json"

    @property
    def trajectory(self) -> Path:
        return self.run_dir / "trajectory.jsonl"

    @property
    def debrief_json(self) -> Path:
        return self.run_dir / "debrief.json"

    @property
    def debrief_markdown(self) -> Path:
        return self.run_dir / "debrief.md"
