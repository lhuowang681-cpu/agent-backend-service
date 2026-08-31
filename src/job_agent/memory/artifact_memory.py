from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from pydantic import Field

from job_agent.schemas import StrictModel


class ArtifactMemoryHit(StrictModel):
    path: str
    content: str
    sha256: str
    chars: int = Field(ge=0)


class ArtifactMemoryRetriever:
    def __init__(self, session_dir: Path, *, max_artifact_chars: int = 50_000) -> None:
        self.session_dir = session_dir.resolve()
        self.max_artifact_chars = max_artifact_chars

    def retrieve(self, artifact_refs: list[str]) -> list[ArtifactMemoryHit]:
        hits: list[ArtifactMemoryHit] = []
        for artifact_ref in artifact_refs:
            relative = PurePosixPath(artifact_ref.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"artifact reference escapes session: {artifact_ref}")
            if relative.suffix.lower() not in {".json", ".md"}:
                raise ValueError(f"unsupported artifact memory extension: {artifact_ref}")
            candidate = self.session_dir.joinpath(*relative.parts)
            if candidate.is_symlink():
                raise ValueError(f"symbolic link artifact is not allowed: {artifact_ref}")
            resolved = candidate.resolve(strict=True)
            try:
                resolved.relative_to(self.session_dir)
            except ValueError as exc:
                raise ValueError(f"artifact reference escapes session: {artifact_ref}") from exc
            content = resolved.read_text(encoding="utf-8")
            if len(content) > self.max_artifact_chars:
                raise ValueError(f"artifact exceeds context budget: {artifact_ref}")
            hits.append(
                ArtifactMemoryHit(
                    path=relative.as_posix(),
                    content=content,
                    sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    chars=len(content),
                )
            )
        return hits
