from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from job_agent.agent_runtime.failures import CheckpointError
from job_agent.agent_runtime.trajectory import CaptureLevel, sanitize_capture_payload
from job_agent.llm.provider import ProviderTrace
from job_agent.schemas import StrictModel


SEMANTIC_NODE_ORDER = (
    "jd_structurer",
    "evidence_mapping",
    "fit_verdict",
    "verdict_route",
    "resume_tailoring",
    "interview_prep",
    "answer_cards",
    "mock_interview",
)


def _payload_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SemanticGraphCheckpoint(StrictModel):
    checkpoint_version: Literal[1] = 1
    checkpoint_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    completed_node: Literal[
        "jd_structurer",
        "evidence_mapping",
        "fit_verdict",
        "verdict_route",
        "resume_tailoring",
        "interview_prep",
        "answer_cards",
        "mock_interview",
    ]
    skill_versions: dict[str, str]
    prompt_versions: dict[str, str]
    state: dict[str, Any]
    state_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    provider_traces: list[ProviderTrace] = Field(default_factory=list)
    saved_at: str
    expires_at: str

    @model_validator(mode="after")
    def validate_integrity(self):
        if self.state_digest != _payload_digest(self.state):
            raise ValueError("semantic checkpoint state digest mismatch")
        saved = datetime.fromisoformat(self.saved_at)
        expires = datetime.fromisoformat(self.expires_at)
        if saved.tzinfo is None or expires.tzinfo is None or expires <= saved:
            raise ValueError("semantic checkpoint retention window is invalid")
        return self

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        run_id: str,
        input_hash: str,
        completed_node: str,
        skill_versions: dict[str, str],
        prompt_versions: dict[str, str],
        state: dict[str, Any],
        provider_traces: list[ProviderTrace],
        now: datetime | None = None,
    ) -> "SemanticGraphCheckpoint":
        saved = now or datetime.now(UTC)
        if saved.tzinfo is None:
            saved = saved.replace(tzinfo=UTC)
        saved = saved.astimezone(UTC)
        return cls(
            checkpoint_id=f"semantic-checkpoint-{uuid4().hex}",
            session_id=session_id,
            run_id=run_id,
            input_hash=input_hash,
            completed_node=completed_node,
            skill_versions=skill_versions,
            prompt_versions=prompt_versions,
            state=state,
            state_digest=_payload_digest(state),
            provider_traces=provider_traces,
            saved_at=saved.isoformat(),
            expires_at=(saved + timedelta(days=30)).isoformat(),
        )


class SemanticCheckpointStore:
    """Atomic full-private semantic checkpoint with a fixed 30-day TTL."""

    def __init__(
        self,
        path: Path | str,
        *,
        workspace_root: Path | str,
        retention_days: int = 30,
    ) -> None:
        if retention_days != 30:
            raise ValueError("semantic checkpoint retention must be exactly 30 days")
        self.path = Path(path).resolve()
        workspace = Path(workspace_root).resolve()
        if self.path == workspace or self.path.is_relative_to(workspace):
            raise CheckpointError("semantic_checkpoint_must_be_outside_workspace")

    def save(self, checkpoint: SemanticGraphCheckpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = sanitize_capture_payload(
            checkpoint.model_dump(mode="json"),
            level=CaptureLevel.FULL_PRIVATE,
        )
        payload["state_digest"] = _payload_digest(payload["state"])
        payload = SemanticGraphCheckpoint.model_validate(payload).model_dump(mode="json")
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="\n",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            os.replace(temporary, self.path)
        except OSError as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise CheckpointError("semantic_checkpoint_write_failed") from exc

    def load(self, *, now: datetime | None = None) -> SemanticGraphCheckpoint:
        if not self.path.is_file():
            raise CheckpointError("semantic_checkpoint_not_found")
        try:
            checkpoint = SemanticGraphCheckpoint.model_validate_json(
                self.path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise CheckpointError("semantic_checkpoint_invalid") from exc
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        if datetime.fromisoformat(checkpoint.expires_at) <= current.astimezone(UTC):
            raise CheckpointError("semantic_checkpoint_expired")
        return checkpoint

    def delete(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:
            raise CheckpointError("semantic_checkpoint_delete_failed") from exc
