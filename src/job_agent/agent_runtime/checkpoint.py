from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import Field, ValidationError

from job_agent.agent_runtime.budgets import BudgetUsage
from job_agent.agent_runtime.contracts import (
    AgentAction,
    AgentRunState,
    AgentStepRecord,
    ApprovalRequest,
    UserInputRecord,
    UserInputRequest,
)
from job_agent.agent_runtime.failures import CheckpointError
from job_agent.schemas import StrictModel


class AgentCheckpoint(StrictModel):
    checkpoint_version: Literal[1] = 1
    checkpoint_id: str = Field(min_length=1)
    saved_at: str
    state: AgentRunState
    budget_usage: BudgetUsage
    steps: list[AgentStepRecord] = Field(default_factory=list)
    user_inputs: list[UserInputRecord] = Field(default_factory=list)
    verifier_feedback: list[str] = Field(default_factory=list)
    completed_action_ids: list[str] = Field(default_factory=list)
    completed_non_idempotent_keys: list[str] = Field(default_factory=list)
    inflight_non_idempotent_key: str | None = None
    pending_action: AgentAction | None = None
    pending_approval: ApprovalRequest | None = None
    pending_user_input: UserInputRequest | None = None
    registry_fingerprint: str
    runtime_version: str
    skill_version: str
    prompt_version: str

    @classmethod
    def create(cls, **values) -> "AgentCheckpoint":
        return cls(
            checkpoint_id=f"checkpoint-{uuid4().hex}",
            saved_at=datetime.now(UTC).isoformat(),
            **values,
        )


class CheckpointStore(Protocol):
    def save(self, checkpoint: AgentCheckpoint) -> None:
        ...

    def load(self) -> AgentCheckpoint:
        ...

    def delete(self) -> None:
        ...


class JsonCheckpointStore:
    """Single-run checkpoint store using same-directory atomic replacement."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def save(self, checkpoint: AgentCheckpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            checkpoint.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        temporary_path: Path | None = None
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
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                temporary_path = Path(handle.name)
            os.replace(temporary_path, self.path)
        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise CheckpointError("checkpoint_write_failed") from exc

    def load(self) -> AgentCheckpoint:
        if not self.path.exists():
            raise CheckpointError("checkpoint_not_found")
        try:
            return AgentCheckpoint.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as exc:
            raise CheckpointError("checkpoint_invalid") from exc

    def delete(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:
            raise CheckpointError("checkpoint_delete_failed") from exc
