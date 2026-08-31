from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

from pydantic import Field

from job_agent.schemas import ApplicationRecord, ApplicationStatus, ApplicationTracker, StrictModel


DEFAULT_USER_STATE_PATH = Path("output/.internal/state.json")
DEFAULT_TRACKER_PATH = Path("output/tracker.json")

MemoryKind = Literal["preference", "fact", "decision", "session_summary"]


class MemorySource(str, Enum):
    USER_STATED = "user_stated"
    ARTIFACT_VERIFIED = "artifact_verified"
    MODEL_SUGGESTED = "model_suggested"


class MemoryRecord(StrictModel):
    memory_id: str
    user_id: str
    kind: MemoryKind
    content: str
    source: MemorySource
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: str
    updated_at: str
    expires_at: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)


class MemoryHit(StrictModel):
    record: MemoryRecord
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    artifact_refs: list[str] = Field(default_factory=list)


class UserState(StrictModel):
    user_id: str
    version: int = Field(default=1, ge=1)
    user: dict[str, Any] = Field(default_factory=dict)
    current_hunt: dict[str, Any] = Field(default_factory=dict)
    last_session: dict[str, Any] = Field(default_factory=dict)
    preferences: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
    memories: list[MemoryRecord] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class RunState(StrictModel):
    session_id: str
    run_id: str
    values: dict[str, Any] = Field(default_factory=dict)
    route: str | None = None
    traces: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


class UserStateStore(Protocol):
    @property
    def path(self) -> Path:
        """Return the private user-level state path."""

    def load(self, *, user_id: str) -> UserState:
        """Fresh-read the requested user namespace."""

    def save(self, *, user_id: str, state: UserState, expected_version: int | None = None) -> UserState:
        """Atomically persist a complete state after optional version checking."""

    def delete(self, *, user_id: str) -> None:
        """Delete only the requested user's private state."""


class MemoryRetriever(Protocol):
    def retrieve(
        self,
        *,
        user_id: str,
        query: str,
        kinds: Sequence[MemoryKind] | None = None,
        limit: int = 5,
        include_unconfirmed: bool = False,
    ) -> list[MemoryHit]:
        """Return bounded, provenance-carrying hits instead of an entire memory store."""


class ApplicationTrackerStore(Protocol):
    @property
    def canonical_path(self) -> Path:
        """Return the user-level single source of truth path."""

    def load(self, *, user_id: str) -> ApplicationTracker:
        """Fresh-read and validate the canonical tracker."""

    def add_application(
        self,
        *,
        user_id: str,
        application: ApplicationRecord,
        user_confirmed: bool,
    ) -> ApplicationRecord:
        """Add or explicitly resolve a duplicate, then atomically persist."""

    def update_status(
        self,
        *,
        user_id: str,
        application_id: str,
        new_status: ApplicationStatus,
        notes: str,
        user_confirmed: bool,
        timestamp: str | None = None,
    ) -> ApplicationRecord:
        """Validate a confirmed transition and append history."""

    def write_session_snapshot(
        self,
        *,
        user_id: str,
        session_dir: Path,
        application_id: str,
    ) -> Path:
        """Write a compatibility snapshot without creating a second source of truth."""
