from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Protocol

from pydantic import Field

from job_agent.schemas import StrictModel


class CareerSnapshotNotFoundError(LookupError):
    pass


class CareerSnapshotConflictError(ValueError):
    pass


class CareerSnapshotPayload(StrictModel):
    source_ref: str = Field(min_length=1, max_length=512)
    resume_text: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class CareerSnapshotRecord(StrictModel):
    user_id: str
    snapshot_revision: str
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    payload: CareerSnapshotPayload
    created_at: datetime


class CareerSnapshotRepository(Protocol):
    def put_career_snapshot(
        self,
        *,
        user_id: str,
        snapshot_revision: str,
        content_hash: str,
        payload: dict[str, object],
    ) -> CareerSnapshotRecord: ...

    def get_career_snapshot(
        self,
        *,
        user_id: str,
        snapshot_revision: str,
    ) -> CareerSnapshotRecord: ...

    def has_career_snapshot(
        self,
        *,
        user_id: str,
        snapshot_revision: str,
    ) -> bool: ...


class VersionedCareerSnapshotStore:
    """Immutable, user-scoped snapshots; payload remains private to the Worker."""

    def __init__(self, repository: CareerSnapshotRepository) -> None:
        self.repository = repository

    def put(
        self,
        *,
        user_id: str,
        source_ref: str,
        resume_text: str,
        evidence_refs: list[str] | None = None,
    ) -> CareerSnapshotRecord:
        payload = CareerSnapshotPayload(
            source_ref=source_ref,
            resume_text=resume_text,
            evidence_refs=evidence_refs or [],
        )
        encoded = json.dumps(
            payload.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        content_hash = hashlib.sha256(encoded).hexdigest()
        return self.repository.put_career_snapshot(
            user_id=user_id,
            snapshot_revision=f"snapshot-{content_hash[:24]}",
            content_hash=content_hash,
            payload=payload.model_dump(mode="json"),
        )

    def resolve(
        self,
        *,
        user_id: str,
        snapshot_revision: str,
    ) -> CareerSnapshotRecord:
        return self.repository.get_career_snapshot(
            user_id=user_id,
            snapshot_revision=snapshot_revision,
        )

    def require_owned(self, *, user_id: str, snapshot_revision: str) -> None:
        if not self.repository.has_career_snapshot(
            user_id=user_id,
            snapshot_revision=snapshot_revision,
        ):
            raise CareerSnapshotNotFoundError("career_snapshot_not_found")
