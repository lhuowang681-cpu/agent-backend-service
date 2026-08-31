from __future__ import annotations

import json
import os
import re
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from job_agent.agent_runtime.failures import TrajectoryError
from job_agent.agent_runtime.trajectory import (
    CaptureLevel,
    JsonlTrajectoryStore,
    sanitize_capture_payload,
)


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class PrivateRunStore:
    """Repository-external private capture with a fixed V1 30-day retention policy."""

    def __init__(
        self,
        root: Path | str,
        *,
        workspace_root: Path | str,
        retention_days: int = 30,
    ) -> None:
        if retention_days != 30:
            raise ValueError("Harness V1 full_private retention must be exactly 30 days")
        self.root = Path(root).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        self.retention_days = retention_days
        if self.root == self.workspace_root or self.root.is_relative_to(self.workspace_root):
            raise TrajectoryError("private_store_must_be_outside_workspace")

    def trajectory(self, run_id: str) -> JsonlTrajectoryStore:
        return JsonlTrajectoryStore(
            self._run_dir(run_id) / "trajectory.jsonl",
            capture_level=CaptureLevel.FULL_PRIVATE,
            _private_storage_validated=True,
        )

    def write_bundle(self, run_id: str, bundle: dict[str, Any]) -> Path:
        path = self._run_dir(run_id) / "private_bundle.json"
        payload = sanitize_capture_payload(bundle, level=CaptureLevel.FULL_PRIVATE)
        self._atomic_json_write(path, payload)
        return path

    def mark_completed(self, run_id: str, *, completed_at: datetime | None = None) -> Path:
        completed = completed_at or datetime.now(UTC)
        if completed.tzinfo is None:
            completed = completed.replace(tzinfo=UTC)
        expires = completed.astimezone(UTC) + timedelta(days=self.retention_days)
        path = self._run_dir(run_id) / "retention.json"
        self._atomic_json_write(
            path,
            {
                "retention_version": 1,
                "run_id": run_id,
                "completed_at": completed.astimezone(UTC).isoformat(),
                "expires_at": expires.isoformat(),
                "retention_days": self.retention_days,
            },
        )
        return path

    def list_expired(self, *, now: datetime | None = None) -> list[str]:
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        expired: list[str] = []
        if not self.root.exists():
            return expired
        for run_dir in self.root.iterdir():
            if not run_dir.is_dir():
                continue
            retention_path = run_dir / "retention.json"
            if not retention_path.exists():
                continue
            try:
                metadata = json.loads(retention_path.read_text(encoding="utf-8"))
                expires_at = datetime.fromisoformat(metadata["expires_at"])
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            if expires_at <= current.astimezone(UTC):
                expired.append(run_dir.name)
        return sorted(expired)

    def cleanup_expired(self, *, now: datetime | None = None) -> list[str]:
        removed: list[str] = []
        for run_id in self.list_expired(now=now):
            run_dir = self._run_dir(run_id)
            if run_dir.exists():
                shutil.rmtree(run_dir)
                removed.append(run_id)
        return sorted(removed)

    def delete_run(self, run_id: str) -> None:
        run_dir = self._run_dir(run_id)
        if run_dir.exists():
            shutil.rmtree(run_dir)

    def _run_dir(self, run_id: str) -> Path:
        if not _SAFE_RUN_ID.fullmatch(run_id):
            raise TrajectoryError("invalid_private_run_id")
        path = (self.root / run_id).resolve()
        if path.parent != self.root:
            raise TrajectoryError("private_run_path_escape")
        return path

    @staticmethod
    def _atomic_json_write(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="\n",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            os.replace(temporary, path)
        except OSError as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise TrajectoryError("private_capture_write_failed") from exc
