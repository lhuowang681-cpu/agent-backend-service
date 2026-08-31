from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from job_agent.memory.contracts import DEFAULT_USER_STATE_PATH, UserState


class UserStateVersionError(ValueError):
    pass


class MemoryPolicyError(ValueError):
    pass


_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*\S+"
)


class UserStateStore:
    """Atomic, fresh-read JSON store with an explicit user namespace."""

    def __init__(self, path: Path | str = DEFAULT_USER_STATE_PATH) -> None:
        self.path = Path(path)

    def load(self, *, user_id: str) -> UserState:
        users = self._read_users()
        payload = users.get(user_id)
        if payload is None:
            return UserState(user_id=user_id)
        return UserState.model_validate(payload)

    def save(
        self,
        *,
        user_id: str,
        state: UserState,
        expected_version: int | None = None,
    ) -> UserState:
        if state.user_id != user_id:
            raise ValueError("state user_id does not match requested namespace")
        for record in state.memories:
            if _CREDENTIAL_PATTERN.search(record.content):
                raise MemoryPolicyError("credential-like content must not be written to memory")
        users = self._read_users()
        current = UserState.model_validate(users[user_id]) if user_id in users else UserState(user_id=user_id)
        if expected_version is not None and current.version != expected_version:
            raise UserStateVersionError(
                f"user state version mismatch: expected {expected_version}, found {current.version}"
            )
        saved = state.model_copy(update={"version": current.version + 1})
        users[user_id] = saved.model_dump(mode="json")
        self._write_users(users)
        return self.load(user_id=user_id)

    def delete(self, *, user_id: str) -> None:
        users = self._read_users()
        if user_id not in users:
            return
        del users[user_id]
        if not users:
            if self.path.exists():
                self.path.unlink()
            return
        self._write_users(users)

    def _read_users(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("users"), dict):
            return {str(key): dict(value) for key, value in payload["users"].items()}
        if isinstance(payload, dict) and "user" in payload:
            legacy = dict(payload)
            legacy.setdefault("user_id", "default-user")
            legacy.setdefault("version", 1)
            legacy.setdefault("memories", [])
            legacy.setdefault("meta", {})
            return {"default-user": legacy}
        raise ValueError("invalid user state document")

    def _write_users(self, users: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {"version": 1, "users": users}
        try:
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_path.replace(self.path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        json.loads(self.path.read_text(encoding="utf-8"))
