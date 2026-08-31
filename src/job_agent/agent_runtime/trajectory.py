from __future__ import annotations

import json
import re
import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from job_agent.agent_runtime.events import AgentEvent, parse_agent_event
from job_agent.agent_runtime.failures import TrajectoryError


class CaptureLevel(str, Enum):
    METADATA = "metadata"
    SANITIZED = "sanitized"
    FULL_PRIVATE = "full_private"


_NEVER_CAPTURE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "credentials",
    "otp",
    "password",
    "refresh_token",
    "access_token",
    "client_secret",
}
_PRIVATE_CONTENT_KEYS = {
    "prompt",
    "raw_prompt",
    "raw_response",
    "resume",
    "resume_text",
    "tool_input",
    "tool_output",
    "artifact_snapshot",
}
_AUTH_VALUE = re.compile(r"(?i)^\s*(?:bearer|basic)\s+\S+")


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def sanitize_capture_payload(value: Any, *, level: CaptureLevel | str) -> Any:
    """Redact never-capture material and apply the selected content boundary."""
    capture_level = CaptureLevel(level)
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            normalized = _normalized_key(str(key))
            if normalized == "authorization" and isinstance(item, dict):
                sanitized[key] = sanitize_capture_payload(item, level=capture_level)
            elif normalized in _NEVER_CAPTURE_KEYS:
                sanitized[key] = "[REDACTED_CREDENTIAL]"
            elif capture_level != CaptureLevel.FULL_PRIVATE and normalized in _PRIVATE_CONTENT_KEYS:
                sanitized[key] = "[REDACTED_PRIVATE_CONTENT]"
            else:
                sanitized[key] = sanitize_capture_payload(item, level=capture_level)
        return sanitized
    if isinstance(value, list):
        return [sanitize_capture_payload(item, level=capture_level) for item in value]
    if isinstance(value, tuple):
        return [sanitize_capture_payload(item, level=capture_level) for item in value]
    if isinstance(value, str) and _AUTH_VALUE.match(value):
        return "[REDACTED_CREDENTIAL]"
    return value


def _metadata_payload(event: AgentEvent) -> dict[str, Any]:
    payload = event.model_dump(mode="json")
    event_type = event.event_type
    if event_type == "run_started":
        payload["goal"] = {}
    elif event_type == "model_decision":
        decision = payload["decision"]
        if decision["kind"] == "action":
            decision["tool_arguments"] = {}
            decision["expected_observation"] = "[omitted]"
            decision["progress_claim"] = "[omitted]"
        elif decision["kind"] == "finish":
            decision["result"] = {}
            decision["unresolved_items"] = []
        else:
            decision["request"]["prompt"] = "[omitted]"
            decision["request"]["context_summary"] = "[omitted]"
            decision["progress_claim"] = "[omitted]"
    elif event_type == "policy_decision":
        request = payload["authorization"].get("approval_request")
        if request is not None:
            request["arguments_preview"] = {}
    elif event_type == "tool_observed":
        payload["observation"]["data"] = {}
        payload["observation"]["evidence_refs"] = []
    elif event_type == "verification":
        payload["verification"]["feedback"] = "[omitted]"
    elif event_type == "run_finished":
        payload["result"] = {}
    elif event_type == "approval_resolved":
        payload["approval"]["reason"] = "[omitted]"
    elif event_type == "user_input_resolved":
        payload["response"]["value"] = {}
    return payload


def _digest_summary(value: Any) -> dict[str, str]:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "_capture": "sanitized",
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def _sanitized_event_payload(event: AgentEvent) -> dict[str, Any]:
    payload = event.model_dump(mode="json")
    event_type = event.event_type
    if event_type == "run_started":
        payload["goal"] = _digest_summary(payload["goal"])
    elif event_type == "model_decision":
        decision = payload["decision"]
        if decision["kind"] == "action":
            decision["tool_arguments"] = _digest_summary(decision["tool_arguments"])
            decision["expected_observation"] = "[sanitized]"
            decision["progress_claim"] = "[sanitized]"
        elif decision["kind"] == "finish":
            decision["result"] = _digest_summary(decision["result"])
            decision["unresolved_items"] = []
        else:
            decision["request"]["prompt"] = "[sanitized]"
            decision["request"]["context_summary"] = "[sanitized]"
            decision["progress_claim"] = "[sanitized]"
    elif event_type == "policy_decision":
        request = payload["authorization"].get("approval_request")
        if request is not None:
            request["arguments_preview"] = _digest_summary(request["arguments_preview"])
    elif event_type == "tool_observed":
        payload["observation"]["data"] = _digest_summary(payload["observation"]["data"])
    elif event_type == "verification":
        payload["verification"]["feedback"] = "[sanitized]"
    elif event_type == "run_finished":
        payload["result"] = _digest_summary(payload["result"])
    elif event_type == "approval_resolved":
        payload["approval"]["reason"] = "[sanitized]"
    elif event_type == "user_input_resolved":
        payload["response"]["value"] = _digest_summary(payload["response"]["value"])
    return payload


def capture_event_payload(event: AgentEvent, *, level: CaptureLevel | str) -> dict[str, Any]:
    capture_level = CaptureLevel(level)
    if capture_level == CaptureLevel.METADATA:
        payload = _metadata_payload(event)
    elif capture_level == CaptureLevel.SANITIZED:
        payload = _sanitized_event_payload(event)
    else:
        payload = event.model_dump(mode="json")
    return sanitize_capture_payload(payload, level=capture_level)


class TrajectorySink(Protocol):
    def append(self, event: AgentEvent) -> None:
        ...


class InMemoryTrajectory:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def append(self, event: AgentEvent) -> None:
        self.events.append(event)


class JsonlTrajectoryStore:
    def __init__(
        self,
        path: Path | str,
        *,
        capture_level: CaptureLevel | str = CaptureLevel.SANITIZED,
        _private_storage_validated: bool = False,
    ) -> None:
        self.path = Path(path)
        self.capture_level = CaptureLevel(capture_level)
        if self.capture_level == CaptureLevel.FULL_PRIVATE and not _private_storage_validated:
            raise TrajectoryError("full_private_requires_validated_private_store")
        self._identity: tuple[str, str, str] | None = None

    def append(self, event: AgentEvent) -> None:
        identity = (event.session_id, event.run_id, event.agent_id)
        if self._identity is None:
            existing = self.load() if self.path.exists() else []
            if existing:
                self._identity = (
                    existing[0].session_id,
                    existing[0].run_id,
                    existing[0].agent_id,
                )
            else:
                self._identity = identity
        if identity != self._identity:
            raise TrajectoryError("trajectory identity mismatch")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = capture_event_payload(event, level=self.capture_level)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    def load(self) -> list[AgentEvent]:
        if not self.path.exists():
            return []
        events: list[AgentEvent] = []
        expected_identity: tuple[str, str, str] | None = None
        previous_step = -1
        for line_number, raw_line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                event = parse_agent_event(payload)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                raise TrajectoryError(f"invalid trajectory event at line {line_number}") from exc
            identity = (event.session_id, event.run_id, event.agent_id)
            expected_identity = expected_identity or identity
            if identity != expected_identity:
                raise TrajectoryError(f"trajectory identity mismatch at line {line_number}")
            if event.step_index < previous_step:
                raise TrajectoryError(f"trajectory step regression at line {line_number}")
            previous_step = event.step_index
            events.append(event)
        return events
