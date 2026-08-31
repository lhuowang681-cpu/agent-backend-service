from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from job_agent.atomic_io import atomic_write_json


_DIAGNOSTIC_FIELDS = (
    "started_at",
    "finished_at",
    "mode",
    "action_id",
    "status",
    "current_stage",
    "duration_ms",
    "error_code",
    "previous_version_preserved",
    "attempt_count",
    "schema_repair_count",
    "fallback_count",
    "traces",
)

_TRACE_FIELDS = (
    "provider",
    "model",
    "skill_id",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "schema_valid",
    "fallback_used",
    "error_code",
)

_SCHEMA_ERROR_CODES = {
    "schema_error",
    "invalid_json",
    "invalid_response",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def start_run_diagnostic(
    *,
    mode: str,
    action_id: str = "unspecified",
) -> dict[str, Any]:
    return {
        "started_at": utc_now_iso(),
        "_started_perf_counter": perf_counter(),
        "finished_at": None,
        "mode": mode,
        "action_id": action_id,
        "status": "running",
        "current_stage": "preparing_inputs",
        "duration_ms": 0,
        "error_code": None,
        "previous_version_preserved": False,
        "attempt_count": 0,
        "schema_repair_count": 0,
        "fallback_count": 0,
        "traces": [],
    }


def finish_run_diagnostic(
    diagnostic: dict[str, Any],
    *,
    status: str,
    current_stage: str,
    harness=None,
    error_code: str | None = None,
    previous_version_preserved: bool = False,
) -> dict[str, Any]:
    started = float(diagnostic.pop("_started_perf_counter", perf_counter()))
    traces = [
        {
            "provider": trace.provider,
            "model": trace.model,
            "skill_id": trace.skill_id,
            "latency_ms": trace.latency_ms,
            "input_tokens": trace.input_tokens,
            "output_tokens": trace.output_tokens,
            "schema_valid": trace.schema_valid,
            "fallback_used": trace.fallback_used,
            "error_code": trace.error_code,
        }
        for trace in (getattr(harness, "traces", None) or [])
    ]
    schema_repair_count = sum(
        1
        for index, trace in enumerate(traces)
        if index < len(traces) - 1
        and trace.get("error_code") in _SCHEMA_ERROR_CODES
    )
    diagnostic.update(
        {
            "finished_at": utc_now_iso(),
            "status": status,
            "current_stage": current_stage,
            "duration_ms": max(int((perf_counter() - started) * 1000), 0),
            "error_code": error_code,
            "previous_version_preserved": previous_version_preserved,
            "attempt_count": len(traces),
            "schema_repair_count": schema_repair_count,
            "fallback_count": sum(
                1 for trace in traces if trace.get("fallback_used")
            ),
            "traces": traces,
        }
    )
    return diagnostic


def sanitize_run_diagnostic(diagnostic: dict[str, Any]) -> dict[str, Any]:
    sanitized = {
        field: diagnostic.get(field)
        for field in _DIAGNOSTIC_FIELDS
        if field != "traces"
    }
    raw_traces = diagnostic.get("traces")
    traces = raw_traces if isinstance(raw_traces, list) else []
    sanitized["traces"] = [
        {
            field: trace.get(field)
            for field in _TRACE_FIELDS
        }
        for trace in traces
        if isinstance(trace, dict)
    ]
    return sanitized


def write_run_diagnostic(path: Path, diagnostic: dict[str, Any]) -> Path:
    path = Path(path)
    return atomic_write_json(path, sanitize_run_diagnostic(diagnostic))


def load_run_diagnostic(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
