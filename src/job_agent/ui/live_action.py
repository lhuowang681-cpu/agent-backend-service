from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from job_agent.agents.base import AgentGuardError
from job_agent.atomic_io import atomic_write_bytes
from job_agent.llm.harness import LLMInvocationError
from job_agent.llm.provider import ProviderError
from job_agent.ui.run_diagnostics import (
    finish_run_diagnostic,
    start_run_diagnostic,
    write_run_diagnostic,
)


@dataclass(frozen=True)
class LiveActionResult:
    ok: bool
    value: Any = None
    error_code: str | None = None
    user_message: str | None = None
    previous_version_preserved: bool = False
    diagnostic: dict[str, Any] | None = None
    diagnostic_path: Path | None = None


def _snapshot_artifacts(
    paths: Sequence[Path | str],
) -> dict[Path, bytes | None]:
    snapshots: dict[Path, bytes | None] = {}
    for raw_path in paths:
        path = Path(raw_path)
        try:
            snapshots[path] = path.read_bytes() if path.is_file() else None
        except OSError:
            snapshots[path] = None
    return snapshots


def _restore_artifacts(snapshots: dict[Path, bytes | None]) -> bool:
    restored = True
    for path, previous in snapshots.items():
        try:
            if previous is None:
                if path.is_file():
                    path.unlink()
            else:
                atomic_write_bytes(path, previous)
        except OSError:
            restored = False
    return restored


def _resolve_harness(harness: Any) -> Any:
    return harness() if callable(harness) else harness


def _classify_error(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, AgentGuardError):
        return (
            exc.error_code,
            "模型结果未通过事实校验，系统已拒绝写入。",
        )
    if isinstance(exc, LLMInvocationError):
        messages = {
            "schema_error": (
                "模型返回的结构化结果不完整，自动修复后仍未通过校验。"
            ),
            "invalid_json": "模型本次没有返回可读取的结构化结果。",
            "invalid_response": "模型服务本次返回内容不完整。",
            "timeout": "模型服务响应超时，请直接重试。",
            "rate_limit": "模型服务当前请求较多，请稍后重试。",
        }
        return (
            exc.error_code,
            messages.get(exc.error_code, "模型调用暂时失败，请重试。"),
        )
    if isinstance(exc, ProviderError):
        return (
            exc.error_code,
            "模型服务调用失败，请检查模型配置后重试。",
        )
    if isinstance(exc, OSError):
        return (
            "artifact_write_error",
            "新版本保存失败，系统没有提交本次更新。",
        )
    lowered = str(exc).casefold()
    exception_name = type(exc).__name__.casefold()
    if (
        "authentication" in exception_name
        or "permission" in exception_name
        or "api key" in lowered
        or "api_key" in lowered
        or "credential" in lowered
    ):
        return (
            "provider_config_error",
            "模型凭据或服务配置不可用，请检查本机配置后重试。",
        )
    if "timeout" in exception_name or "timed out" in lowered:
        return (
            "timeout",
            "模型服务响应超时，请直接重试。",
        )
    if "ratelimit" in exception_name or "rate limit" in lowered:
        return (
            "rate_limit",
            "模型服务当前请求较多，请稍后重试。",
        )
    if (
        "connection" in exception_name
        or "connection refused" in lowered
        or "connect error" in lowered
    ):
        return (
            "transport_error",
            "暂时无法连接模型服务，请检查服务地址或网络后重试。",
        )
    if "budget" in lowered and ("exceed" in lowered or "耗尽" in lowered):
        return (
            "budget_exceeded",
            "本轮执行预算已耗尽，现有进度已经保留。",
        )
    return (
        "unexpected_error",
        "操作暂时失败，请重试；如果持续失败，请下载脱敏诊断。",
    )


def _safe_write_diagnostic(
    path: Path,
    diagnostic: dict[str, Any],
) -> Path | None:
    try:
        return write_run_diagnostic(path, diagnostic)
    except OSError:
        return None


def execute_live_action(
    *,
    action_id: str,
    operation: Callable[[], Any],
    session_dir: Path | str,
    harness: Any = None,
    current_stage: str | Callable[[], str],
    previous_artifacts: Sequence[Path | str] = (),
    mode: str = "agent_api_live",
    diagnostic_path: Path | str | None = None,
) -> LiveActionResult:
    session_path = Path(session_dir)
    target_diagnostic_path = (
        Path(diagnostic_path)
        if diagnostic_path is not None
        else session_path / ".internal" / "ui_run_diagnostics.json"
    )
    diagnostic = start_run_diagnostic(mode=mode, action_id=action_id)
    resolve_stage = (
        current_stage
        if callable(current_stage)
        else lambda: current_stage
    )
    diagnostic["current_stage"] = resolve_stage()
    snapshots = _snapshot_artifacts(previous_artifacts)
    try:
        value = operation()
    except Exception as exc:
        previous_version_preserved = (
            any(value is not None for value in snapshots.values())
            and _restore_artifacts(snapshots)
        )
        error_code, user_message = _classify_error(exc)
        if previous_version_preserved:
            user_message += "本次更新失败，当前仍展示上一成功版本。"
        finish_run_diagnostic(
            diagnostic,
            status="failed",
            current_stage=resolve_stage(),
            harness=_resolve_harness(harness),
            error_code=error_code,
            previous_version_preserved=previous_version_preserved,
        )
        written_path = _safe_write_diagnostic(
            target_diagnostic_path,
            diagnostic,
        )
        return LiveActionResult(
            ok=False,
            error_code=error_code,
            user_message=user_message,
            previous_version_preserved=previous_version_preserved,
            diagnostic=diagnostic,
            diagnostic_path=written_path,
        )

    finish_run_diagnostic(
        diagnostic,
        status="completed",
        current_stage="completed",
        harness=_resolve_harness(harness),
    )
    written_path = _safe_write_diagnostic(target_diagnostic_path, diagnostic)
    return LiveActionResult(
        ok=True,
        value=value,
        diagnostic=diagnostic,
        diagnostic_path=written_path,
    )
