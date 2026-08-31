from __future__ import annotations

from pathlib import Path

from job_agent.session_orchestrator import (
    OrchestratorDecision,
    OrchestratorExecutionResult,
    plan_next_action,
    run_next_action,
)


def plan_refill(session_dir: Path, user_request: str) -> OrchestratorDecision:
    """检查 session 下一步前置（缺什么 artifact）。薄包装 plan_next_action。

    所有模式都用 offline 规则补齐（不烧 token）。
    """
    return plan_next_action(session_dir, user_request)


def run_refill(session_dir: Path, user_request: str) -> OrchestratorExecutionResult:
    """执行下一步（补齐缺失环节，offline 规则，不烧 token）。薄包装 run_next_action。"""
    return run_next_action(session_dir, user_request)
