from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from job_agent.agent_runtime.contracts import ToolEffect
from job_agent.agent_runtime.tool_registry import ToolRegistry


@dataclass(frozen=True)
class MCPToolPolicyConfig:
    effect: ToolEffect
    idempotent: bool = True
    requires_approval: bool = False
    timeout_s: float = 30.0