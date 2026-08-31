from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# 模块级 import：测试通过 patch("job_agent.ui.modes.SkillRegistry.from_sources")
# 注入 fake registry，因此 SkillRegistry 必须是模块属性，不能用函数内延迟 import
from job_agent.llm.harness import LLMHarness
from job_agent.llm.providers.anthropic_compatible import AnthropicCompatibleProvider
from job_agent.llm.providers.mock import MockLLMProvider
from job_agent.llm.skill_registry import SkillRegistry
from job_agent.ui.local_config import load_local_runtime_defaults


class WorkbenchMode(str, Enum):
    OFFLINE_RULE = "offline_rule"
    AGENT_API_MOCK = "agent_api_mock"
    AGENT_API_LIVE = "agent_api_live"


@dataclass(frozen=True)
class RuntimeContext:
    # offline 模式三者皆为 None；mock/live 模式三者皆有值
    provider: Any | None = None
    harness: Any | None = None
    registry: Any | None = None

    @property
    def is_semantic(self) -> bool:
        return self.harness is not None


@dataclass(frozen=True)
class RuntimeReadiness:
    ready: bool
    message: str
    api_key_source: str | None = None


def check_live_runtime(
    *,
    skill_root: str | None,
    base_url: str,
    model: str,
) -> RuntimeReadiness:
    defaults = load_local_runtime_defaults()
    missing = []
    if not defaults.api_key:
        missing.append("JOB_AGENT_LIVE_API_KEY")
    if not skill_root or not Path(skill_root).exists():
        missing.append("skill_root")
    if not base_url.strip():
        missing.append("base_url")
    if not model.strip():
        missing.append("model")
    if missing:
        return RuntimeReadiness(
            False,
            "Live runtime not ready; missing " + ", ".join(missing),
        )
    return RuntimeReadiness(
        True,
        "Live runtime ready",
        defaults.api_key_source,
    )


def build_live_runtime(
    *,
    skill_root: str,
    base_url: str,
    model: str,
    timeout_s: float = 60.0,
) -> RuntimeContext:
    readiness = check_live_runtime(
        skill_root=skill_root,
        base_url=base_url,
        model=model,
    )
    if not readiness.ready:
        raise RuntimeError(readiness.message)
    return build_runtime(
        WorkbenchMode.AGENT_API_LIVE,
        skill_root=skill_root,
        base_url=base_url,
        model=model,
        timeout_s=timeout_s,
    )


def build_runtime(
    mode: WorkbenchMode,
    *,
    skill_root: str | None = None,
    mock_responses: list[Any] | None = None,
    api_key_env: str = "JOB_AGENT_LIVE_API_KEY",
    base_url: str | None = None,
    model: str = "glm-4.7",
    timeout_s: float = 60.0,
) -> RuntimeContext:
    """构造 semantic 运行时（provider/harness/registry）。offline 返回空 context。

    Live key 只从 api_key_env 指定的环境变量读取。
    调用方不得将 key 写入 artifact、设置文件或日志。
    """
    if mode == WorkbenchMode.OFFLINE_RULE:
        return RuntimeContext()

    if skill_root is None:
        raise ValueError(f"{mode.value} 模式需要 skill_root")

    registry = SkillRegistry.from_sources(skill_root=skill_root)

    if mode == WorkbenchMode.AGENT_API_MOCK:
        if mock_responses is None:
            raise ValueError("AGENT_API_MOCK 模式需要 mock_responses")
        provider = MockLLMProvider(mock_responses, model=model)
    elif mode == WorkbenchMode.AGENT_API_LIVE:
        if base_url is None:
            raise ValueError("AGENT_API_LIVE 模式需要 base_url")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"AGENT_API_LIVE 模式需要环境变量 {api_key_env}"
                "（key 不硬编码/不回显）"
            )
        provider = AnthropicCompatibleProvider(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
        )
    else:  # pragma: no cover — Enum 已穷举
        raise ValueError(f"unsupported mode: {mode}")

    return RuntimeContext(provider=provider, harness=LLMHarness(provider), registry=registry)
