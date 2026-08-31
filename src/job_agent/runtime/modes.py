from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from job_agent.schemas import StrictModel


class RuntimeMode(str, Enum):
    OFFLINE_RULE = "offline_rule"
    AGENT_API = "agent_api"
    AGENT_LOCAL = "agent_local"
    SERVER_AGENT_LOCAL_LORA = "server_agent_local_lora"


class AgentRuntimeConfig(StrictModel):
    mode: RuntimeMode = RuntimeMode.OFFLINE_RULE
    skill_root: str | None = None
    provider: str | None = None
    model: str = "fixture-model"
    timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    temperature: float = Field(default=0.0, ge=0)
    max_output_tokens: int = Field(default=1024, gt=0)
    max_parallel_nodes: int = Field(default=4, gt=0)

    @property
    def uses_semantic_agents(self) -> bool:
        return self.mode != RuntimeMode.OFFLINE_RULE

    @model_validator(mode="after")
    def validate_agent_mode(self):
        if self.uses_semantic_agents and not self.skill_root:
            raise ValueError("skill_root is required for agent runtime modes")
        if self.provider is None:
            self.provider = {
                RuntimeMode.OFFLINE_RULE: "rule_fallback",
                RuntimeMode.AGENT_API: "openai_compatible",
                RuntimeMode.AGENT_LOCAL: "transformers",
                RuntimeMode.SERVER_AGENT_LOCAL_LORA: "transformers",
            }[self.mode]
        allowed = {
            RuntimeMode.OFFLINE_RULE: {"mock", "rule_fallback"},
            RuntimeMode.AGENT_API: {"mock", "openai_compatible", "anthropic_compatible"},
            RuntimeMode.AGENT_LOCAL: {"mock", "transformers"},
            RuntimeMode.SERVER_AGENT_LOCAL_LORA: {"mock", "transformers"},
        }
        if self.provider not in allowed[self.mode]:
            raise ValueError(f"provider {self.provider!r} is not valid for mode {self.mode.value}")
        return self
