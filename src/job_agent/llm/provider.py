from __future__ import annotations

from typing import Any, Generic, Protocol, Sequence, TypeVar

from pydantic import BaseModel, Field

from job_agent.schemas import StrictModel


StructuredT = TypeVar("StructuredT", bound=BaseModel)


class ProviderError(RuntimeError):
    error_code = "provider_error"
    retryable = False


class ProviderTransportError(ProviderError):
    error_code = "transport_error"
    retryable = True


class ProviderTimeoutError(ProviderTransportError):
    error_code = "timeout"


class ProviderRateLimitError(ProviderTransportError):
    error_code = "rate_limit"


class ProviderHTTPError(ProviderError):
    error_code = "http_error"


class ProviderContractError(ProviderError):
    error_code = "provider_contract_error"


class ToolSpec(StrictModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class TraceContext(StrictModel):
    session_id: str
    run_id: str
    node_id: str
    skill_id: str
    skill_version: str
    prompt_version: str


class ProviderTrace(StrictModel):
    provider: str
    model: str
    skill_id: str
    skill_version: str
    prompt_version: str
    latency_ms: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    schema_valid: bool
    fallback_used: bool = False
    error_code: str | None = None
    error_detail: str | None = None


class ProviderResult(ProviderTrace):
    """One provider call result; raw output is internal audit data only."""

    raw_output: str | None = None
    parsed_output: dict[str, Any] | None = None


class ToolUseResult(ProviderTrace):
    """One native provider tool-use turn with sanitized trace metadata."""

    tool_use_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    tool_input: dict[str, Any] = Field(default_factory=dict)


class AgentResult(StrictModel, Generic[StructuredT]):
    value: StructuredT
    trace: ProviderTrace
    warnings: list[str] = Field(default_factory=list)
    fallback_used: bool = False


class LLMProvider(Protocol):
    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[StructuredT],
        tools: Sequence[ToolSpec],
        temperature: float,
        max_output_tokens: int,
        trace: TraceContext,
    ) -> ProviderResult:
        """Return one auditable structured-generation attempt."""


class ToolUseProvider(Protocol):
    def generate_tool_use(
        self,
        *,
        system_prompt: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[ToolSpec],
        temperature: float,
        max_output_tokens: int,
        trace: TraceContext,
    ) -> ToolUseResult:
        """Return exactly one native provider tool-use decision."""
