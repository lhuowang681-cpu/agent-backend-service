from __future__ import annotations

from time import perf_counter
from typing import Protocol, Sequence

from pydantic import BaseModel

from job_agent.llm.provider import ProviderResult, ToolSpec, TraceContext
from job_agent.llm.providers.mock import result_from_raw_output


class LocalStructuredRunner(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class LocalTransformersProvider:
    provider_name = "transformers"

    def __init__(self, *, runner: LocalStructuredRunner, model: str, clock=perf_counter) -> None:
        self.runner = runner
        self.model = model
        self._clock = clock

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        tools: Sequence[ToolSpec],
        temperature: float,
        max_output_tokens: int,
        trace: TraceContext,
    ) -> ProviderResult:
        if tools:
            raise ValueError("local Transformers provider does not support tools in Phase 1")
        prompt = f"{system_prompt}\n\n{user_prompt}"
        started = self._clock()
        raw_output = self.runner.generate(prompt)
        latency_ms = int((self._clock() - started) * 1000)
        return result_from_raw_output(
            provider=self.provider_name,
            model=self.model,
            raw_output=raw_output,
            output_schema=output_schema,
            trace=trace,
            latency_ms=latency_ms,
        )
