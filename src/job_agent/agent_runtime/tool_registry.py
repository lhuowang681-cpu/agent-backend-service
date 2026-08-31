from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable, Generic, Iterable, TypeVar

from pydantic import BaseModel

from job_agent.agent_runtime.contracts import RuntimeToolSpec, ToolContext, ToolEffect
from job_agent.agent_runtime.failures import ToolRegistryError
from job_agent.llm.provider import ToolSpec


InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)
ToolHandler = Callable[[InputT, ToolContext], OutputT]


@dataclass(frozen=True)
class RegisteredTool(Generic[InputT, OutputT]):
    spec: RuntimeToolSpec
    input_model: type[InputT]
    output_model: type[OutputT]
    handler: ToolHandler[InputT, OutputT]


class ToolRegistry:
    def __init__(self, tools: Iterable[RegisteredTool] = ()) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        for tool in tools:
            self.register_tool(tool)

    def register(
        self,
        *,
        name: str,
        description: str,
        input_model: type[InputT],
        output_model: type[OutputT],
        handler: ToolHandler[InputT, OutputT],
        effect: ToolEffect = ToolEffect.READ_ONLY,
        timeout_s: float = 30.0,
        idempotent: bool = True,
        requires_approval: bool | None = None,
        sandbox_only: bool = False,
    ) -> RegisteredTool[InputT, OutputT]:
        if requires_approval is None:
            requires_approval = effect in {
                ToolEffect.LOCAL_STATE_MUTATION,
                ToolEffect.EXTERNAL_DRAFT,
                ToolEffect.EXTERNAL_REVERSIBLE_WRITE,
                ToolEffect.EXTERNAL_IRREVERSIBLE_WRITE,
                ToolEffect.SENSITIVE_READ,
            }
        tool = RegisteredTool(
            spec=RuntimeToolSpec(
                name=name,
                description=description,
                input_schema=input_model.model_json_schema(),
                output_schema=output_model.model_json_schema(),
                effect=effect,
                timeout_s=timeout_s,
                idempotent=idempotent,
                requires_approval=requires_approval,
                sandbox_only=sandbox_only,
            ),
            input_model=input_model,
            output_model=output_model,
            handler=handler,
        )
        self.register_tool(tool)
        return tool

    def register_tool(self, tool: RegisteredTool) -> None:
        name = tool.spec.name
        if name in self._tools:
            raise ToolRegistryError(f"duplicate tool: {name}")
        if tool.spec.input_schema != tool.input_model.model_json_schema():
            raise ToolRegistryError(f"input schema drift: {name}")
        if tool.spec.output_schema != tool.output_model.model_json_schema():
            raise ToolRegistryError(f"output schema drift: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolRegistryError(f"unknown tool: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def fingerprint(self) -> str:
        payload = [
            self.get(name).spec.model_dump(mode="json")
            for name in self.names()
        ]
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def provider_specs(self, names: Iterable[str] | None = None) -> list[ToolSpec]:
        selected = self.names() if names is None else tuple(names)
        return [
            ToolSpec(
                name=self.get(name).spec.name,
                description=self.get(name).spec.description,
                input_schema=self.get(name).spec.input_schema,
            )
            for name in selected
        ]
