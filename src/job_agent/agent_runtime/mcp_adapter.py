from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, create_model

from job_agent.agent_runtime.contracts import ToolContext, ToolEffect
from job_agent.agent_runtime.tool_registry import ToolRegistry


TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


class MCPClientBridge(Protocol):
    def list_tools(self) -> Any:
        ...

    def call_tool_sync(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        ...


class MCPToolOutput(BaseModel):
    data: Any


def schema_to_model(
    name: str,
    schema: dict[str, Any],
) -> type[BaseModel]:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields = {}

    for field_name, field_schema in properties.items():
        json_type = field_schema.get("type", "string")
        python_type = TYPE_MAP.get(json_type, Any)

        if field_name in required:
            fields[field_name] = (python_type, ...)
        else:
            fields[field_name] = (python_type, None)

    return create_model(
        f"{name}Input",
        **fields,
    )


def build_handler(
    bridge: MCPClientBridge,
    tool_name: str,
):
    def handler(
        value: BaseModel,
        context: ToolContext,
    ) -> MCPToolOutput:
        arguments = value.model_dump()

        result = bridge.call_tool_sync(
            tool_name,
            arguments,
        )

        return MCPToolOutput(data=result)

    return handler


class MCPToolAdapter:
    def __init__(self, bridge: MCPClientBridge):
        self.bridge = bridge

    def register_tool(
        self,
        registry: ToolRegistry,
        tool: dict[str, Any],
        timeout_s: float = 30.0,
    ):
        tool_name = tool["name"]
        description = tool.get("description") or tool_name
        input_schema = tool["input_schema"]

        InputModel = schema_to_model(
            tool_name,
            input_schema,
        )

        handler = build_handler(
            self.bridge,
            tool_name,
        )

        return registry.register(
            name=tool_name,
            description=description,
            input_model=InputModel,
            output_model=MCPToolOutput,
            handler=handler,
            effect=ToolEffect.READ_ONLY,
            timeout_s=timeout_s, 
        )