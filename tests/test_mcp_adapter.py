from typing import Any

from job_agent.agent_runtime.contracts import ToolContext
from job_agent.agent_runtime.mcp_adapter import (
    MCPToolAdapter,
    MCPToolOutput,
)
from job_agent.agent_runtime.tool_registry import ToolRegistry
from job_agent.agent_runtime.contracts import (
    AgentAction,
    ToolContext,
    ToolObservationStatus,
)
from job_agent.agent_runtime.policy import PolicyContext, PolicyEngine
from job_agent.agent_runtime.tool_executor import ToolExecutor


class FakeMCPBridge:
    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "add",
                "description": "Add two integers",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            }
        ]

    def call_tool_sync(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        assert name == "add"
        return {
            "result": arguments["a"] + arguments["b"]
        }


def _context() -> ToolContext:
    return ToolContext(
        session_id="session-1",
        run_id="run-1",
        agent_id="mcp-test-agent",
        workspace_root="fixtures/workspace",
        sandbox_root="fixtures/workspace/sandbox",
    )


def test_mcp_adapter_register_and_call() -> None:
    bridge = FakeMCPBridge()
    registry = ToolRegistry()

    adapter = MCPToolAdapter(bridge)

    tool = bridge.list_tools()[0]

    registered = adapter.register_tool(
        registry=registry,
        tool=tool,
    )

    assert registry.get("add") is registered
    assert registered.spec.name == "add"
    assert registered.output_model is MCPToolOutput

    value = registered.input_model.model_validate(
        {
            "a": 10,
            "b": 20,
        }
    )

    result = registered.handler(
        value,
        _context,
    )

    assert isinstance(result, MCPToolOutput)
    assert result.data == {"result": 30}
    
def test_mcp_tool_executes_through_runtime() -> None:
    bridge = FakeMCPBridge()
    registry = ToolRegistry()
    adapter = MCPToolAdapter(bridge)

    tool = bridge.list_tools()[0]
    adapter.register_tool(
        registry=registry,
        tool=tool,
    )

    action = AgentAction(
        action_id="a-1",
        tool_name="add",
        tool_arguments={
            "a": 10,
            "b": 20,
        },
        expected_observation="sum result",
        progress_claim="add two integers",
    )

    policy_context = PolicyContext(
        skill_allowed_tools=("add",),
        agent_allowed_tools=("add",),
        runtime_allowed_tools=("add",),
    )

    authorization = PolicyEngine().authorize(
        action,
        registry.get("add").spec,
        policy_context,
    )

    observation = ToolExecutor(registry).execute(
        action,
        authorization,
        _context(),
    )

    assert observation.status == ToolObservationStatus.OK
    assert observation.data == {
        "data": {
            "result": 30,
        }
    }