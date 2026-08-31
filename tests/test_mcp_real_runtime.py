from job_agent.agent_runtime.contracts import (
    AgentAction,
    ToolContext,
    ToolObservationStatus,
)
from job_agent.agent_runtime.mcp_adapter import MCPToolAdapter
from job_agent.agent_runtime.mcp_bridge import StdioMCPClientBridge
from job_agent.agent_runtime.policy import PolicyContext, PolicyEngine
from job_agent.agent_runtime.tool_executor import ToolExecutor
from job_agent.agent_runtime.tool_registry import ToolRegistry


def _context() -> ToolContext:
    return ToolContext(
        session_id="session-1",
        run_id="run-1",
        agent_id="mcp-test-agent",
        workspace_root="fixtures/workspace",
        sandbox_root="fixtures/workspace/sandbox",
    )


def test_real_mcp_tool_executes_through_runtime() -> None:
    bridge = StdioMCPClientBridge(
        "tests/fixtures/mcp_add_server.py"
    )
    bridge.start()

    try:
        registry = ToolRegistry()
        adapter = MCPToolAdapter(bridge)

        for tool in bridge.list_tools():
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

    finally:
        bridge.close()
        
def test_real_mcp_invalid_input_is_rejected_before_execution() -> None:
    bridge = StdioMCPClientBridge(
        "tests/fixtures/mcp_add_server.py"
    )
    bridge.start()

    try:
        registry = ToolRegistry()
        adapter = MCPToolAdapter(bridge)

        for tool in bridge.list_tools():
            adapter.register_tool(
                registry=registry,
                tool=tool,
            )

        action = AgentAction(
            action_id="a-invalid",
            tool_name="add",
            tool_arguments={
                "a": "hello",   # add 要求 integer
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

        assert observation.status == ToolObservationStatus.FATAL_ERROR
        assert observation.error_code == "tool_input_error"

    finally:
        bridge.close()
        

def test_real_mcp_tool_failure_becomes_runtime_observation() -> None:
    bridge = StdioMCPClientBridge(
        "tests/fixtures/mcp_add_server.py"
    )
    bridge.start()

    try:
        registry = ToolRegistry()
        adapter = MCPToolAdapter(bridge)

        for tool in bridge.list_tools():
            adapter.register_tool(
                registry=registry,
                tool=tool,
            )

        action = AgentAction(
            action_id="a-fail",
            tool_name="always_fail",
            tool_arguments={},
            expected_observation="failure result",
            progress_claim="run failing MCP tool",
        )

        policy_context = PolicyContext(
            skill_allowed_tools=("always_fail",),
            agent_allowed_tools=("always_fail",),
            runtime_allowed_tools=("always_fail",),
        )

        authorization = PolicyEngine().authorize(
            action,
            registry.get("always_fail").spec,
            policy_context,
        )

        observation = ToolExecutor(registry).execute(
            action,
            authorization,
            _context(),
        )

        assert observation.status == ToolObservationStatus.FATAL_ERROR
        assert observation.error_code == "tool_execution_error"

    finally:
        bridge.close()
        
def test_real_mcp_tool_timeout_becomes_retryable_observation() -> None:
    bridge = StdioMCPClientBridge(
        "tests/fixtures/mcp_add_server.py"
    )
    bridge.start()

    try:
        registry = ToolRegistry()
        adapter = MCPToolAdapter(bridge)

        for tool in bridge.list_tools():
            if tool["name"] == "slow_add":
                registered = adapter.register_tool(
                    registry=registry,
                    tool=tool,
                    timeout_s=0.05,
                )

                print("registered timeout_s:", registered.spec.timeout_s)

        action = AgentAction(
            action_id="a-timeout",
            tool_name="slow_add",
            tool_arguments={
                "a": 10,
                "b": 20,
            },
            expected_observation="sum result",
            progress_claim="run slow MCP tool",
        )

        policy_context = PolicyContext(
            skill_allowed_tools=("slow_add",),
            agent_allowed_tools=("slow_add",),
            runtime_allowed_tools=("slow_add",),
        )

        authorization = PolicyEngine().authorize(
            action,
            registry.get("slow_add").spec,
            policy_context,
        )

        observation = ToolExecutor(registry).execute(
            action,
            authorization,
            _context(),
        )

        assert observation.status == ToolObservationStatus.RETRYABLE_ERROR
        assert observation.error_code == "tool_timeout"

    finally:
        bridge.close()