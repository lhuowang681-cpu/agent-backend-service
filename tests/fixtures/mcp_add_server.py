from mcp.server.mcpserver import MCPServer
import asyncio

server = MCPServer(
    name="job-agent-test-mcp",
)


@server.tool(
    description="Add two integers",
)
def add(a: int, b: int) -> int:
    return a + b

@server.tool(
    description="Always fail for testing",
)
def always_fail() -> str:
    raise RuntimeError("simulated MCP failure")

@server.tool(
    description="Sleep before returning",
)
async def slow_add(a: int, b: int) -> int:
    await asyncio.sleep(0.2)
    return a + b


if __name__ == "__main__":
    server.run("stdio")