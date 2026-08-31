from __future__ import annotations

import asyncio
import sys
import threading
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class StdioMCPClientBridge:
    def __init__(
        self,
        server_script: str,
    ) -> None:
        self.server_script = server_script

        self._loop = None
        self._session = None
        self._thread = None

        self._ready = threading.Event()
        
        self._stop_event = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._thread_main,
            daemon=True,
        )
        self._thread.start()

        self._ready.wait(timeout=10)

        if self._session is None:
            raise RuntimeError("MCP bridge failed to start")

    def _thread_main(self) -> None:
        asyncio.run(self._run_session())

    async def _run_session(self) -> None:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[self.server_script],
        )

        async with stdio_client(server_params) as (
            read_stream,
            write_stream,
        ):
            async with ClientSession(
                read_stream,
                write_stream,
            ) as session:
                await session.initialize()

                self._session = session
                self._loop = asyncio.get_running_loop()
                self._stop_event = asyncio.Event()

                self._ready.set()

                await self._stop_event.wait()

    def list_tools(self) -> list[dict[str, Any]]:
        future = asyncio.run_coroutine_threadsafe(
            self._session.list_tools(),
            self._loop,
        )

        result = future.result(timeout=10)

        tools = []

        for tool in result.tools:
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
            )

        return tools

    def call_tool_sync(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(
                name,
                arguments,
            ),
            self._loop,
        )

        result = future.result(timeout=30)

        if result.is_error:
            raise RuntimeError(f"MCP tool failed: {name}")

        if result.structured_content is not None:
            return result.structured_content

        return {
            "content": [
                item.model_dump(mode="json")
                for item in result.content
            ]
        }
        
    def close(self) -> None:
        if self._loop is None or self._stop_event is None:
            return

        self._loop.call_soon_threadsafe(
            self._stop_event.set
        )

        if self._thread is not None:
            self._thread.join(timeout=10)

        self._session = None
        self._loop = None
        self._thread = None
        self._stop_event = None