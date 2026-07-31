"""Central async tool registry used by the Qwen agent.

The registry is the single choke point between the LLM and any side effect:
every tool call the planner emits goes through :meth:`ToolRegistry.execute_one`,
which guarantees three things regardless of how the tool itself behaves:

1. **Validation** — raw JSON args are checked against the tool's Pydantic
   schema before its code runs.
2. **Timeout** — each call is bounded by the tool's own wall-clock budget.
3. **No exceptions escape** — unknown tools, bad args, timeouts and crashes
   all come back as structured ``{"error": ...}`` records the agent can read,
   never as a broken graph run.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import ValidationError

from src.tools.base import ToolCallRecord, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Register, discover, and execute tools concurrently."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec[Any]] = {}

    def register(self, tool: ToolSpec[Any]) -> None:
        """Add a tool; duplicate names are a programming error, so fail loudly."""
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get(self, name: str) -> ToolSpec[Any] | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec[Any]]:
        return list(self._tools.values())

    def openai_tools(self) -> list[dict[str, Any]]:
        """OpenAI function-calling schemas for every registered tool."""
        return [t.openai_schema() for t in self._tools.values()]

    async def execute_one(self, name: str, args: dict[str, Any]) -> ToolCallRecord:
        """Validate and run a single tool call; never raises (see module doc)."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolCallRecord(
                tool=name,
                args=args,
                result={"error": f"Unknown tool: {name}"},
            )
        try:
            validated = tool.args_schema.model_validate(args)
        except ValidationError as exc:
            return ToolCallRecord(
                tool=name,
                args=args,
                result={"error": f"Invalid arguments: {exc.errors()}"},
            )

        try:
            result: ToolResult = await asyncio.wait_for(
                tool.execute(validated),
                timeout=tool.timeout,
            )
        except asyncio.TimeoutError:
            return ToolCallRecord(
                tool=name,
                args=args,
                result={"error": f"Tool timed out after {tool.timeout}s"},
            )
        except Exception as exc:  # noqa: BLE001 — surface to agent, don't crash graph
            logger.exception("Tool %s failed", name)
            return ToolCallRecord(
                tool=name,
                args=args,
                result={"error": str(exc)},
            )

        return ToolCallRecord(tool=name, args=args, result=result.as_trace_result())

    async def execute_many(
        self,
        calls: list[tuple[str, dict[str, Any]]],
    ) -> list[ToolCallRecord]:
        """Run independent tool calls in parallel to cut wall-clock latency."""
        if not calls:
            return []
        if len(calls) == 1:
            name, args = calls[0]
            return [await self.execute_one(name, args)]
        return list(
            await asyncio.gather(*(self.execute_one(name, args) for name, args in calls))
        )


# Module-level singleton — populated at app startup.
registry = ToolRegistry()
