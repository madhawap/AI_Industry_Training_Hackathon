"""Placeholder tool — template for adding new Qwen-callable tools."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.tools.base import ToolResult, ToolSpec
from src.tools.registry import ToolRegistry


class PlaceholderInput(BaseModel):
    value: str = Field(..., description="Example string argument for the placeholder tool")


async def _run_placeholder(payload: PlaceholderInput) -> ToolResult:
    return ToolResult(
        success=True,
        data="Placeholder tool executed successfully",
    )


def register_placeholder(registry: ToolRegistry, timeout: float = 5.0) -> None:
    registry.register(
        ToolSpec(
            name="placeholder_tool",
            description=(
                "A simple placeholder tool used to verify the tool-calling path. "
                "Call it when you need a demo tool result or when no other tool applies."
            ),
            args_schema=PlaceholderInput,
            execute=_run_placeholder,
            timeout=timeout,
        )
    )
