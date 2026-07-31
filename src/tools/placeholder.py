"""Annotated template for adding a new agent-callable tool.

This module is NOT registered in production — ``register_all_tools`` skips it
because exposing a demo tool tempts the planner into calling it instead of a
real data tool. Keep it as the copy-me starting point for new tools.

To create a new tool:

1. Copy this file, e.g. ``src/tools/my_tool.py``.
2. Define a Pydantic input model — the registry validates the model's raw
   JSON arguments against it before your code runs, and its JSON schema is
   what the planner sees as the function signature.
3. Implement ``async (validated_input) -> ToolResult``. Raise nothing for
   expected failures; return ``ToolResult(success=False, error=...)`` so the
   agent can react instead of the graph crashing.
4. Write a ``register_*`` helper like the one below, and call it from
   ``register_all_tools`` in ``src/tools/__init__.py``.

See ARCHITECTURE.md → "Adding a new tool" for the full walkthrough.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.tools.base import ToolResult, ToolSpec
from src.tools.registry import ToolRegistry


class PlaceholderInput(BaseModel):
    """Arguments schema — field descriptions become part of the tool schema."""

    value: str = Field(..., description="Example string argument for the placeholder tool")


async def _run_placeholder(payload: PlaceholderInput) -> ToolResult:
    """The executor: receives already-validated input, returns a ToolResult."""
    return ToolResult(
        success=True,
        data="Placeholder tool executed successfully",
    )


def register_placeholder(registry: ToolRegistry, timeout: float = 5.0) -> None:
    """Register the placeholder (call manually — not part of production setup)."""
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
