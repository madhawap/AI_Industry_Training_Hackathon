"""Tool base types for the central registry.

A *tool* here is anything the planning model may call through the OpenAI
function-calling protocol. Every tool is described by a :class:`ToolSpec`
(name + description + Pydantic args schema + async executor + timeout) and
registered with the :class:`~src.tools.registry.ToolRegistry`. See
ARCHITECTURE.md ("Tool registry") for how to add a new tool.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

InputT = TypeVar("InputT", bound=BaseModel)


class ToolResult(BaseModel):
    """Structured result returned by every tool execution.

    Three shapes are possible:
      - success:            ``success=True``,  ``data`` set, ``error=None``
      - partial success:    ``success=True``,  ``data`` set, ``error`` = warning text
      - failure:            ``success=False``, ``error`` = failure text
    """

    success: bool = True
    data: Any = None
    error: str | None = None

    def as_trace_result(self) -> Any:
        """Flatten into the payload stored in ``tool_trace`` / shown to the LLM."""
        if self.success:
            return self.data if self.error is None else {"data": self.data, "warning": self.error}
        return {"error": self.error or "tool failed"}


class ToolSpec(Generic[InputT]):
    """Definition of a single registered tool.

    Args:
        name: Function name the model calls (must be unique in the registry).
        description: What the model reads when deciding whether/how to call
            the tool. This is effectively a prompt — be precise about argument
            values and when the tool applies.
        args_schema: Pydantic model for the arguments; the registry validates
            raw args against it before the executor ever runs.
        execute: ``async (validated_args) -> ToolResult``.
        timeout: Per-call wall-clock budget enforced by the registry.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        args_schema: type[InputT],
        execute: Callable[[InputT], Awaitable[ToolResult]],
        timeout: float = 15.0,
    ) -> None:
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.execute = execute
        self.timeout = timeout

    def openai_schema(self) -> dict[str, Any]:
        """OpenAI / LiteLLM function-calling schema for the Qwen agent."""
        schema = self.args_schema.model_json_schema()
        # OpenAI expects parameters without $defs noise when possible;
        # keep full schema for correctness with nested models.
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


class ToolCallRecord(BaseModel):
    """One executed tool call — the unit stored in the API's ``tool_trace``."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
