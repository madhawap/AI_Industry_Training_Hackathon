"""Tool base types for the central registry."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

InputT = TypeVar("InputT", bound=BaseModel)

# async (validated_input) -> structured result
ToolFn = Callable[[BaseModel], Awaitable[Any]]


class ToolResult(BaseModel):
    """Structured result returned by every tool execution."""

    success: bool = True
    data: Any = None
    error: str | None = None

    def as_trace_result(self) -> Any:
        if self.success:
            return self.data if self.error is None else {"data": self.data, "warning": self.error}
        return {"error": self.error or "tool failed"}


class ToolSpec(Generic[InputT]):
    """Definition of a single registered tool."""

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
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
