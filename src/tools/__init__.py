"""Custom tool framework for the Qwen agent."""

from src.tools.base import ToolCallRecord, ToolResult, ToolSpec
from src.tools.placeholder import register_placeholder
from src.tools.registry import ToolRegistry, registry


def register_all_tools(reg: ToolRegistry | None = None, *, default_timeout: float = 15.0) -> ToolRegistry:
    """Register every built-in tool. Call once at application startup."""
    target = reg or registry
    # Avoid double-registration on reload.
    if target.get("placeholder_tool") is None:
        register_placeholder(target, timeout=min(default_timeout, 5.0))
    return target


__all__ = [
    "ToolCallRecord",
    "ToolResult",
    "ToolSpec",
    "ToolRegistry",
    "registry",
    "register_all_tools",
]
