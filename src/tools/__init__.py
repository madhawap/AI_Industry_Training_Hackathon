"""Custom tool framework for the Qwen agent."""

from src.tfql import Store
from src.tools.base import ToolCallRecord, ToolResult, ToolSpec
from src.tools.placeholder import register_placeholder
from src.tools.registry import ToolRegistry, registry
from src.tools.tfql_tool import TOOL_NAME as TFQL_TOOL_NAME
from src.tools.tfql_tool import register_tfql


def register_all_tools(
    reg: ToolRegistry | None = None,
    *,
    default_timeout: float = 15.0,
    store: Store | None = None,
) -> ToolRegistry:
    """Register every built-in tool. Call once at application startup.

    ``store`` is the preloaded TFQL warehouse. When omitted the data tool is
    skipped, which keeps the placeholder-only path usable for smoke tests.
    """
    target = reg or registry
    # Avoid double-registration on reload.
    if target.get("placeholder_tool") is None:
        register_placeholder(target, timeout=min(default_timeout, 5.0))
    if store is not None and target.get(TFQL_TOOL_NAME) is None:
        register_tfql(target, store, timeout=default_timeout)
    return target


__all__ = [
    "ToolCallRecord",
    "ToolResult",
    "ToolSpec",
    "ToolRegistry",
    "registry",
    "register_all_tools",
    "register_tfql",
]
