"""Custom tool framework for the Qwen agent.

Public surface:

- :class:`ToolRegistry` / module-level ``registry`` singleton
- :class:`ToolSpec`, :class:`ToolResult`, :class:`ToolCallRecord` base types
- :func:`register_all_tools` — startup hook that registers every production
  tool against the preloaded TFQL :class:`~src.tfql.Store`

To add a new tool, copy ``src/tools/placeholder.py`` (the annotated template)
and register it here. Full walkthrough: ARCHITECTURE.md → "Adding a new tool".
"""

from src.tfql import Store
from src.tools.base import ToolCallRecord, ToolResult, ToolSpec
from src.tools.registry import ToolRegistry, registry
from src.tools.tfql_tool import TOOL_NAME as TFQL_TOOL_NAME
from src.tools.tfql_tool import register_tfql


def register_all_tools(
    reg: ToolRegistry | None = None,
    *,
    default_timeout: float = 15.0,
    store: Store | None = None,
) -> ToolRegistry:
    """Register every production tool. Called once at application startup.

    ``store`` is the preloaded TFQL warehouse; ``execute_plan`` is currently
    the only production tool, so nothing is registered when it is omitted.
    Guarded against double-registration so hot reloads stay safe.
    """
    target = reg or registry
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
