"""LangGraph shared state for the query workflow."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


def _merge_traces(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return left + right


class GraphState(TypedDict):
    question: str
    messages: list[dict[str, Any]]
    pending_tool_calls: list[dict[str, Any]]
    tool_trace: Annotated[list[dict[str, Any]], _merge_traces]
    steps: int
    max_steps: int
    answer: str
    done: bool
