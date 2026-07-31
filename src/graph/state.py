"""Shared LangGraph state for the query workflow.

Every node receives the current ``GraphState`` and returns a partial update.
LangGraph merges updates back into the state: plain keys are overwritten by
the latest node, while keys annotated with a reducer (here ``tool_trace``)
are combined, so trace items accumulate across agent/tool loop iterations
instead of being replaced.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class GraphState(TypedDict):
    """State threaded through the agent → tools → synthesize graph.

    Attributes:
        question: The raw user question from ``POST /query``.
        messages: Full OpenAI-style chat transcript (system / user /
            assistant / tool messages) sent to the planning model each turn.
        pending_tool_calls: Tool calls the agent emitted this turn, normalised
            to ``{"id", "name", "args"}`` dicts. Consumed (and cleared) by the
            tool-executor node; routing checks this to decide agent → tools.
        tool_trace: Accumulated ``{"tool", "args", "result"}`` records for the
            API response. ``operator.add`` makes LangGraph append rather than
            overwrite when multiple loop iterations each contribute items.
        steps: Number of agent (planner) turns taken so far.
        max_steps: Hard budget for agent turns (``MAX_AGENT_STEPS``).
        answer: Final user-facing answer, written by the synthesis node.
    """

    question: str
    messages: list[dict[str, Any]]
    pending_tool_calls: list[dict[str, Any]]
    tool_trace: Annotated[list[dict[str, Any]], operator.add]
    steps: int
    max_steps: int
    answer: str
