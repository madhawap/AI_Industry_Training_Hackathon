"""LangGraph workflow wiring: agent ⇄ tools → synthesize.

Graph shape (see ARCHITECTURE.md for the full diagram):

    START → agent ─┬─(tool calls pending)→ tools ─┬─(success / budget)→ synthesize → END
                   │                              └─(all calls failed)→ agent
                   └─(no tool calls)→ synthesize → END

Design decisions that matter for latency and correctness:

- **Fully async nodes + ``ainvoke``** — safe concurrent ``/query`` handling on
  one event loop; no threads or worker processes needed.
- **No checkpointer** — every query is single-shot, so persisting intermediate
  state would only add serialization overhead.
- **Hard step budget** — the agent ⇄ tools loop is bounded by
  ``MAX_AGENT_STEPS`` and a matching LangGraph ``recursion_limit``.
- **Synthesize-first routing after tools** — one successful tool result is
  enough to answer; a second planner turn re-sends the ~2k-token tool
  catalogue and frequently overflows the 4096-token model context.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from src.config import Settings
from src.graph.nodes import (
    _tool_trace_has_success,
    make_agent_node,
    make_synthesis_node,
    make_tool_executor_node,
)
from src.graph.state import GraphState
from src.llm.client import LLMClients
from src.tools.registry import ToolRegistry


def _route_after_agent(state: GraphState) -> Literal["tools", "synthesize"]:
    """agent → tools when it emitted tool calls, otherwise straight to synthesis."""
    if state.get("pending_tool_calls"):
        return "tools"
    return "synthesize"


def _route_after_tools(state: GraphState) -> Literal["agent", "synthesize"]:
    """Prefer synthesize after a successful plan to avoid context overflow.

    A second brain turn re-sends the full tool catalogue (~2k tokens) plus the
    growing transcript; with vLLM max_model_len=4096 that frequently 400s and
    surfaces as HTTP 502. Retry the planner only when every tool call failed
    and there is still step budget left.
    """
    if _tool_trace_has_success(state.get("tool_trace")):
        return "synthesize"
    if int(state.get("steps", 0)) >= int(state.get("max_steps", 5)):
        return "synthesize"
    return "agent"


def build_workflow(llm: LLMClients, tools: ToolRegistry):
    """Assemble and compile the LangGraph app.

    Nodes are created via factories so each one closes over its dependencies
    (LLM clients, tool registry) instead of reaching for globals.
    """
    graph = StateGraph(GraphState)

    graph.add_node("agent", make_agent_node(llm, tools))
    graph.add_node("tools", make_tool_executor_node(tools))
    graph.add_node("synthesize", make_synthesis_node(llm))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "synthesize": "synthesize"},
    )
    graph.add_conditional_edges(
        "tools",
        _route_after_tools,
        {"agent": "agent", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)

    # No checkpointer: each /query is independent → lower latency under concurrency.
    return graph.compile()


class QueryWorkflow:
    """Thin async facade around the compiled LangGraph app.

    Built once at startup and shared across requests; ``run`` is re-entrant
    because all per-question state lives in the ``GraphState`` it seeds.
    """

    def __init__(self, settings: Settings, llm: LLMClients, tools: ToolRegistry) -> None:
        self.settings = settings
        self.app = build_workflow(llm, tools)

    async def run(self, question: str) -> dict[str, Any]:
        """Answer one question; returns ``{answer, steps, tool_trace}``."""
        initial: GraphState = {
            "question": question,
            "messages": [],
            "pending_tool_calls": [],
            "tool_trace": [],
            "steps": 0,
            "max_steps": self.settings.max_agent_steps,
            "answer": "",
        }
        # Recursion limit: each agent turn costs 2 node visits (agent + tools);
        # +2 covers synthesize plus buffer.
        result = await self.app.ainvoke(
            initial,
            config={"recursion_limit": max(4, self.settings.max_agent_steps * 2 + 2)},
        )
        return {
            "answer": result.get("answer") or "",
            "steps": int(result.get("steps") or 0),
            "tool_trace": list(result.get("tool_trace") or []),
        }
