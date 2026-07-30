"""
Optimized LangGraph workflow:

  POST /query → Qwen Agent ⇄ Tool Executor → Fine-tuned Synthesis → response

Optimizations:
- Fully async nodes + ainvoke (safe concurrent /query handling)
- Parallel tool execution via asyncio.gather
- No checkpointer (single-shot queries; avoids serialization overhead)
- Hard MAX_AGENT_STEPS bound
- Shared LiteLLM AsyncOpenAI connection pool
- Compact tool messages to shrink subsequent brain prompts
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from src.config import Settings
from src.graph.nodes import make_agent_node, make_synthesis_node, make_tool_executor_node
from src.graph.state import GraphState
from src.llm.client import LLMClients
from src.tools.registry import ToolRegistry


def _route_after_agent(state: GraphState) -> Literal["tools", "synthesize"]:
    if state.get("pending_tool_calls"):
        return "tools"
    return "synthesize"


def build_workflow(settings: Settings, llm: LLMClients, tools: ToolRegistry):
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
    graph.add_edge("tools", "agent")
    graph.add_edge("synthesize", END)

    # No checkpointer: each /query is independent → lower latency under concurrency.
    return graph.compile()


class QueryWorkflow:
    """Thin async facade around the compiled LangGraph app."""

    def __init__(self, settings: Settings, llm: LLMClients, tools: ToolRegistry) -> None:
        self.settings = settings
        self.app = build_workflow(settings, llm, tools)

    async def run(self, question: str) -> dict[str, Any]:
        initial: GraphState = {
            "question": question,
            "messages": [],
            "pending_tool_calls": [],
            "tool_trace": [],
            "steps": 0,
            "max_steps": self.settings.max_agent_steps,
            "answer": "",
            "done": False,
        }
        # Recursion limit: agent↔tools can loop; +2 covers synthesize + buffer.
        result = await self.app.ainvoke(
            initial,
            config={"recursion_limit": max(4, self.settings.max_agent_steps * 2 + 2)},
        )
        return {
            "answer": result.get("answer") or "",
            "steps": int(result.get("steps") or 0),
            "tool_trace": list(result.get("tool_trace") or []),
        }
