"""LangGraph node implementations — fully async for concurrent /query handling."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.graph.state import GraphState
from src.llm.client import LLMClients
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

BRAIN_SYSTEM = (
    "You are a routing and research agent. Use tools when they help answer the "
    "question. Prefer the fewest tool calls needed. When you have enough "
    "information, respond with a concise factual summary (no fluff) — do NOT "
    "produce the final user-facing answer; a separate synthesis model will do that."
)

SYNTHESIS_SYSTEM = (
    "You produce the final answer for the user. Be accurate, direct, and grounded "
    "in the provided agent notes and tool results. If evidence is insufficient, "
    "say what is missing rather than inventing facts."
)


def _message_to_dict(message: Any) -> dict[str, Any]:
    """Normalize OpenAI SDK message / tool_call objects into plain dicts."""
    role = getattr(message, "role", None) or message.get("role")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    out: dict[str, Any] = {"role": role, "content": content}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls is None and isinstance(message, dict):
        tool_calls = message.get("tool_calls")
    if tool_calls:
        serialized = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                serialized.append(tc)
                continue
            fn = tc.function
            serialized.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": fn.name,
                        "arguments": fn.arguments,
                    },
                }
            )
        out["tool_calls"] = serialized
    return out


def make_agent_node(llm: LLMClients, tools: ToolRegistry):
    openai_tools = tools.openai_tools()

    async def agent_node(state: GraphState) -> dict[str, Any]:
        messages = list(state["messages"])
        if not messages:
            messages = [
                {"role": "system", "content": BRAIN_SYSTEM},
                {"role": "user", "content": state["question"]},
            ]

        response = await llm.brain_chat(messages, tools=openai_tools or None)
        choice = response.choices[0].message
        assistant = _message_to_dict(choice)
        messages.append(assistant)

        pending: list[dict[str, Any]] = []
        raw_calls = assistant.get("tool_calls") or []
        for tc in raw_calls:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                parsed = {"_raw": raw_args}
            if not isinstance(parsed, dict):
                parsed = {"value": parsed}
            pending.append(
                {
                    "id": tc.get("id"),
                    "name": fn.get("name") or "unknown",
                    "args": parsed,
                }
            )

        steps = int(state.get("steps", 0)) + 1
        max_steps = int(state.get("max_steps", 5))
        # Force completion if we hit the step budget.
        if pending and steps >= max_steps:
            logger.warning("MAX_AGENT_STEPS=%s reached; skipping further tool calls", max_steps)
            pending = []
            if not assistant.get("content"):
                messages.append(
                    {
                        "role": "assistant",
                        "content": "Step budget exhausted; synthesize from collected evidence.",
                    }
                )

        return {
            "messages": messages,
            "pending_tool_calls": pending,
            "steps": steps,
            "done": not pending,
        }

    return agent_node


def make_tool_executor_node(tools: ToolRegistry):
    async def tool_executor_node(state: GraphState) -> dict[str, Any]:
        pending = state.get("pending_tool_calls") or []
        if not pending:
            return {"pending_tool_calls": [], "done": True}

        # Parallel execution — major latency win when the agent emits multiple calls.
        records = await tools.execute_many(
            [(p["name"], p.get("args") or {}) for p in pending]
        )

        messages = list(state["messages"])
        trace_items: list[dict[str, Any]] = []
        for call, record in zip(pending, records, strict=True):
            payload = record.result
            # Keep tool message content compact for faster subsequent LLM turns.
            if isinstance(payload, (dict, list)):
                content = json.dumps(payload, ensure_ascii=False, default=str)
            else:
                content = str(payload)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id") or record.tool,
                    "name": record.tool,
                    "content": content,
                }
            )
            trace_items.append(
                {
                    "tool": record.tool,
                    "args": record.args,
                    "result": record.result,
                }
            )

        return {
            "messages": messages,
            "pending_tool_calls": [],
            "tool_trace": trace_items,
            "done": False,
        }

    return tool_executor_node


def make_synthesis_node(llm: LLMClients):
    async def synthesis_node(state: GraphState) -> dict[str, Any]:
        question = state["question"]
        # Prefer the last assistant content as agent notes; fall back to full transcript.
        notes = ""
        for msg in reversed(state.get("messages") or []):
            if msg.get("role") == "assistant" and msg.get("content"):
                notes = str(msg["content"])
                break

        tool_bits: list[str] = []
        for item in state.get("tool_trace") or []:
            tool_bits.append(
                f"- {item.get('tool')}: args={item.get('args')} result={item.get('result')}"
            )
        tools_block = "\n".join(tool_bits) if tool_bits else "(no tools used)"

        prompt = (
            f"Question:\n{question}\n\n"
            f"Agent notes:\n{notes or '(none)'}\n\n"
            f"Tool results:\n{tools_block}\n\n"
            "Write the final answer only."
        )
        answer = await llm.synthesize(prompt, system=SYNTHESIS_SYSTEM)
        if not answer:
            answer = notes or "Unable to generate an answer."
        return {"answer": answer, "done": True}

    return synthesis_node
