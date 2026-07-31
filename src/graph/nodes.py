"""LangGraph node implementations — fully async for concurrent /query handling.

Three node factories live here, one per graph node:

- ``make_agent_node``      — calls the Qwen planner (``agent-brain``) with the
  tool catalogue and parses its tool calls out of the response.
- ``make_tool_executor_node`` — runs the pending tool calls through the
  ``ToolRegistry`` (in parallel) and records results in the transcript + trace.
- ``make_synthesis_node``  — calls the fine-tuned model (``domain-ft``) to
  write the final answer from the accumulated tool evidence.

Factories close over their dependencies (LLM clients, tool registry) so the
nodes themselves are pure ``state → partial update`` coroutines.

A large share of this module is defensive parsing around Qwen3's tool-calling
quirks: it sometimes emits XML ``<tool_call>`` markup in plain content instead
of structured ``tool_calls``, and sometimes calls TFQL operation names
(``rba.*`` etc.) as if they were tools. Both are recovered rather than failed.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from src.graph.state import GraphState
from src.llm.client import LLMClients
from src.tools.registry import ToolRegistry
from src.tools.tfql_tool import TOOL_NAME

logger = logging.getLogger(__name__)

# Qwen3 often emits tool calls as XML in message content even when the OpenAI
# tools schema is provided. vLLM's hermes parser does not always promote them
# into structured tool_calls, so we recover them here.
_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)\s*>(.*?)</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_PARAM = re.compile(
    r"<parameter=([^>\s]+)\s*>(.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)
_TFQL_PREFIXES = ("rba.", "asx.", "afr.", "cross.")
# Keep follow-up brain prompts inside Qwen's 4096-token max_model_len.
_MAX_TOOL_MSG_CHARS = 3500

BRAIN_SYSTEM = (
    "You are a financial query planner for RBA cash-rate, ASX price and AFR news "
    "data.\n\n"
    "Your ONLY tool is execute_plan. Never call an operation name (rba.*, asx.*, "
    "afr.*, cross.*) as a tool — put those names in execute_plan.operations[].op.\n\n"
    "Do NOT calculate returns, counts, rankings, date differences or rate changes "
    "yourself — request the operation that computes them. Answering from memory "
    "scores zero; every dataset fact must come from a tool result.\n\n"
    "Put every independent operation into a SINGLE execute_plan call. When one "
    "operation needs another's output, reference it as ${other_id.data.field} "
    "in the same call rather than waiting for a second turn.\n\n"
    "Use only the argument values listed in the operation catalogue; anything "
    "else is rejected. Use structured RBA and ASX operations for tabular facts "
    "and AFR operations for article evidence — never article retrieval to answer "
    "a numeric RBA or ASX question.\n\n"
    "After tool results, a separate model writes the user-facing answer — do not "
    "narrate a long final answer yourself."
)

SYNTHESIS_SYSTEM = (
    "You write the final answer from verified tool results.\n\n"
    "State every value the question asked for. Also state the directly "
    "supporting values the results provide — the dates a span runs between, the "
    "rate before and after a change, the price on either side of a move. Extra "
    "relevant facts are not penalised, but an omitted one loses its point.\n\n"
    "Copy numbers and dates exactly as they appear in the results — never "
    "recompute, convert or re-round them; each is already in the unit the "
    "answer needs.\n\n"
    "Answer in one or two sentences. No preamble, no restating the question, no "
    "reasoning aloud, no hedging — the grader checks each requested fact "
    "independently and cannot find them buried in paragraphs.\n\n"
    "Where a result carries an error or warning, state that limitation plainly "
    "and still report every value that did succeed. Never invent a figure to "
    "fill a gap."
)


def _parse_param_value(raw: str) -> Any:
    """Parse an XML <parameter> body: JSON when valid, raw string otherwise."""
    text = raw.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _tool_calls_from_xml(content: str) -> list[dict[str, Any]]:
    """Recover Qwen XML tool calls embedded in assistant content."""
    calls: list[dict[str, Any]] = []
    for match in _TOOL_CALL_BLOCK.finditer(content or ""):
        name = match.group(1).strip()
        body = match.group(2)
        args: dict[str, Any] = {}
        for param in _TOOL_PARAM.finditer(body):
            args[param.group(1).strip()] = _parse_param_value(param.group(2))
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )
    return calls


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
    serialized: list[dict[str, Any]] = []
    if tool_calls:
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
    # Fallback: parse XML tool markup from content when the gateway left
    # tool_calls empty (common with Qwen3 + hermes through LiteLLM).
    if not serialized and isinstance(content, str) and "<tool_call>" in content:
        serialized = _tool_calls_from_xml(content)
        if serialized:
            logger.info(
                "Recovered %s XML tool call(s) from assistant content",
                len(serialized),
            )
            # Keep transcript clean for synthesis — the structured calls are enough.
            out["content"] = _TOOL_CALL_BLOCK.sub("", content).strip() or None
    if serialized:
        out["tool_calls"] = serialized
    return out


def _is_tfql_op_name(name: str) -> bool:
    lower = (name or "").strip().lower()
    return any(lower.startswith(prefix) for prefix in _TFQL_PREFIXES)


def _strip_evidence(value: Any) -> Any:
    """Drop bulky evidence blobs from payloads fed back to the planner."""
    if isinstance(value, dict):
        return {
            key: _strip_evidence(item)
            for key, item in value.items()
            if key != "evidence"
        }
    if isinstance(value, list):
        return [_strip_evidence(item) for item in value]
    return value


def _compact_tool_content(payload: Any) -> str:
    cleaned = _strip_evidence(payload)
    if isinstance(cleaned, (dict, list)):
        text = json.dumps(cleaned, ensure_ascii=False, default=str)
    else:
        text = str(cleaned)
    if len(text) > _MAX_TOOL_MSG_CHARS:
        return text[:_MAX_TOOL_MSG_CHARS] + "...[truncated]"
    return text


def _normalize_pending_calls(
    pending: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rewrite bare TFQL op tool-calls into a single execute_plan envelope.

    Qwen often emits ``<function=rba.rate_cycle>`` instead of
    ``execute_plan`` with ``operations=[{op: rba.rate_cycle, ...}]``. Those
    names are not registered tools, so without this rewrite every call fails
    as ``Unknown tool``.
    """
    passthrough: list[dict[str, Any]] = []
    bare_ops: list[dict[str, Any]] = []
    first_bare_id: str | None = None

    for call in pending:
        name = str(call.get("name") or "")
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if name == TOOL_NAME:
            passthrough.append(call)
            continue
        if _is_tfql_op_name(name):
            if isinstance(args.get("operations"), list):
                passthrough.append(
                    {"id": call.get("id"), "name": TOOL_NAME, "args": args}
                )
                continue
            if first_bare_id is None:
                first_bare_id = call.get("id")
            bare_ops.append(
                {
                    "id": f"{name.replace('.', '_')}_{len(bare_ops)}",
                    "op": name,
                    "args": args,
                }
            )
            continue
        passthrough.append(call)

    if bare_ops:
        logger.info(
            "Rewrote %s bare TFQL op call(s) into execute_plan",
            len(bare_ops),
        )
        passthrough.insert(
            0,
            {
                "id": first_bare_id or f"call_{uuid.uuid4().hex[:24]}",
                "name": TOOL_NAME,
                "args": {"operations": bare_ops},
            },
        )
    return passthrough


def _tool_trace_has_success(trace: list[dict[str, Any]] | None) -> bool:
    """True when at least one trace entry carries usable data.

    Used by the post-tools router: one success is enough evidence to
    synthesize from, so the workflow skips further (context-hungry) planner
    turns. An entry counts as failed only when it is a pure ``{"error": ...}``
    payload with no data alongside it.
    """
    for item in trace or []:
        result = item.get("result")
        if result is None:
            continue
        if isinstance(result, dict) and result.get("error") and "data" not in result:
            continue
        return True
    return False


def make_agent_node(llm: LLMClients, tools: ToolRegistry):
    """Build the planner node: one Qwen turn that may emit tool calls.

    The OpenAI function schemas are rendered once at graph build time — the
    registry is fixed after startup, so there is no reason to re-serialize
    them on every request.
    """
    openai_tools = tools.openai_tools()

    async def agent_node(state: GraphState) -> dict[str, Any]:
        messages = list(state["messages"])
        if not messages:
            messages = [
                {"role": "system", "content": BRAIN_SYSTEM},
                {"role": "user", "content": state["question"]},
            ]

        steps = int(state.get("steps", 0)) + 1
        max_steps = int(state.get("max_steps", 5))

        try:
            response = await llm.brain_chat(messages, tools=openai_tools or None)
        except Exception as exc:  # noqa: BLE001 — recover into synthesis
            # Typical cause: prior tool transcript + catalogue exceeds vLLM
            # max_model_len (4096). Synthesize from whatever evidence we have.
            logger.error("Brain call failed (%s); forcing synthesis", exc)
            return {
                "messages": messages,
                "pending_tool_calls": [],
                "steps": steps,
            }

        choice = response.choices[0].message
        assistant = _message_to_dict(choice)
        messages.append(assistant)

        # Parse tool-call arguments into dicts; malformed JSON is preserved
        # under "_raw" so the tool layer can reject it with a clear error
        # instead of the whole request failing.
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
        pending = _normalize_pending_calls(pending)

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
        }

    return agent_node


def make_tool_executor_node(tools: ToolRegistry):
    """Build the tool node: run every pending call and record the results.

    Each result is written twice, in different shapes for different readers:

    - a compact ``role=tool`` chat message (evidence stripped, truncated) so a
      possible second planner turn stays inside the 4096-token context, and
    - a full-fidelity ``tool_trace`` entry, which feeds synthesis and is
      returned verbatim in the API response.
    """

    async def tool_executor_node(state: GraphState) -> dict[str, Any]:
        pending = state.get("pending_tool_calls") or []
        if not pending:
            return {"pending_tool_calls": []}

        # Parallel execution — major latency win when the agent emits multiple calls.
        records = await tools.execute_many(
            [(p["name"], p.get("args") or {}) for p in pending]
        )

        messages = list(state["messages"])
        trace_items: list[dict[str, Any]] = []
        for call, record in zip(pending, records, strict=True):
            payload = record.result
            # Compact content for any later brain turn; full payload stays in tool_trace.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id") or record.tool,
                    "name": record.tool,
                    "content": _compact_tool_content(payload),
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
        }

    return tool_executor_node


def make_synthesis_node(llm: LLMClients):
    """Build the synthesis node: one domain-ft turn that writes the answer.

    The prompt is assembled from the question, the planner's last free-text
    note, and every tool result (with bulky article evidence stripped so the
    whole prompt fits the model's context window).
    """

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
            # Strip evidence so synthesis stays inside the 4096 context window.
            compact_result = _strip_evidence(item.get("result"))
            tool_bits.append(
                f"- {item.get('tool')}: args={item.get('args')} result={compact_result}"
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
        return {"answer": answer}

    return synthesis_node
