"""Hackathon finance agent used as the eval target.

Implements the two-model architecture required by the challenge brief
(README.md at the repo's ``AI_Industry_Training_Hackathon`` root):

1. **Qwen** ("agent-brain") plans the answer and emits tool calls. It's
   bound to every tool in ``finance_tools.py`` and runs its own
   plan/call-tools/observe loop via ``create_agent`` -- exactly the same
   machinery ``agent.py`` and ``simple-agent.py`` use at the project root.
2. The agent runtime (``finance_tools.py``) executes those calls against
   the real RBA/ASX/AFR datasets and returns structured results to Qwen.
3. Once Qwen's tool loop settles (no further tool calls), the **fine-tuned
   Nemotron** model receives the original question plus the verified tool
   trace -- not Qwen's own prose -- and synthesizes the final ``answer``.
   This is what the brief calls "Use data tools for dataset-derived facts
   rather than relying on model memory": Nemotron never sees the raw
   datasets, only the tool-verified evidence.

The two stages are wired as a 2-node LangGraph graph (``planner`` ->
``synthesize``) so LangGraph Studio can render and step through both
stages, and LangSmith traces every model call and tool call inside each
(same tracing you already get from ``agent.py`` / ``simple-agent.py`` --
no extra wiring needed beyond the ``LANGSMITH_*`` vars already in ``.env``).

Environment variables
----------------------
``DOMAIN_PREDICT_MODE`` -- ``"mock"`` (default, matches the cluster
bootstrap default) skips the Nemotron call entirely and instead returns
Qwen's own tool-grounded reply, so the pipeline runs end-to-end before a
fine-tuned adapter is served. Set to ``"llm"`` once the fine-tuned model is
live behind ``FINETUNED_*``.

``AGENT_BRAIN_MODEL_NAME`` / ``AGENT_BRAIN_BASE_URL`` / ``AGENT_BRAIN_API_KEY``
-- Qwen planner, falling back to this repo's existing ``MODEL_NAME`` /
``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` so the graph still runs against
whatever gateway ``.env`` already points at.

``FINETUNED_MODEL_NAME`` / ``FINETUNED_BASE_URL`` / ``FINETUNED_API_KEY``
-- fine-tuned Nemotron, served by vLLM per the brief. No fallback to the
planner's gateway -- if these aren't set and ``DOMAIN_PREDICT_MODE=llm``,
the graph raises rather than silently reusing Qwen as if it were Nemotron.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

# Make sibling imports (finance_tools, finance_data) resolve regardless of
# how this file is loaded -- LangGraph Studio/CLI loads graphs by file path
# (like the project root's "./simple-agent.py:graph"), which doesn't always
# put this file's own directory on sys.path.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from finance_tools import ALL_TOOLS  # noqa: E402

# The shared hackathon gateway fronts Azure OpenAI, which 404s without an
# ``api-version`` query param on every request.
API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-08-01-preview")


def _verify_ssl(env_var: str) -> bool:
    value = os.getenv(env_var, os.getenv("OPENAI_VERIFY_SSL", "true"))
    return value.strip().lower() not in {"false", "0", "no"}


# ---------------------------------------------------------------------------
# Stage 1: Qwen "agent-brain" -- plans and emits tool calls.
# ---------------------------------------------------------------------------
planner_model = ChatOpenAI(
    openai_api_key=os.getenv(
        "AGENT_BRAIN_API_KEY", os.getenv("OPENAI_API_KEY", "sk-123456")
    ),
    openai_api_base=os.getenv(
        "AGENT_BRAIN_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://ai-gateway:4000")
    ),
    model_name=os.getenv(
        "AGENT_BRAIN_MODEL_NAME", os.getenv("MODEL_NAME", "llama-distributed")
    ),
    temperature=0.01,
    http_async_client=httpx.AsyncClient(verify=_verify_ssl("AGENT_BRAIN_VERIFY_SSL")),
    default_query={"api-version": API_VERSION} if API_VERSION else None,
)

PLANNER_SYSTEM_PROMPT = """\
You are the planning stage of a finance research agent being evaluated on a
hackathon dataset. You answer questions about three real data sources:
- RBA cash-rate decision history (levels, effective dates, streaks, cycles)
- ASX daily stock prices for an 18-company basket (returns, volume,
  rankings, drawdowns, single-day moves)
- AFR news articles (counts, dates, headlines, keyword mentions)

Always call the matching tool(s) before answering -- never rely on your own
memory of rates, prices, or article counts, since only the tools see the
real dataset. State exact figures, dates, and counts the tools returned;
do not round, approximate, or invent numbers. If a question needs more than
one fact (e.g. a count and a date), call every tool needed before replying.
If no tool is relevant, say so briefly instead of guessing.
"""

planner_graph = create_agent(
    planner_model,
    ALL_TOOLS,
    system_prompt=PLANNER_SYSTEM_PROMPT,
    name="finance-planner-qwen",
)


# ---------------------------------------------------------------------------
# Stage 2: fine-tuned Nemotron -- synthesizes the final answer from Qwen's
# verified tool trace (not from Qwen's own prose, and never from the raw
# datasets directly).
# ---------------------------------------------------------------------------
DOMAIN_PREDICT_MODE = os.getenv("DOMAIN_PREDICT_MODE", "mock").strip().lower()

_finetuned_model: ChatOpenAI | None = None
if DOMAIN_PREDICT_MODE == "llm":
    base_url = os.getenv("FINETUNED_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "DOMAIN_PREDICT_MODE=llm requires FINETUNED_BASE_URL (and "
            "FINETUNED_MODEL_NAME) to point at the served fine-tuned "
            "Nemotron endpoint. Set DOMAIN_PREDICT_MODE=mock to run without "
            "it, or set FINETUNED_* in .env once the adapter is serving."
        )
    _finetuned_model = ChatOpenAI(
        openai_api_key=os.getenv("FINETUNED_API_KEY", "sk-123456"),
        openai_api_base=base_url,
        model_name=os.getenv("FINETUNED_MODEL_NAME", "nemotron-finetuned"),
        temperature=0.01,
        http_async_client=httpx.AsyncClient(verify=_verify_ssl("FINETUNED_VERIFY_SSL")),
        default_query={"api-version": API_VERSION} if API_VERSION else None,
    )

SYNTHESIS_SYSTEM_PROMPT = """\
You are the final-answer stage of a finance research agent. You are given
the user's question and a verified trace of tool calls and their results
from the earlier planning stage. Write a single, concise answer using only
the facts in that trace -- do not add, round, or infer numbers beyond what
the tools returned, and do not mention tools, planning, or internal steps
in your answer.
"""


def _tool_trace(messages: list[AnyMessage]) -> str:
    """Render every tool call + result in ``messages`` as plain text
    evidence for the synthesis stage. Deliberately ignores any prose the
    planner itself wrote -- only verified tool calls/results count."""
    calls_by_id: dict[str, str] = {}
    lines: list[str] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls_by_id[call["id"]] = f"{call['name']}({call.get('args', {})})"
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id is not None:
            call_desc = calls_by_id.get(tool_call_id, "tool")
            lines.append(f"- {call_desc} -> {message.content}")
    return "\n".join(lines) if lines else "(no tool calls were made)"


def extract_tool_trace(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """Structured ``[{"tool", "args", "result"}, ...]`` view of every tool
    call in ``messages`` -- the same shape the submission contract's
    ``answer_template.json`` uses for its ``tool_trace`` field. Shared by
    ``api.py`` (the HTTP wrapper) and ``llm_judge_grader.py`` (in-process
    grading) so both report identical trace data for the same run."""
    calls_by_id: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls_by_id[call["id"]] = {"tool": call["name"], "args": call.get("args", {})}
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id is not None and tool_call_id in calls_by_id:
            entry = dict(calls_by_id[tool_call_id])
            entry["result"] = str(message.content)
            trace.append(entry)
    return trace


def final_answer_text(messages: list[AnyMessage]) -> str:
    """The last non-empty AI-authored message -- the same extraction logic
    used by every entry point (api.py, llm_judge_grader.py, run_eval_offline.py)."""
    return next(
        (m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
        "",
    )


class FinanceState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


async def synthesize(state: FinanceState) -> dict[str, Any]:
    messages = state["messages"]
    question = next(
        (m.content for m in messages if isinstance(m, HumanMessage)),
        "",
    )
    trace = _tool_trace(messages)

    if DOMAIN_PREDICT_MODE != "llm" or _finetuned_model is None:
        # Bootstrap/mock mode: no fine-tuned model wired up yet. Fall back to
        # Qwen's own final reply so the pipeline still produces an answer.
        planner_reply = next(
            (
                m.content
                for m in reversed(messages)
                if isinstance(m, AIMessage) and m.content
            ),
            "",
        )
        return {"messages": [AIMessage(content=planner_reply)]}

    response = await _finetuned_model.ainvoke([
        ("system", SYNTHESIS_SYSTEM_PROMPT),
        (
            "human",
            f"QUESTION:\n{question}\n\nVERIFIED TOOL TRACE:\n{trace}",
        ),
    ])
    return {"messages": [AIMessage(content=response.text)]}


builder = StateGraph(FinanceState)
builder.add_node("planner", planner_graph)
builder.add_node("synthesize", synthesize)
builder.set_entry_point("planner")
builder.add_edge("planner", "synthesize")
builder.set_finish_point("synthesize")

graph = builder.compile(name="finance-hackathon-agent")
