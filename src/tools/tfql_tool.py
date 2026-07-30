"""TFQL exposed to the Qwen agent as a single batched tool.

Qwen sees one tool, ``execute_plan``, and calls it once per question with up to
six operations in a single envelope. That is deliberate: the expensive thing in
the pipeline is the round trip to the planning model, not the data work, so
every operation packed into one call is a planner pass not paid for. A
conventional one-tool-per-metric registry would need three or four turns for the
cross-dataset questions and would not fit the 60-second budget.

The description Qwen reads is generated from the TFQL registry, so the prompt
can never advertise an operation that does not exist, or omit one that does.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.tfql import PlanRequest, Store
from src.tfql import registry as tfql_registry
from src.tfql.executor import execute as execute_plan
from src.tools.base import ToolResult, ToolSpec
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

TOOL_NAME = "execute_plan"

_PREAMBLE = """\
Run one or more deterministic financial data operations over the approved RBA
cash-rate, ASX price and AFR news datasets, and return the computed results.

Do not calculate returns, counts, rankings, date differences or rate changes
yourself -- request the operation that computes them. Put every independent
operation into a single execute_plan call rather than making several calls.

Each operation needs a unique `id`, an `op` name from the catalogue below, and
`args`. An operation may use an earlier result by referencing it as
${other_id.data.field_name}; dependencies are resolved automatically.

Operations that fail return a structured error while the others still return
their results, so a partial answer is always better than no call at all.
"""


def _render_catalogue(store: Store) -> str:
    """Build the tool description Qwen reads when choosing operations.

    Generated from the live registry and the loaded store, so the ticker list
    and coverage dates are always accurate. Injecting them statically here
    saves a discovery round trip that the latency budget cannot afford.
    """
    lines = [_PREAMBLE, "", "AVAILABLE DATA"]
    lines.append(f"  ASX tickers: {', '.join(store.tickers)}")
    lines.append(f"  RBA decisions: {store.rba.coverage.describe()}")
    lines.append(f"  ASX prices:    {store.asx_coverage().describe()}")
    lines.append(f"  AFR articles:  {store.afr_coverage.describe()}")
    lines.append(
        "  Cross-dataset questions are limited to the span all three share; "
        "say so when a question reaches outside it."
    )
    lines.append("")
    lines.append("OPERATION CATALOGUE")
    for entry in tfql_registry.catalogue():
        required = entry["required"]
        rendered = [
            _render_arg(name, schema, name in required)
            for name, schema in entry["args"].items()
        ]
        lines.append(f"  {entry['op']}({', '.join(rendered)})")
        lines.append(f"      {entry['summary']}")
    lines.append("")
    lines.append(
        "Arguments marked * are required; the rest show their default. Use only "
        "the listed values for enumerated arguments -- any other value is "
        "rejected. Dates are ISO YYYY-MM-DD."
    )
    return "\n".join(lines)


def _render_arg(name: str, schema: dict[str, Any], required: bool) -> str:
    """Render one argument as ``name: type = default``.

    The tool's JSON schema types ``args`` as a free-form object, so this text is
    the only place the planner learns an argument's type or its permitted enum
    values. Omitting them makes the model guess.
    """
    label = f"{name}*" if required else name
    kind = _render_type(schema)
    if required:
        return f"{label}: {kind}"
    default = schema.get("default")
    if default is None:
        return f"{label}: {kind}"
    return f"{label}: {kind} = {default}"


def _render_type(schema: dict[str, Any]) -> str:
    """A compact type description: enum members, item type, or JSON type."""
    if "enum" in schema:
        return "|".join(str(v) for v in schema["enum"])
    # Optional fields arrive as anyOf[type, null]; describe the non-null branch.
    if "anyOf" in schema:
        branches = [b for b in schema["anyOf"] if b.get("type") != "null"]
        if branches:
            return _render_type(branches[0])
    json_type = schema.get("type")
    if json_type == "array":
        return f"list[{_render_type(schema.get('items', {}))}]"
    return {
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "string": "str",
    }.get(str(json_type), "str")


def make_executor(store: Store):
    """Build the async execute callable, closing over the preloaded store."""

    async def _run(payload: PlanRequest) -> ToolResult:
        # TFQL is synchronous and CPU-bound. Offloading keeps the event loop
        # free so three concurrent /query requests do not serialise behind
        # each other's data work.
        bundle = await asyncio.to_thread(execute_plan, payload, store)

        if bundle.status == "error":
            # Every operation failed. Surface it as a tool-level error so the
            # agent can repair the plan, but keep the structured detail.
            failures = "; ".join(
                f"{r.id}: {(r.error or {}).get('message', 'failed')}" for r in bundle.failures()
            )
            logger.warning("execute_plan produced no results: %s", failures)
            return ToolResult(
                success=False,
                data=bundle.model_dump(mode="json", exclude_none=True),
                error=failures or "no operation produced a result",
            )

        payload_out: dict[str, Any] = bundle.model_dump(mode="json", exclude_none=True)
        if bundle.status == "partial":
            # A warning rather than a failure -- the successful components are
            # still worth points under component-based grading.
            failures = "; ".join(
                f"{r.id}: {(r.error or {}).get('message', 'failed')}" for r in bundle.failures()
            )
            return ToolResult(success=True, data=payload_out, error=failures)

        return ToolResult(success=True, data=payload_out)

    return _run


def register_tfql(
    registry: ToolRegistry,
    store: Store,
    timeout: float = 15.0,
) -> None:
    """Register the batched TFQL planner tool against a preloaded store."""
    registry.register(
        ToolSpec(
            name=TOOL_NAME,
            description=_render_catalogue(store),
            args_schema=PlanRequest,
            execute=make_executor(store),
            timeout=timeout,
        )
    )
