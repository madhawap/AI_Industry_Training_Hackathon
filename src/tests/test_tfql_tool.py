"""The adapter that exposes TFQL to the Qwen agent.

Checks the contract in both directions: that the tool registry can hold it and
generate a function schema Qwen can call, and that plan results come back in a
shape the synthesiser can read -- including when operations fail.
"""

from __future__ import annotations

import pytest
from src.tfql import registry as tfql_registry
from src.tools.base import ToolResult
from src.tools.registry import ToolRegistry
from src.tools.tfql_tool import TOOL_NAME, _render_catalogue, register_tfql


@pytest.fixture
def tools(store) -> ToolRegistry:
    registry = ToolRegistry()
    register_tfql(registry, store)
    return registry


class TestRegistration:
    def test_registers_exactly_one_tool(self, tools):
        assert [t.name for t in tools.list_tools()] == [TOOL_NAME]

    def test_exposes_a_function_schema_for_the_planner(self, tools):
        schema = tools.openai_tools()[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == TOOL_NAME
        assert "operations" in schema["function"]["parameters"]["properties"]


class TestCatalogue:
    def test_lists_every_registered_operation(self, store):
        text = _render_catalogue(store)
        for name in tfql_registry.names():
            assert name in text, f"{name} missing from the tool description"

    def test_injects_tickers_and_coverage_to_avoid_a_discovery_call(self, store):
        text = _render_catalogue(store)
        for ticker in store.tickers:
            assert ticker in text
        assert store.rba.coverage.describe() in text
        assert store.afr_coverage.describe() in text

    def test_tells_the_planner_not_to_do_arithmetic(self, store):
        text = _render_catalogue(store).lower()
        assert "do not calculate" in text

    def test_marks_required_arguments(self, store):
        # asx.return needs ticker, start and end.
        assert "ticker*" in _render_catalogue(store)


class TestExecution:
    @pytest.mark.asyncio
    async def test_runs_a_batched_plan(self, tools):
        record = await tools.execute_one(
            TOOL_NAME,
            {
                "operations": [
                    {"id": "hold", "op": "rba.longest_hold"},
                    {
                        "id": "peak",
                        "op": "asx.price_extreme",
                        "args": {"ticker": "BHP.AX"},
                    },
                ]
            },
        )
        assert record.result["status"] == "ok"
        results = {r["id"]: r for r in record.result["results"]}
        assert results["hold"]["data"]["gap_days"] == 1036
        assert results["peak"]["data"]["date"] == "2021-08-26"

    @pytest.mark.asyncio
    async def test_resolves_references_between_operations(self, tools):
        record = await tools.execute_one(
            TOOL_NAME,
            {
                "operations": [
                    {
                        "id": "peak",
                        "op": "asx.price_extreme",
                        "args": {"ticker": "BHP.AX"},
                    },
                    {
                        "id": "rate",
                        "op": "rba.rate_at_date",
                        "args": {"date": "${peak.data.date}"},
                    },
                ]
            },
        )
        results = {r["id"]: r for r in record.result["results"]}
        assert results["rate"]["data"]["requested_date"] == "2021-08-26"

    @pytest.mark.asyncio
    async def test_every_result_carries_evidence(self, tools):
        record = await tools.execute_one(
            TOOL_NAME, {"operations": [{"id": "hold", "op": "rba.longest_hold"}]}
        )
        evidence = record.result["results"][0]["evidence"]
        assert evidence["dataset"] == "rba"
        assert evidence["method"]

    @pytest.mark.asyncio
    async def test_partial_failure_keeps_the_good_results(self, tools):
        """Component grading means a partial bundle still earns points."""
        record = await tools.execute_one(
            TOOL_NAME,
            {
                "operations": [
                    {"id": "good", "op": "rba.longest_hold"},
                    {
                        "id": "bad",
                        "op": "asx.price_extreme",
                        "args": {"ticker": "NOPE.AX"},
                    },
                ]
            },
        )
        # A warning rides alongside the data rather than replacing it.
        assert record.result["data"]["status"] == "partial"
        results = {r["id"]: r for r in record.result["data"]["results"]}
        assert results["good"]["data"]["gap_days"] == 1036
        assert results["bad"]["error"]["code"] == "UNKNOWN_TICKER"

    @pytest.mark.asyncio
    async def test_total_failure_is_reported_as_a_tool_error(self, tools):
        record = await tools.execute_one(
            TOOL_NAME, {"operations": [{"id": "a", "op": "rba.does_not_exist"}]}
        )
        assert "error" in record.result

    @pytest.mark.asyncio
    async def test_malformed_arguments_are_rejected_before_execution(self, tools):
        record = await tools.execute_one(TOOL_NAME, {"operations": []})
        assert "error" in record.result

    @pytest.mark.asyncio
    async def test_concurrent_calls_do_not_mix_results(self, tools):
        import asyncio

        calls = [
            (
                TOOL_NAME,
                {
                    "operations": [
                        {
                            "id": "peak",
                            "op": "asx.price_extreme",
                            "args": {"ticker": ticker},
                        }
                    ]
                },
            )
            for ticker in ["BHP.AX", "CBA.AX", "CSL.AX"]
        ]
        records = await asyncio.gather(*(tools.execute_one(name, args) for name, args in calls))
        seen = [r.result["results"][0]["data"]["ticker"] for r in records]
        assert seen == ["BHP.AX", "CBA.AX", "CSL.AX"]


class TestResultShape:
    @pytest.mark.asyncio
    async def test_result_is_json_serialisable(self, store):
        """The bundle crosses into the LLM message stream, so it must serialise."""
        import json

        from src.tfql import PlanRequest
        from src.tools.tfql_tool import make_executor

        run = make_executor(store)
        result: ToolResult = await run(
            PlanRequest(operations=[{"id": "hold", "op": "rba.longest_hold"}])
        )
        json.dumps(result.data)
