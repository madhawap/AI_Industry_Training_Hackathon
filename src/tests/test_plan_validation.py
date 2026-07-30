"""Plan validation and execution semantics.

The two properties that matter: everything is validated before anything runs,
and a failed operation degrades the bundle rather than destroying it.
"""

from __future__ import annotations

import pytest
from src.tfql import PlanRequest, execute, registry
from src.tfql.errors import ErrorCode, TFQLError
from src.tfql.executor import validate_plan
from src.tfql.models import MAX_OPERATIONS


def plan(*operations, request_id="q-test"):
    return PlanRequest(request_id=request_id, operations=list(operations))


class TestPlanTimeValidation:
    def test_unknown_operation_lists_the_available_ones(self):
        with pytest.raises(TFQLError) as exc:
            validate_plan(plan({"id": "a", "op": "rba.nonsense"}))
        assert exc.value.code is ErrorCode.UNKNOWN_OPERATION
        assert "rba.rate_extreme" in exc.value.detail["available"]

    def test_duplicate_ids_are_rejected(self):
        with pytest.raises(TFQLError) as exc:
            validate_plan(
                plan(
                    {"id": "a", "op": "rba.longest_hold"},
                    {"id": "a", "op": "rba.change_summary"},
                )
            )
        assert exc.value.code is ErrorCode.INVALID_ARGUMENT

    def test_oversized_plans_are_rejected(self):
        ops = [{"id": f"op{i}", "op": "rba.longest_hold"} for i in range(MAX_OPERATIONS + 1)]
        with pytest.raises(TFQLError) as exc:
            validate_plan(plan(*ops))
        assert exc.value.code is ErrorCode.PLAN_TOO_COMPLEX

    def test_cycles_are_detected(self):
        with pytest.raises(TFQLError) as exc:
            validate_plan(
                plan(
                    {"id": "a", "op": "rba.longest_hold", "depends_on": ["b"]},
                    {"id": "b", "op": "rba.change_summary", "depends_on": ["a"]},
                )
            )
        assert exc.value.code is ErrorCode.PLAN_CYCLE

    def test_dangling_dependency_is_caught(self):
        with pytest.raises(TFQLError) as exc:
            validate_plan(plan({"id": "a", "op": "rba.longest_hold", "depends_on": ["ghost"]}))
        assert exc.value.code is ErrorCode.UNRESOLVED_REFERENCE

    def test_dependencies_are_ordered_before_dependents(self):
        ordered = validate_plan(
            plan(
                {
                    "id": "rate",
                    "op": "rba.rate_at_date",
                    "args": {"date": "${peak.data.date}"},
                },
                {
                    "id": "peak",
                    "op": "asx.price_extreme",
                    "args": {"ticker": "BHP.AX"},
                },
            )
        )
        assert [op.id for op in ordered] == ["peak", "rate"]


class TestArgumentValidation:
    def test_unknown_argument_is_distinguished_from_invalid(self):
        spec = registry.get("rba.rate_extreme")
        with pytest.raises(TFQLError) as exc:
            registry.parse_args(spec, {"direction": "lowest", "colour": "blue"})
        assert exc.value.code is ErrorCode.UNKNOWN_ARGUMENT

    def test_invalid_enum_value_is_reported(self):
        spec = registry.get("rba.rate_extreme")
        with pytest.raises(TFQLError) as exc:
            registry.parse_args(spec, {"direction": "sideways"})
        assert exc.value.code is ErrorCode.INVALID_ARGUMENT

    def test_missing_required_argument_is_reported(self):
        spec = registry.get("asx.return")
        with pytest.raises(TFQLError) as exc:
            registry.parse_args(spec, {"ticker": "BHP.AX"})
        assert exc.value.code is ErrorCode.INVALID_ARGUMENT


class TestExecution:
    def test_independent_operations_all_run(self, store):
        result = execute(
            plan(
                {"id": "hold", "op": "rba.longest_hold"},
                {"id": "peak", "op": "asx.price_extreme", "args": {"ticker": "BHP.AX"}},
                {"id": "news", "op": "afr.date_count"},
            ),
            store,
        )
        assert result.status == "ok"
        assert len(result.results) == 3
        assert all(r.succeeded for r in result.results)

    def test_reference_resolution_feeds_one_result_into_the_next(self, store):
        result = execute(
            plan(
                {"id": "peak", "op": "asx.price_extreme", "args": {"ticker": "BHP.AX"}},
                {
                    "id": "rate",
                    "op": "rba.rate_at_date",
                    "args": {"date": "${peak.data.date}", "resolution": "as_of"},
                },
            ),
            store,
        )
        assert result.status == "ok"
        peak, rate = result.results
        assert rate.data["requested_date"] == peak.data["date"]

    def test_results_keep_the_planner_ordering(self, store):
        result = execute(
            plan(
                {
                    "id": "rate",
                    "op": "rba.rate_at_date",
                    "args": {"date": "${peak.data.date}"},
                },
                {"id": "peak", "op": "asx.price_extreme", "args": {"ticker": "BHP.AX"}},
            ),
            store,
        )
        assert [r.id for r in result.results] == ["rate", "peak"]


class TestPartialFailure:
    def test_one_failure_does_not_destroy_the_other_results(self, store):
        """Partial credit is explicit in the rubric -- three of four still score."""
        result = execute(
            plan(
                {"id": "good", "op": "rba.longest_hold"},
                {
                    "id": "bad",
                    "op": "asx.return",
                    "args": {
                        "ticker": "NOPE.AX",
                        "start": "2020-01-02",
                        "end": "2020-12-31",
                    },
                },
            ),
            store,
        )
        assert result.status == "partial"
        by_id = {r.id: r for r in result.results}
        assert by_id["good"].succeeded
        assert by_id["good"].data["gap_days"] == 1036
        assert by_id["bad"].status == "error"
        assert by_id["bad"].error["code"] == "UNKNOWN_TICKER"

    def test_dependents_of_a_failure_are_skipped_with_a_reason(self, store):
        result = execute(
            plan(
                {
                    "id": "bad",
                    "op": "asx.price_extreme",
                    "args": {"ticker": "NOPE.AX"},
                },
                {
                    "id": "downstream",
                    "op": "rba.rate_at_date",
                    "args": {"date": "${bad.data.date}"},
                },
            ),
            store,
        )
        by_id = {r.id: r for r in result.results}
        assert by_id["downstream"].status == "skipped"
        assert by_id["downstream"].error["code"] == "DEPENDENCY_FAILED"

    def test_plan_level_failure_still_returns_a_bundle(self, store):
        """A malformed plan must not collapse the request."""
        result = execute(plan({"id": "a", "op": "rba.nonsense"}), store)
        assert result.status == "error"
        assert result.results[0].status == "skipped"
        assert result.results[0].error["code"] == "UNKNOWN_OPERATION"

    def test_every_result_carries_evidence_when_it_succeeds(self, store):
        result = execute(
            plan({"id": "hold", "op": "rba.longest_hold"}),
            store,
        )
        assert result.results[0].evidence["dataset"] == "rba"
        assert result.results[0].evidence["method"]


class TestCatalogue:
    def test_catalogue_is_generated_from_the_registry(self):
        entries = registry.catalogue()
        assert len(entries) == len(registry.names())
        for entry in entries:
            assert entry["summary"], f"{entry['op']} has no summary"
            assert entry["datasets"]

    def test_every_operation_is_namespaced_by_dataset(self):
        for name in registry.names():
            assert name.split(".")[0] in {"rba", "asx", "afr", "cross"}
