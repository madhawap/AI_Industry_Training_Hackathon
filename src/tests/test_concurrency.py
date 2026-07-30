"""Concurrency and latency.

The harness sends up to three questions at once and penalises responses over
60 seconds. The data layer is not where that time goes -- these tests exist to
prove it stays that way, and that a shared Store serves concurrent readers
without mixing state.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from src.tfql import PlanRequest, execute

CONCURRENT_REQUESTS = 3
"""The harness default. The system must handle at least this many safely."""


def _question_plan(ticker: str, request_id: str) -> PlanRequest:
    """A representative three-operation, cross-dataset plan."""
    return PlanRequest(
        request_id=request_id,
        operations=[
            {"id": "peak", "op": "asx.price_extreme", "args": {"ticker": ticker}},
            {
                "id": "rate",
                "op": "rba.rate_at_date",
                "args": {"date": "${peak.data.date}"},
            },
            {"id": "news", "op": "afr.pattern_count", "args": {"patterns": ["RBA"]}},
        ],
    )


class TestConcurrentExecution:
    def test_three_simultaneous_plans_all_succeed(self, store):
        tickers = ["BHP.AX", "CBA.AX", "CSL.AX"]
        with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as pool:
            results = list(pool.map(lambda t: execute(_question_plan(t, f"q-{t}"), store), tickers))
        assert all(r.status == "ok" for r in results)

    def test_concurrent_results_are_not_mixed_between_requests(self, store):
        """Each response must carry its own request's data, not a neighbour's."""
        tickers = ["BHP.AX", "CBA.AX", "CSL.AX", "WES.AX"]
        with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as pool:
            results = list(
                pool.map(
                    lambda t: (t, execute(_question_plan(t, f"q-{t}"), store)),
                    tickers,
                )
            )
        for ticker, result in results:
            assert result.request_id == f"q-{ticker}"
            peak = next(r for r in result.results if r.id == "peak")
            assert peak.data["ticker"] == ticker

    def test_repeated_concurrent_runs_are_deterministic(self, store):
        """Same plan, many threads, identical answers -- no shared mutable state."""
        plan = _question_plan("BHP.AX", "q-repeat")
        with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as pool:
            results = list(pool.map(lambda _: execute(plan, store), range(12)))
        payloads = {
            tuple(sorted((r.id, str(r.data)) for r in result.results)) for result in results
        }
        assert len(payloads) == 1


class TestLatencyBudget:
    def test_a_three_operation_plan_is_far_inside_the_budget(self, store):
        """Data work must be noise against the model calls it sits between."""
        plan = _question_plan("BHP.AX", "q-latency")
        start = time.perf_counter()
        result = execute(plan, store)
        elapsed = time.perf_counter() - start
        assert result.status == "ok"
        # The real budget is 60s for the whole pipeline including two LLM
        # calls; the data layer should not consume a measurable share of it.
        assert elapsed < 1.0, f"plan took {elapsed:.3f}s"

    def test_a_full_size_plan_stays_inside_the_budget(self, store):
        plan = PlanRequest(
            request_id="q-full",
            operations=[
                {"id": "a", "op": "rba.longest_hold"},
                {"id": "b", "op": "rba.change_summary"},
                {
                    "id": "c",
                    "op": "asx.rank_returns",
                    "args": {"start": "2015-01-08", "end": "2021-12-31"},
                },
                {"id": "d", "op": "asx.max_drawdown", "args": {"ticker": "BHP.AX"}},
                {
                    "id": "e",
                    "op": "afr.pattern_count",
                    "args": {"patterns": ["RBA", "BHP", "CBA", "bank"]},
                },
                {"id": "f", "op": "afr.date_count", "args": {"granularity": "month"}},
            ],
        )
        start = time.perf_counter()
        result = execute(plan, store)
        elapsed = time.perf_counter() - start
        assert result.status == "ok"
        assert elapsed < 2.0, f"full plan took {elapsed:.3f}s"


class TestStoreImmutability:
    def test_precomputed_series_are_shared_not_copied(self, store):
        """Startup builds these once; requests read them without rebuilding."""
        first = store.ticker("BHP.AX")
        second = store.ticker("BHP.AX")
        assert first is second
        assert first.close is second.close
