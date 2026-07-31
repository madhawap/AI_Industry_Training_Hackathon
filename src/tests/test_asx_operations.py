"""ASX operations, including the edge semantics that decide graded numbers."""

from __future__ import annotations

import pytest
from src.tfql.errors import ErrorCode, TFQLError


class TestKnownAnswers:
    """Values independently derivable from the warehouse."""

    def test_bhp_highest_close(self, run):
        data = run("asx.price_extreme", ticker="BHP.AX", direction="highest").data
        assert data["date"] == "2021-08-26"
        assert data["close"] == 37.09

    def test_bhp_largest_single_day_decline(self, run):
        data = run("asx.biggest_move", ticker="BHP.AX", direction="decline").data
        assert data["date"] == "2020-03-03"
        assert data["pct_change"] == pytest.approx(-5.0931, abs=1e-4)
        assert data["previous_close"] == 26.31
        assert data["close"] == 24.97

    def test_average_close_ranking(self, run):
        data = run("asx.summary_stat", agg="avg", field="close").data
        order = [row["ticker"] for row in data["ranked"]]
        assert order == ["CSL.AX", "CBA.AX", "WES.AX", "BHP.AX"]

    def test_three_highest_volumes(self, run):
        data = run(
            "asx.price_extreme",
            ticker="BHP.AX",
            field="volume",
            direction="highest",
            n=3,
        ).data
        volumes = [row["volume"] for row in data["ranked"]]
        assert volumes == sorted(volumes, reverse=True)
        assert volumes[0] == 18439800


class TestReturn:
    def test_return_pct_is_the_decimal_times_one_hundred(self, run):
        """The synthesiser must never have to multiply."""
        data = run("asx.return", ticker="BHP.AX", start="2020-01-02", end="2020-12-31").data
        assert data["return_pct"] == pytest.approx(data["return_decimal"] * 100, abs=1e-4)

    def test_return_sign_matches_the_price_move(self, run):
        data = run("asx.return", ticker="BHP.AX", start="2020-01-02", end="2020-12-31").data
        rising = data["end_close"] > data["start_close"]
        assert (data["return_decimal"] > 0) == rising

    def test_non_trading_date_is_aligned_and_reported(self, run):
        # 2020-03-22 was a Sunday.
        out = run(
            "asx.return",
            ticker="BHP.AX",
            start="2020-03-22",
            end="2020-03-31",
            alignment="next",
        )
        assert out.data["resolved_start"] != "2020-03-22"
        assert out.warnings

    def test_alignment_choice_changes_the_resolved_day(self, run):
        previous = run(
            "asx.return",
            ticker="BHP.AX",
            start="2020-03-22",
            end="2020-03-31",
            alignment="previous",
        ).data
        following = run(
            "asx.return",
            ticker="BHP.AX",
            start="2020-03-22",
            end="2020-03-31",
            alignment="next",
        ).data
        assert previous["resolved_start"] < following["resolved_start"]

    def test_unknown_ticker_lists_the_valid_ones(self, run):
        with pytest.raises(TFQLError) as exc:
            run("asx.return", ticker="NOPE.AX", start="2020-01-02", end="2020-12-31")
        assert exc.value.code is ErrorCode.UNKNOWN_TICKER
        assert "BHP.AX" in exc.value.detail["available"]

    def test_inverted_window_is_rejected(self, run):
        with pytest.raises(TFQLError) as exc:
            run("asx.return", ticker="BHP.AX", start="2020-12-31", end="2020-01-02")
        assert exc.value.code is ErrorCode.DATE_RANGE_INVALID


class TestMaxDrawdown:
    def test_drawdown_is_negative_and_ordered(self, run):
        data = run("asx.max_drawdown", ticker="BHP.AX").data
        assert data["max_drawdown_pct"] <= 0
        assert data["trough_date"] >= data["peak_date"]
        assert data["trough_price"] <= data["peak_price"]

    def test_intraday_basis_is_at_least_as_deep_as_close(self, run):
        on_close = run("asx.max_drawdown", ticker="BHP.AX", basis="close").data
        intraday = run("asx.max_drawdown", ticker="BHP.AX", basis="intraday").data
        assert intraday["max_drawdown_pct"] <= on_close["max_drawdown_pct"]

    def test_basis_is_recorded_in_evidence(self, run):
        out = run("asx.max_drawdown", ticker="BHP.AX", basis="intraday")
        assert out.evidence.to_dict()["basis"] == "intraday"


class TestRankReturns:
    def test_ranking_is_descending_and_numbered(self, run):
        data = run("asx.rank_returns", start="2020-01-02", end="2020-12-31").data
        returns = [row["return_decimal"] for row in data["ranked"]]
        assert returns == sorted(returns, reverse=True)
        assert [row["rank"] for row in data["ranked"]] == [1, 2, 3, 4]
        assert data["best"] == data["ranked"][0]["ticker"]

    def test_single_ticker_ranking_matches_asx_return(self, run):
        """Metamorphic: ranking one ticker must agree with computing it alone."""
        ranked = run(
            "asx.rank_returns",
            tickers=["BHP.AX"],
            start="2020-01-02",
            end="2020-12-31",
        ).data["ranked"][0]
        single = run("asx.return", ticker="BHP.AX", start="2020-01-02", end="2020-12-31").data
        assert ranked["return_pct"] == pytest.approx(single["return_pct"])


class TestBasket:
    def test_one_ticker_basket_equals_that_ticker(self, run):
        """Metamorphic: a basket of one is just the constituent."""
        basket = run(
            "asx.equal_weight_basket",
            tickers=["BHP.AX"],
            start="2020-01-02",
            end="2020-12-31",
        ).data
        single = run("asx.return", ticker="BHP.AX", start="2020-01-02", end="2020-12-31").data
        assert basket["return_pct"] == pytest.approx(single["return_pct"], abs=1e-3)

    def test_rebalance_mode_is_explicit_and_recorded(self, run):
        held = run("asx.equal_weight_basket", start="2020-01-02", end="2020-12-31").data
        daily = run(
            "asx.equal_weight_basket",
            start="2020-01-02",
            end="2020-12-31",
            rebalance="daily",
        ).data
        assert held["rebalance"] == "none"
        assert daily["rebalance"] == "daily"


class TestEventWindow:
    def test_window_is_counted_in_trading_days(self, run):
        data = run(
            "asx.event_window",
            ticker="BHP.AX",
            event_date="2020-03-19",
            pre_days=3,
            post_days=3,
        ).data
        assert data["trading_days"] == 7

    def test_truncation_at_the_coverage_edge_is_warned(self, run):
        out = run(
            "asx.event_window",
            ticker="BHP.AX",
            event_date="2015-01-08",
            pre_days=10,
            post_days=3,
        )
        assert any("truncated" in w for w in out.warnings)


class TestVolumeRank:
    def test_total_and_average_are_distinct_aggregations(self, run):
        total = run("asx.volume_rank", agg="total").data
        average = run("asx.volume_rank", agg="average").data
        assert "total_volume" in total["ranked"][0]
        assert "average_volume" in average["ranked"][0]


class TestSummaryStat:
    def test_compare_to_splits_above_and_below(self, run):
        data = run("asx.summary_stat", agg="avg", field="close", compare_to="CBA.AX").data
        assert data["above"] == [
            row for row in data["ranked"] if row["close_avg"] > data["compare_to_value"]
        ]
        assert all(row["close_avg"] < data["compare_to_value"] for row in data["below"])

    def test_unknown_compare_to_is_rejected(self, run):
        with pytest.raises(TFQLError) as exc:
            run("asx.summary_stat", compare_to="NOPE.AX")
        assert exc.value.code is ErrorCode.UNKNOWN_TICKER
