"""RBA operations, anchored on the challenge brief's worked examples.

Three of the four reference answers reproduce exactly. The fourth is a known
divergence in the mock dataset itself, documented in its own test.
"""

from __future__ import annotations

import pytest
from src.tfql.errors import ErrorCode, TFQLError


class TestWorkedExamples:
    """The full-credit answers quoted in the challenge brief."""

    def test_lowest_rate_easy_example(self, run):
        # "The lowest cash-rate target was 0.1, which first took effect on
        #  2020-11-04, and 16 decision records show that rate."
        data = run("rba.rate_extreme", direction="lowest").data
        assert data["cash_rate_target_pct"] == 0.1
        assert data["first_effective_date"] == "2020-11-04"
        assert data["record_count"] == 16

    def test_longest_hold_medium_example(self, run):
        # "1036 days, from 2016-08-03 to 2019-06-05, rate held at 1.5 before
        #  changing to 1.25."
        data = run("rba.longest_hold").data
        assert data["gap_days"] == 1036
        assert data["start_date"] == "2016-08-03"
        assert data["end_date"] == "2019-06-05"
        assert data["rate_during_pct"] == 1.5
        assert data["rate_after_pct"] == 1.25

    def test_tightening_cycle_hard_example(self, run):
        # "13 hikes from 4 May 2022 to 8 Nov 2023, cumulative +4.25 points,
        #  0.1 before the first hike, 4.35 at the end."
        data = run(
            "rba.rate_cycle",
            direction="tightening",
            start="2022-05-01",
            end="2023-11-30",
        ).data
        assert data["move_count"] == 13
        assert data["cumulative_change_pct_points"] == 4.25
        assert data["start_date"] == "2022-05-04"
        assert data["end_date"] == "2023-11-08"
        assert data["rate_before_pct"] == 0.1
        assert data["rate_after_pct"] == 4.35

    def test_highest_rate_partial_credit_example(self, run):
        """The rate and count match; the date differs *in the mock data*.

        The brief's partial-credit example states the judge expects
        2010-11-02 for the first 4.75 record. The mock CSV contains
        "3 Nov 2010,+0.25,4.75", so this operation correctly reports what it
        was given. Re-check against the real dataset before the event.
        """
        data = run("rba.rate_extreme", direction="highest").data
        assert data["cash_rate_target_pct"] == 4.75
        assert data["record_count"] == 11
        assert data["first_effective_date"] == "2010-11-03"  # real data: 11-02


class TestRateExtreme:
    def test_reports_first_and_last_date_at_the_extreme(self, run):
        data = run("rba.rate_extreme", direction="lowest").data
        assert data["first_effective_date"] <= data["last_effective_date"]

    def test_window_narrows_the_search(self, run):
        data = run(
            "rba.rate_extreme",
            direction="highest",
            start="2020-01-01",
            end="2021-12-31",
        ).data
        assert data["cash_rate_target_pct"] < 4.75


class TestRateAtDate:
    def test_as_of_uses_the_last_decision_on_or_before(self, run):
        out = run("rba.rate_at_date", date="2021-08-26", resolution="as_of")
        assert out.data["effective_date"] == "2021-08-04"
        assert out.data["cash_rate_target_pct"] == 0.1
        # A resolution that moved the date must say so.
        assert out.warnings

    def test_exact_requires_a_decision_that_day(self, run):
        with pytest.raises(TFQLError) as exc:
            run("rba.rate_at_date", date="2021-08-26", resolution="exact")
        assert exc.value.code is ErrorCode.NO_MATCHING_RECORDS

    def test_exact_succeeds_on_a_decision_date(self, run):
        out = run("rba.rate_at_date", date="2020-11-04", resolution="exact")
        assert out.data["effective_date"] == "2020-11-04"
        assert not out.warnings

    def test_outside_coverage_is_an_error_not_a_guess(self, run):
        with pytest.raises(TFQLError) as exc:
            run("rba.rate_at_date", date="1990-01-01")
        assert exc.value.code is ErrorCode.DATE_OUTSIDE_COVERAGE


class TestChangeSummary:
    def test_counts_partition_the_records(self, run):
        data = run("rba.change_summary").data
        assert data["increases"] + data["decreases"] + data["holds"] == data["record_count"]
        assert data["changed_count"] == data["increases"] + data["decreases"]

    def test_cumulative_reconciles_with_start_and_end_rates(self, run):
        data = run("rba.change_summary").data
        assert (
            pytest.approx(data["rate_before_pct"] + data["cumulative_change_pct_points"])
            == data["rate_after_pct"]
        )

    def test_windowed_summary_reconciles_too(self, run):
        data = run("rba.change_summary", start="2022-01-01", end="2023-12-31").data
        assert (
            pytest.approx(data["rate_before_pct"] + data["cumulative_change_pct_points"])
            == data["rate_after_pct"]
        )
        assert data["increases"] == 13


class TestLongestHold:
    def test_hikes_and_cuts_are_measured_separately(self, run):
        any_change = run("rba.longest_hold", kind="any_change").data
        hikes = run("rba.longest_hold", kind="hike").data
        cuts = run("rba.longest_hold", kind="cut").data
        # A gap between same-signed moves cannot be shorter than the gap
        # between any two moves.
        assert hikes["gap_days"] >= any_change["gap_days"]
        assert cuts["gap_days"] >= any_change["gap_days"]

    def test_gap_days_matches_the_date_difference(self, run):
        from datetime import date

        data = run("rba.longest_hold").data
        start = date.fromisoformat(data["start_date"])
        end = date.fromisoformat(data["end_date"])
        assert (end - start).days == data["gap_days"]

    def test_ranked_output_is_descending(self, run):
        data = run("rba.longest_hold", n=3).data
        gaps = [entry["gap_days"] for entry in data["ranked"]]
        assert gaps == sorted(gaps, reverse=True)

    def test_zero_change_records_do_not_count_as_events(self, run, store):
        """The distinction behind the brief's zero-score Example 2.

        The RBA meets monthly and files change=0 rows. If those counted as
        events the longest gap would collapse to roughly a month.
        """
        data = run("rba.longest_hold").data
        assert data["gap_days"] > 365


class TestRateCycle:
    def test_reconciliation_invariant_holds(self, run):
        data = run("rba.rate_cycle", direction="tightening").data
        assert (
            pytest.approx(data["rate_before_pct"] + data["cumulative_change_pct_points"])
            == data["rate_after_pct"]
        )

    def test_easing_cycles_are_supported(self, run):
        data = run("rba.rate_cycle", direction="easing").data
        assert data["cumulative_change_pct_points"] < 0
        assert data["move_count"] >= 1
        assert (
            pytest.approx(data["rate_before_pct"] + data["cumulative_change_pct_points"])
            == data["rate_after_pct"]
        )

    def test_no_moves_in_window_is_an_error(self, run):
        with pytest.raises(TFQLError) as exc:
            run(
                "rba.rate_cycle",
                direction="tightening",
                start="2016-09-01",
                end="2017-01-01",
            )
        assert exc.value.code is ErrorCode.NO_MATCHING_RECORDS


class TestPeriodComparison:
    def test_difference_is_period_b_minus_period_a(self, run):
        data = run(
            "rba.period_comparison",
            period_a_start="2015-01-01",
            period_a_end="2015-12-31",
            period_b_start="2022-01-01",
            period_b_end="2022-12-31",
        ).data
        expected = data["period_b"]["change_pct_points"] - data["period_a"]["change_pct_points"]
        assert pytest.approx(data["difference_pct_points"]) == expected


class TestEvidence:
    def test_every_operation_reports_dataset_and_method(self, run):
        for op, args in [
            ("rba.rate_extreme", {"direction": "lowest"}),
            ("rba.longest_hold", {}),
            ("rba.change_summary", {}),
            ("rba.rate_at_date", {"date": "2020-11-04"}),
        ]:
            evidence = run(op, **args).evidence.to_dict()
            assert evidence["dataset"] == "rba"
            assert evidence["method"]
            assert evidence["records_used"] > 0
