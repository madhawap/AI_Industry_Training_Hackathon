"""Cross-dataset operations and calendar reconciliation.

The three datasets use three different calendars -- RBA decision dates, ASX
trading days, AFR publication dates -- and cover different spans. Every join
must state the alignment it applied and the coverage the datasets share.
"""

from __future__ import annotations

import pytest
from src.tfql.coverage import Coverage, describe_overlap, overlap
from src.tfql.errors import ErrorCode, TFQLError


class TestCoverageArithmetic:
    def test_overlap_of_disjoint_intervals_is_none(self):
        from datetime import date

        a = Coverage("a", date(2015, 1, 1), date(2015, 12, 31), 1)
        b = Coverage("b", date(2020, 1, 1), date(2020, 12, 31), 1)
        assert overlap(a, b) is None

    def test_shared_span_is_the_intersection(self, store):
        shared = overlap(store.rba.coverage, store.asx_coverage())
        assert shared is not None
        start, end = shared
        assert start >= store.asx_coverage().start
        assert end <= store.asx_coverage().end

    def test_three_way_overlap_is_bounded_at_both_ends(self, store):
        """Each end of the shared window is set by a different dataset.

        In the mock, ASX starts on 2015-01-08 but AFR starts 2015-01-05, while
        AFR stops at 2015-03-31 and ASX runs to 2021. So the usable window for
        a three-dataset question is ASX-bounded at the start and AFR-bounded at
        the end -- roughly twelve weeks out of the RBA series' sixteen years.
        """
        start, end = overlap(store.rba.coverage, store.asx_coverage(), store.afr_coverage)
        assert start == store.asx_coverage().start
        assert end == store.afr_coverage.end
        assert (end - start).days < 100

    def test_description_names_every_dataset(self, store):
        described = describe_overlap(store.rba.coverage, store.afr_coverage)
        assert set(described["datasets"]) == {"rba", "afr"}
        assert described["shared"]


class TestRateEventMarketReturn:
    def test_joins_the_rate_to_each_ticker(self, run):
        data = run("cross.rate_event_market_return", event_date="2020-03-19").data
        assert data["cash_rate_target_pct"] > 0
        assert len(data["tickers"]) == 4
        assert data["best_performer"] == data["tickers"][0]["ticker"]

    def test_non_decision_date_resolves_as_of_and_warns(self, run):
        out = run("cross.rate_event_market_return", event_date="2020-03-19")
        assert out.data["rate_effective_date"] <= "2020-03-19"
        assert out.warnings

    def test_window_is_measured_in_trading_days(self, run):
        data = run(
            "cross.rate_event_market_return",
            event_date="2020-03-19",
            pre_days=2,
            post_days=2,
        ).data
        assert data["window_trading_days"] == 5

    def test_agrees_with_the_single_ticker_event_window(self, run):
        """Metamorphic: the cross op must not disagree with its components."""
        cross = run(
            "cross.rate_event_market_return",
            event_date="2020-03-19",
            tickers=["BHP.AX"],
            pre_days=3,
            post_days=3,
        ).data["tickers"][0]
        single = run(
            "asx.event_window",
            ticker="BHP.AX",
            event_date="2020-03-19",
            pre_days=3,
            post_days=3,
        ).data
        assert cross["window_return_pct"] == pytest.approx(single["window_return_pct"])

    def test_event_outside_rba_coverage_is_an_error(self, run):
        with pytest.raises(TFQLError) as exc:
            run("cross.rate_event_market_return", event_date="1990-01-01")
        assert exc.value.code is ErrorCode.DATE_OUTSIDE_COVERAGE


class TestNewsRateContext:
    def test_each_article_carries_the_rate_in_effect(self, run):
        data = run("cross.news_rate_context", query="rate cut", limit=3).data
        assert data["articles"]
        for article in data["articles"]:
            assert article["cash_rate_target_pct"] > 0
            # The rate must have been set on or before publication.
            assert article["rate_effective_date"] <= article["publication_date"]

    def test_bundle_carries_what_the_synthesiser_needs_for_sentiment(self, run):
        """This is the bundle routed to the fine-tuned model for sentiment."""
        article = run("cross.news_rate_context", query="RBA", limit=1).data["articles"][0]
        assert {
            "headline",
            "publication_date",
            "excerpt",
            "cash_rate_target_pct",
        } <= set(article)
        assert article["excerpt"]

    def test_coverage_detail_names_all_three_datasets(self, run):
        evidence = run("cross.news_rate_context", query="RBA").evidence.to_dict()
        assert set(evidence["coverage_detail"]["datasets"]) == {"rba", "asx", "afr"}

    def test_no_match_warns_rather_than_raising(self, run):
        out = run("cross.news_rate_context", query="zzzznotaword")
        assert out.data["article_count"] == 0
        assert out.warnings
