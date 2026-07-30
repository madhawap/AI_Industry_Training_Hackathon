"""The predecessor/successor helper.

Tested hard because it is used in roughly eight places and a one-day error in
any of them is the mistake that cost the portal's worked example half its marks.
"""

from __future__ import annotations

from datetime import date

import pytest
from src.tfql.dates import (
    Alignment,
    day_span,
    lookup,
    parse_date,
    resolve,
    validate_window,
)
from src.tfql.errors import ErrorCode, TFQLError

# A deliberately gappy series: a weekend sits between the 2nd and the 5th.
DATES = [
    date(2020, 3, 2),
    date(2020, 3, 5),
    date(2020, 3, 6),
    date(2020, 3, 10),
]


class TestExactHits:
    """A target that is present must resolve to itself under every alignment."""

    @pytest.mark.parametrize("alignment", list(Alignment))
    def test_present_date_resolves_to_itself(self, alignment):
        assert lookup(DATES, date(2020, 3, 5), alignment) == 1


class TestPrevious:
    def test_picks_the_last_earlier_date(self):
        assert lookup(DATES, date(2020, 3, 4), Alignment.PREVIOUS) == 0

    def test_returns_none_before_the_series(self):
        assert lookup(DATES, date(2020, 3, 1), Alignment.PREVIOUS) is None

    def test_picks_the_final_entry_after_the_series(self):
        assert lookup(DATES, date(2020, 3, 31), Alignment.PREVIOUS) == 3


class TestNext:
    def test_picks_the_first_later_date(self):
        assert lookup(DATES, date(2020, 3, 4), Alignment.NEXT) == 1

    def test_returns_none_after_the_series(self):
        assert lookup(DATES, date(2020, 3, 31), Alignment.NEXT) is None

    def test_picks_the_first_entry_before_the_series(self):
        assert lookup(DATES, date(2020, 3, 1), Alignment.NEXT) == 0


class TestNearest:
    def test_picks_the_closer_neighbour(self):
        # 3 Mar is one day after the 2nd, two days before the 5th.
        assert lookup(DATES, date(2020, 3, 3), Alignment.NEAREST) == 0
        # 8 Mar is two days after the 6th, two days before the 10th -- a tie.
        assert lookup(DATES, date(2020, 3, 7), Alignment.NEAREST) == 2

    def test_ties_break_toward_the_earlier_date(self):
        # 8 Mar is equidistant from 6 Mar and 10 Mar; the earlier one wins so
        # the result is reproducible between runs.
        assert lookup(DATES, date(2020, 3, 8), Alignment.NEAREST) == 2

    def test_works_outside_the_series_on_both_sides(self):
        assert lookup(DATES, date(2019, 1, 1), Alignment.NEAREST) == 0
        assert lookup(DATES, date(2021, 1, 1), Alignment.NEAREST) == 3


class TestExact:
    def test_returns_none_for_a_missing_date(self):
        assert lookup(DATES, date(2020, 3, 4), Alignment.EXACT) is None


class TestEdges:
    def test_empty_series_returns_none(self):
        assert lookup([], date(2020, 3, 4), Alignment.NEAREST) is None

    def test_single_entry_series(self):
        one = [date(2020, 3, 5)]
        assert lookup(one, date(2020, 1, 1), Alignment.NEAREST) == 0
        assert lookup(one, date(2020, 1, 1), Alignment.PREVIOUS) is None


class TestResolve:
    def test_raises_outside_coverage(self):
        with pytest.raises(TFQLError) as exc:
            resolve(DATES, date(2020, 3, 1), Alignment.PREVIOUS, dataset="asx:TEST")
        assert exc.value.code is ErrorCode.DATE_OUTSIDE_COVERAGE

    def test_returns_index_and_resolved_date(self):
        idx, resolved = resolve(DATES, date(2020, 3, 4), Alignment.PREVIOUS, dataset="asx:TEST")
        assert (idx, resolved) == (0, date(2020, 3, 2))


class TestParsing:
    def test_parses_iso(self):
        assert parse_date("2020-03-05") == date(2020, 3, 5)

    @pytest.mark.parametrize("bad", ["5 Mar 2020", "2020/03/05", "not-a-date", ""])
    def test_rejects_non_iso(self, bad):
        with pytest.raises(TFQLError) as exc:
            parse_date(bad)
        assert exc.value.code is ErrorCode.INVALID_ARGUMENT


class TestWindows:
    def test_rejects_inverted_window(self):
        with pytest.raises(TFQLError) as exc:
            validate_window(date(2020, 3, 5), date(2020, 3, 1))
        assert exc.value.code is ErrorCode.DATE_RANGE_INVALID

    def test_open_ended_windows_are_allowed(self):
        validate_window(None, date(2020, 3, 1))
        validate_window(date(2020, 3, 1), None)


class TestDaySpan:
    def test_matches_the_portal_reference_gap(self):
        # The 1036-day figure the challenge brief quotes.
        assert day_span(date(2016, 8, 3), date(2019, 6, 5)) == 1036

    def test_is_exclusive_of_the_start(self):
        assert day_span(date(2020, 1, 1), date(2020, 1, 2)) == 1
