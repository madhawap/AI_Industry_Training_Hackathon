"""Date resolution: the predecessor / successor lookup.

This is the single most reused primitive in TFQL. It answers one question:

    given a sorted list of dates and a target date, which entry applies?

The question has no universally correct answer, only a *chosen* one, so the
choice is always explicit at the call site:

    previous  last date <= target   ("the rate in effect on 19 Mar 2020")
    next      first date >= target  ("the next trading day after the decision")
    nearest   whichever is closer, ties broken toward the earlier date
    exact     only an entry on that exact date, else None

It is used by rba.rate_at_date, every ASX trading-day alignment, event windows
and all cross-dataset joins. A one-day error here is the mistake that cost the
portal's worked example half its marks, so the semantics live in one tested
place rather than being re-derived per operation.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from datetime import date, datetime
from enum import StrEnum

from .errors import ErrorCode, TFQLError


class Alignment(StrEnum):
    PREVIOUS = "previous"
    NEXT = "next"
    NEAREST = "nearest"
    EXACT = "exact"


def parse_date(value: str | date | datetime) -> date:
    """Parse an ISO ``YYYY-MM-DD`` date, or pass through a date object."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise TFQLError(
            ErrorCode.INVALID_ARGUMENT,
            f"expected an ISO YYYY-MM-DD date, got {value!r}",
            value=str(value),
        ) from exc


def iso(value: date) -> str:
    """Format a date for output. Dates leave TFQL as ISO strings, always."""
    return value.isoformat()


def lookup(
    dates: Sequence[date],
    target: date,
    alignment: Alignment | str = Alignment.PREVIOUS,
) -> int | None:
    """Index of the entry in ``dates`` that applies to ``target``.

    ``dates`` must be sorted ascending and free of duplicates. Returns None
    when no entry satisfies the alignment (e.g. ``previous`` for a target
    before the first entry).
    """
    alignment = Alignment(alignment)
    n = len(dates)
    if n == 0:
        return None

    if alignment is Alignment.EXACT:
        i = bisect_left(dates, target)
        return i if i < n and dates[i] == target else None

    if alignment is Alignment.PREVIOUS:
        # bisect_right places target *after* an equal entry, so -1 lands on
        # the entry itself when the target is an exact hit.
        i = bisect_right(dates, target) - 1
        return i if i >= 0 else None

    if alignment is Alignment.NEXT:
        # bisect_left places target *at* an equal entry, so an exact hit
        # resolves to itself rather than to the following entry.
        i = bisect_left(dates, target)
        return i if i < n else None

    # NEAREST: consider both neighbours, break ties toward the earlier date so
    # the result is reproducible across runs.
    before = bisect_right(dates, target) - 1
    after = bisect_left(dates, target)
    candidates = [i for i in (before, after) if 0 <= i < n]
    if not candidates:
        return None
    return min(candidates, key=lambda i: (abs((dates[i] - target).days), dates[i]))


def resolve(
    dates: Sequence[date],
    target: date,
    alignment: Alignment | str,
    *,
    dataset: str,
    label: str = "date",
) -> tuple[int, date]:
    """``lookup`` that raises DATE_OUTSIDE_COVERAGE instead of returning None.

    Returns ``(index, resolved_date)``. Callers record the resolved date in
    evidence whenever it differs from what was requested.
    """
    i = lookup(dates, target, alignment)
    if i is None:
        raise TFQLError(
            ErrorCode.DATE_OUTSIDE_COVERAGE,
            f"no {dataset} record satisfies {label}={iso(target)} "
            f"with alignment={Alignment(alignment)}",
            dataset=dataset,
            requested=iso(target),
            alignment=str(Alignment(alignment)),
            available=(f"{iso(dates[0])} to {iso(dates[-1])}" if dates else None),
        )
    return i, dates[i]


def validate_window(start: date | None, end: date | None) -> None:
    """Reject inverted windows before any data is touched."""
    if start is not None and end is not None and start > end:
        raise TFQLError(
            ErrorCode.DATE_RANGE_INVALID,
            f"start {iso(start)} is after end {iso(end)}",
            start=iso(start),
            end=iso(end),
        )


def day_span(start: date, end: date) -> int:
    """Calendar days between two dates, exclusive of the start.

    Matches the portal's reference answer: 2016-08-03 to 2019-06-05 is 1036
    days.
    """
    return (end - start).days
