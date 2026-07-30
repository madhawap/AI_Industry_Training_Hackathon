"""RBA cash-rate operations.

Rates are handled as integer basis points throughout; the conversion back to
percent happens once, at the output boundary. Every operation returns the full
set of components its question shape is normally graded on -- a rate without its
effective date and record count is a half-credit answer.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import date
from enum import StrEnum
from itertools import pairwise
from typing import Literal

import numpy as np
from pydantic import Field

from ..dates import Alignment, day_span, iso, parse_date, resolve, validate_window
from ..errors import ErrorCode, TFQLError
from ..evidence import Evidence, OpOutput
from ..invariants import check, check_equal, check_non_empty
from ..precision import bp_to_pct
from ..registry import Args, register
from ..store import RbaSeries, Store

DATASET = "rba"


# --------------------------------------------------------------------- shared


def _window(rba: RbaSeries, start: str | date | None, end: str | date | None) -> tuple[int, int]:
    """Inclusive index range covering the requested window.

    Clamps to coverage first, so a window that overhangs the data is narrowed
    and reported rather than silently returning nothing.
    """
    start = parse_date(start) if start else None
    end = parse_date(end) if end else None
    validate_window(start, end)
    lo, hi = rba.coverage.clamp(start, end)
    i = bisect_left(rba.dates, lo)
    j = bisect_right(rba.dates, hi) - 1
    if i > j:
        raise TFQLError(
            ErrorCode.NO_MATCHING_RECORDS,
            f"no RBA decisions between {iso(lo)} and {iso(hi)}",
            dataset=DATASET,
            available=rba.coverage.describe(),
        )
    return i, j


def _window_note(rba: RbaSeries, i: int, j: int) -> str:
    return f"{iso(rba.dates[i])} to {iso(rba.dates[j])}"


class Direction(StrEnum):
    HIGHEST = "highest"
    LOWEST = "lowest"


class HoldKind(StrEnum):
    ANY_CHANGE = "any_change"
    HIKE = "hike"
    CUT = "cut"


class CycleDirection(StrEnum):
    TIGHTENING = "tightening"
    EASING = "easing"


def _qualifying(rba: RbaSeries, kind: HoldKind) -> np.ndarray:
    """Indices of decisions that count as an event for gap/cycle purposes.

    The RBA meets roughly monthly and emits rows with ``change = 0``. "Held
    unchanged" means the gap between consecutive *changes*, not between
    consecutive *records* -- conflating the two is the portal's zero-score
    Example 2.
    """
    if kind is HoldKind.HIKE:
        return np.flatnonzero(rba.change_bp > 0)
    if kind is HoldKind.CUT:
        return np.flatnonzero(rba.change_bp < 0)
    return np.flatnonzero(rba.change_bp != 0)


# ------------------------------------------------------------- rate_extreme


class RateExtremeArgs(Args):
    direction: Direction = Field(description="highest or lowest cash-rate target")
    start: str | None = Field(default=None, description="ISO window start, inclusive")
    end: str | None = Field(default=None, description="ISO window end, inclusive")


@register(
    "rba.rate_extreme",
    RateExtremeArgs,
    summary=(
        "Highest or lowest cash-rate target, with the date it first took effect, "
        "the date it last applied, and how many decision records show it."
    ),
    datasets=("rba",),
)
def rate_extreme(args: RateExtremeArgs, store: Store) -> OpOutput:
    rba = store.rba
    i, j = _window(rba, args.start, args.end)
    targets = rba.target_bp[i : j + 1]

    extreme_bp = int(targets.max() if args.direction is Direction.HIGHEST else targets.min())
    # Integer comparison: exact by construction, which float rates would not be.
    matches = np.flatnonzero(targets == extreme_bp) + i
    check_non_empty(matches.size, "no records matched the extreme rate")

    first_idx, last_idx = int(matches[0]), int(matches[-1])
    return OpOutput(
        data={
            "direction": str(args.direction),
            "cash_rate_target_pct": bp_to_pct(extreme_bp),
            "first_effective_date": iso(rba.dates[first_idx]),
            "last_effective_date": iso(rba.dates[last_idx]),
            "record_count": int(matches.size),
        },
        evidence=Evidence(
            dataset=DATASET,
            method=f"{args.direction} cash_rate_target over the window, "
            "then count of decision records holding that exact rate",
            records_used=j - i + 1,
            coverage=_window_note(rba, i, j),
        ),
    )


# ------------------------------------------------------------- rate_at_date


class RateAtDateArgs(Args):
    date: str = Field(description="ISO date to look up")
    resolution: Literal["as_of", "exact"] = Field(
        default="as_of",
        description=(
            "as_of = the rate in effect on that day (last decision on or before it); "
            "exact = only if a decision took effect that exact day"
        ),
    )


@register(
    "rba.rate_at_date",
    RateAtDateArgs,
    summary=(
        "The cash-rate target in effect on a given date. Use resolution=as_of for "
        "'the rate on day X', exact for 'was there a decision on day X'."
    ),
    datasets=("rba",),
)
def rate_at_date(args: RateAtDateArgs, store: Store) -> OpOutput:
    rba = store.rba
    target_day = parse_date(args.date)
    alignment = Alignment.PREVIOUS if args.resolution == "as_of" else Alignment.EXACT

    if args.resolution == "exact":
        idx = None
        pos = bisect_left(rba.dates, target_day)
        if pos < len(rba.dates) and rba.dates[pos] == target_day:
            idx = pos
        if idx is None:
            raise TFQLError(
                ErrorCode.NO_MATCHING_RECORDS,
                f"no RBA decision took effect on {iso(target_day)}",
                dataset=DATASET,
                requested=iso(target_day),
            )
        resolved = rba.dates[idx]
    else:
        rba.coverage.require(target_day, label="date")
        idx, resolved = resolve(rba.dates, target_day, alignment, dataset=DATASET, label="date")

    out = OpOutput(
        data={
            "requested_date": iso(target_day),
            "effective_date": iso(resolved),
            "cash_rate_target_pct": bp_to_pct(int(rba.target_bp[idx])),
            "change_pct_points": bp_to_pct(int(rba.change_bp[idx])),
        },
        evidence=Evidence(
            dataset=DATASET,
            method=f"decision resolved with alignment={alignment}",
            records_used=1,
            coverage=rba.coverage.describe(),
        ).note("resolution", args.resolution),
    )
    if resolved != target_day:
        out.warn(f"no decision on {iso(target_day)}; used the rate set on {iso(resolved)}")
    return out


# ------------------------------------------------------------ change_summary


class ChangeSummaryArgs(Args):
    start: str | None = Field(default=None, description="ISO window start, inclusive")
    end: str | None = Field(default=None, description="ISO window end, inclusive")


@register(
    "rba.change_summary",
    ChangeSummaryArgs,
    summary=(
        "Over a window: how many decision records there were, how many changed the "
        "rate, the increase/decrease/hold split, and the cumulative move."
    ),
    datasets=("rba",),
)
def change_summary(args: ChangeSummaryArgs, store: Store) -> OpOutput:
    rba = store.rba
    i, j = _window(rba, args.start, args.end)
    changes = rba.change_bp[i : j + 1]

    increases = int(np.count_nonzero(changes > 0))
    decreases = int(np.count_nonzero(changes < 0))
    holds = int(np.count_nonzero(changes == 0))
    cumulative_bp = int(changes.sum())

    # Reconciliation: summing the individual changes must equal the net move of
    # the target across the window. Two independent paths to one number -- a
    # dropped or double-counted decision breaks this.
    rate_before_bp = int(rba.target_bp[i]) - int(rba.change_bp[i])
    check_equal(
        cumulative_bp,
        int(rba.target_bp[j]) - rate_before_bp,
        "sum of rate changes does not reconcile with the net target move",
        window=_window_note(rba, i, j),
    )

    return OpOutput(
        data={
            "record_count": j - i + 1,
            "changed_count": increases + decreases,
            "increases": increases,
            "decreases": decreases,
            "holds": holds,
            "cumulative_change_pct_points": bp_to_pct(cumulative_bp),
            "rate_before_pct": bp_to_pct(rate_before_bp),
            "rate_after_pct": bp_to_pct(int(rba.target_bp[j])),
            "window_start": iso(rba.dates[i]),
            "window_end": iso(rba.dates[j]),
        },
        evidence=Evidence(
            dataset=DATASET,
            method="sign counts over change_pct, cumulative sum reconciled "
            "against the net target move",
            records_used=j - i + 1,
            coverage=_window_note(rba, i, j),
        ),
    )


# ------------------------------------------------------------- longest_hold


class LongestHoldArgs(Args):
    kind: HoldKind = Field(
        default=HoldKind.ANY_CHANGE,
        description=(
            "any_change = longest stretch with no rate move at all; "
            "hike / cut = longest gap between two consecutive moves of that type"
        ),
    )
    n: int = Field(default=1, ge=1, le=10, description="how many top gaps to return")


@register(
    "rba.longest_hold",
    LongestHoldArgs,
    summary=(
        "The longest stretch between rate moves, in days, with its start and end "
        "dates, the rate held during it, and the rate it moved to."
    ),
    datasets=("rba",),
)
def longest_hold(args: LongestHoldArgs, store: Store) -> OpOutput:
    rba = store.rba
    events = _qualifying(rba, args.kind)
    if events.size < 2:
        raise TFQLError(
            ErrorCode.NO_MATCHING_RECORDS,
            f"fewer than two {args.kind} events in the RBA series",
            dataset=DATASET,
            found=int(events.size),
        )

    gaps = [
        (day_span(rba.dates[int(a)], rba.dates[int(b)]), int(a), int(b))
        for a, b in pairwise(events)
    ]
    # Ties break toward the earlier start date so results are reproducible.
    gaps.sort(key=lambda g: (-g[0], rba.dates[g[1]]))
    top = gaps[: args.n]

    def describe(gap_days: int, a: int, b: int) -> dict[str, object]:
        check_equal(
            gap_days,
            day_span(rba.dates[a], rba.dates[b]),
            "gap_days does not match the date difference",
        )
        return {
            "gap_days": gap_days,
            "start_date": iso(rba.dates[a]),
            "end_date": iso(rba.dates[b]),
            "rate_during_pct": bp_to_pct(int(rba.target_bp[a])),
            "rate_after_pct": bp_to_pct(int(rba.target_bp[b])),
        }

    entries = [describe(*g) for g in top]
    data: dict[str, object] = dict(entries[0])
    data["kind"] = str(args.kind)
    if args.n > 1:
        data["ranked"] = entries

    return OpOutput(
        data=data,
        evidence=Evidence(
            dataset=DATASET,
            method=f"maximum day gap between consecutive {args.kind} events "
            "(zero-change records excluded from the event series)",
            records_used=int(events.size),
            coverage=rba.coverage.describe(),
        ),
    )


# --------------------------------------------------------------- rate_cycle


class RateCycleArgs(Args):
    direction: CycleDirection = Field(
        description="tightening = a run of consecutive hikes; easing = a run of cuts"
    )
    start: str | None = Field(default=None, description="ISO window start, inclusive")
    end: str | None = Field(default=None, description="ISO window end, inclusive")
    select: Literal["largest", "latest"] = Field(
        default="largest",
        description="which cycle to return when the window contains several",
    )


@register(
    "rba.rate_cycle",
    RateCycleArgs,
    summary=(
        "A tightening or easing cycle: how many moves it contained, the cumulative "
        "change in percentage points, its start and end dates, and the rate before "
        "and after."
    ),
    datasets=("rba",),
)
def rate_cycle(args: RateCycleArgs, store: Store) -> OpOutput:
    rba = store.rba
    i, j = _window(rba, args.start, args.end)

    sign = 1 if args.direction is CycleDirection.TIGHTENING else -1
    moves = [
        k for k in range(i, j + 1) if (rba.change_bp[k] > 0 if sign > 0 else rba.change_bp[k] < 0)
    ]
    if not moves:
        raise TFQLError(
            ErrorCode.NO_MATCHING_RECORDS,
            f"no {args.direction} moves between {iso(rba.dates[i])} and {iso(rba.dates[j])}",
            dataset=DATASET,
        )

    # A cycle is a maximal run of same-signed moves with no opposite move in
    # between. Zero-change records do not interrupt a cycle; a reversal does.
    runs: list[list[int]] = [[moves[0]]]
    for prev, cur in pairwise(moves):
        reversed_between = any(
            (rba.change_bp[k] < 0 if sign > 0 else rba.change_bp[k] > 0)
            for k in range(prev + 1, cur)
        )
        if reversed_between:
            runs.append([cur])
        else:
            runs[-1].append(cur)

    def run_magnitude(run: list[int]) -> int:
        return abs(sum(int(rba.change_bp[k]) for k in run))

    chosen = max(runs, key=run_magnitude) if args.select == "largest" else runs[-1]

    first, last = chosen[0], chosen[-1]
    cumulative_bp = sum(int(rba.change_bp[k]) for k in chosen)
    rate_before_bp = int(rba.target_bp[first]) - int(rba.change_bp[first])
    rate_after_bp = int(rba.target_bp[last])

    # The identity that catches a dropped or double-counted move.
    check_equal(
        rate_before_bp + cumulative_bp,
        rate_after_bp,
        "cycle moves do not reconcile with the rate before and after",
        cycle=f"{iso(rba.dates[first])} to {iso(rba.dates[last])}",
    )
    check(
        (cumulative_bp > 0) if sign > 0 else (cumulative_bp < 0),
        f"{args.direction} cycle has a cumulative move of the wrong sign",
    )

    return OpOutput(
        data={
            "direction": str(args.direction),
            "move_count": len(chosen),
            "cumulative_change_pct_points": bp_to_pct(cumulative_bp),
            "start_date": iso(rba.dates[first]),
            "end_date": iso(rba.dates[last]),
            "rate_before_pct": bp_to_pct(rate_before_bp),
            "rate_after_pct": bp_to_pct(rate_after_bp),
            "duration_days": day_span(rba.dates[first], rba.dates[last]),
        },
        evidence=Evidence(
            dataset=DATASET,
            method="maximal run of same-signed rate moves, uninterrupted by a "
            "reversal; cumulative sum reconciled against rate before/after",
            records_used=len(chosen),
            coverage=_window_note(rba, i, j),
        ).note("cycles_found", len(runs)),
    )


# -------------------------------------------------------- period_comparison


class PeriodComparisonArgs(Args):
    period_a_start: str = Field(description="ISO start of the first period")
    period_a_end: str = Field(description="ISO end of the first period")
    period_b_start: str = Field(description="ISO start of the second period")
    period_b_end: str = Field(description="ISO end of the second period")


@register(
    "rba.period_comparison",
    PeriodComparisonArgs,
    summary=(
        "Compare two date ranges: the rate at each end, the move within each, the "
        "number of decisions, and the difference between the two periods."
    ),
    datasets=("rba",),
)
def period_comparison(args: PeriodComparisonArgs, store: Store) -> OpOutput:
    rba = store.rba

    def summarise(start: str, end: str) -> dict[str, object]:
        i, j = _window(rba, start, end)
        changes = rba.change_bp[i : j + 1]
        opening_bp = int(rba.target_bp[i]) - int(rba.change_bp[i])
        return {
            "start": iso(rba.dates[i]),
            "end": iso(rba.dates[j]),
            "rate_at_start_pct": bp_to_pct(opening_bp),
            "rate_at_end_pct": bp_to_pct(int(rba.target_bp[j])),
            "change_pct_points": bp_to_pct(int(changes.sum())),
            "decision_count": j - i + 1,
            "_change_bp": int(changes.sum()),
        }

    a = summarise(args.period_a_start, args.period_a_end)
    b = summarise(args.period_b_start, args.period_b_end)
    difference_bp = int(b.pop("_change_bp")) - int(a.pop("_change_bp"))

    return OpOutput(
        data={
            "period_a": a,
            "period_b": b,
            "difference_pct_points": bp_to_pct(difference_bp),
        },
        evidence=Evidence(
            dataset=DATASET,
            method="per-period opening/closing target and summed change, "
            "differenced between periods",
            records_used=int(a["decision_count"]) + int(b["decision_count"]),
            coverage=rba.coverage.describe(),
        ),
    )
