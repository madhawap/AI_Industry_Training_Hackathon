"""ASX price operations.

Two semantics are pinned here rather than left to the caller, because leaving
them open is how two correct implementations produce different graded numbers:

  * **Returns are close-to-close.** Stated in every result's evidence.
  * **Requested dates are aligned to trading days** through the shared
    predecessor/successor helper, and the resolved date is always reported
    alongside the requested one.

Where a genuine modelling choice exists -- drawdown on closes or on intraday
lows, a basket rebalanced or held -- it is an explicit argument with a
documented default, never an implicit assumption.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import Field

from ..dates import Alignment, iso, parse_date, resolve, validate_window
from ..errors import ErrorCode, TFQLError
from ..evidence import Evidence, OpOutput
from ..invariants import check, check_non_empty
from ..precision import decimal_to_pct, round_price
from ..registry import Args, register
from ..store import Store, TickerSeries

DATASET = "asx"

PriceField = Literal["open", "high", "low", "close", "volume"]


class Direction(StrEnum):
    HIGHEST = "highest"
    LOWEST = "lowest"


# --------------------------------------------------------------------- shared


def _resolve_day(
    series: TickerSeries,
    day: date,
    alignment: Alignment | str,
    label: str,
) -> tuple[int, date]:
    """Align a calendar date to one of this ticker's trading days."""
    return resolve(
        series.dates,
        day,
        alignment,
        dataset=f"asx:{series.ticker}",
        label=label,
    )


def _span(
    series: TickerSeries,
    start: str | None,
    end: str | None,
) -> tuple[int, int]:
    """Inclusive trading-day index range for an optional window."""
    lo = parse_date(start) if start else series.dates[0]
    hi = parse_date(end) if end else series.dates[-1]
    validate_window(lo, hi)
    i, _ = _resolve_day(series, lo, Alignment.NEXT, "start")
    j, _ = _resolve_day(series, hi, Alignment.PREVIOUS, "end")
    if i > j:
        raise TFQLError(
            ErrorCode.NO_MATCHING_RECORDS,
            f"no {series.ticker} trading days between {iso(lo)} and {iso(hi)}",
            dataset=DATASET,
            available=series.coverage.describe(),
        )
    return i, j


def _alignment_note(requested: date, resolved: date, alignment: str) -> str | None:
    if requested == resolved:
        return None
    return f"{iso(requested)} was not a trading day; used {iso(resolved)} (alignment={alignment})"


# --------------------------------------------------------------------- return


class ReturnArgs(Args):
    ticker: str = Field(description="ASX ticker, e.g. BHP.AX")
    start: str = Field(description="ISO start date")
    end: str = Field(description="ISO end date")
    alignment: Alignment = Field(
        default=Alignment.NEAREST,
        description="how to map non-trading dates onto trading days",
    )


@register(
    "asx.return",
    ReturnArgs,
    summary=(
        "Close-to-close return for one ticker between two dates, with the "
        "resolved trading days and both closing prices."
    ),
    datasets=("asx",),
)
def asx_return(args: ReturnArgs, store: Store) -> OpOutput:
    series = store.ticker(args.ticker)
    start_day, end_day = parse_date(args.start), parse_date(args.end)
    validate_window(start_day, end_day)

    i, resolved_start = _resolve_day(series, start_day, args.alignment, "start")
    j, resolved_end = _resolve_day(series, end_day, args.alignment, "end")

    start_close = float(series.close[i])
    end_close = float(series.close[j])
    check(start_close > 0, "start close must be positive to compute a return")

    decimal = end_close / start_close - 1.0
    # The result's sign must agree with the price move it describes.
    check(
        np.sign(decimal) == np.sign(end_close - start_close),
        "return sign disagrees with the underlying price move",
    )

    out = OpOutput(
        data={
            "ticker": series.ticker,
            "resolved_start": iso(resolved_start),
            "resolved_end": iso(resolved_end),
            "start_close": round_price(start_close),
            "end_close": round_price(end_close),
            "return_decimal": decimal,
            "return_pct": decimal_to_pct(decimal),
            "trading_days": j - i + 1,
        },
        evidence=Evidence(
            dataset=DATASET,
            method="(end_close / start_close) - 1",
            records_used=j - i + 1,
            coverage=series.coverage.describe(),
        )
        .note("price_field", "close")
        .note("alignment", str(args.alignment)),
    )
    for note in (
        _alignment_note(start_day, resolved_start, str(args.alignment)),
        _alignment_note(end_day, resolved_end, str(args.alignment)),
    ):
        if note:
            out.warn(note)
    return out


# -------------------------------------------------------------- price_extreme


class PriceExtremeArgs(Args):
    ticker: str = Field(description="ASX ticker, e.g. BHP.AX")
    field: PriceField = Field(default="close", description="which price or volume series to rank")
    direction: Direction = Field(default=Direction.HIGHEST)
    n: int = Field(default=1, ge=1, le=20, description="how many rows to return")
    start: str | None = Field(default=None, description="ISO window start")
    end: str | None = Field(default=None, description="ISO window end")


@register(
    "asx.price_extreme",
    PriceExtremeArgs,
    summary=(
        "Highest or lowest close/open/high/low/volume for a ticker, with the "
        "date it occurred. Set n>1 for a ranked list (e.g. three biggest "
        "volume days)."
    ),
    datasets=("asx",),
)
def price_extreme(args: PriceExtremeArgs, store: Store) -> OpOutput:
    series = store.ticker(args.ticker)
    i, j = _span(series, args.start, args.end)
    values = series.field(args.field)[i : j + 1]
    check_non_empty(values.size, "no rows in the requested window")

    order = np.argsort(values, kind="stable")
    if args.direction is Direction.HIGHEST:
        order = order[::-1]
    # Stable sort plus an explicit date tiebreak keeps ties reproducible.
    picks = sorted(
        (int(k) + i for k in order[: args.n]),
        key=lambda idx: (
            -float(series.field(args.field)[idx])
            if args.direction is Direction.HIGHEST
            else float(series.field(args.field)[idx]),
            series.dates[idx],
        ),
    )

    rounder = (lambda v: int(v)) if args.field == "volume" else round_price
    entries = [
        {
            "date": iso(series.dates[k]),
            args.field: rounder(float(series.field(args.field)[k])),
        }
        for k in picks
    ]

    data: dict[str, object] = {
        "ticker": series.ticker,
        "field": args.field,
        "direction": str(args.direction),
        **entries[0],
    }
    if args.n > 1:
        data["ranked"] = entries

    return OpOutput(
        data=data,
        evidence=Evidence(
            dataset=DATASET,
            method=f"{args.direction} {args.field} over the window, ties broken by earliest date",
            records_used=j - i + 1,
            coverage=f"{iso(series.dates[i])} to {iso(series.dates[j])}",
        ),
    )


# --------------------------------------------------------------- biggest_move


class BiggestMoveArgs(Args):
    ticker: str = Field(description="ASX ticker, e.g. BHP.AX")
    direction: Literal["gain", "decline"] = Field(
        description="largest single-day rise or fall in closing price"
    )
    n: int = Field(default=1, ge=1, le=20)
    start: str | None = Field(default=None, description="ISO window start")
    end: str | None = Field(default=None, description="ISO window end")


@register(
    "asx.biggest_move",
    BiggestMoveArgs,
    summary=(
        "Largest single-day percentage gain or decline in closing price, with "
        "the date and the closes on either side. Distinct from price_extreme, "
        "which ranks price levels rather than changes."
    ),
    datasets=("asx",),
)
def biggest_move(args: BiggestMoveArgs, store: Store) -> OpOutput:
    series = store.ticker(args.ticker)
    i, j = _span(series, args.start, args.end)
    # index 0 of daily_return is nan; a window starting there has no prior close
    lo = max(i, 1)
    returns = series.daily_return[lo : j + 1]
    check_non_empty(returns.size, "window has no consecutive trading-day pair")

    order = np.argsort(returns, kind="stable")
    if args.direction == "gain":
        order = order[::-1]
    picks = [int(k) + lo for k in order[: args.n]]

    entries = [
        {
            "date": iso(series.dates[k]),
            "pct_change": decimal_to_pct(float(series.daily_return[k])),
            "previous_close": round_price(float(series.close[k - 1])),
            "close": round_price(float(series.close[k])),
            "previous_date": iso(series.dates[k - 1]),
        }
        for k in picks
    ]
    top = entries[0]
    check(
        (top["pct_change"] >= 0) if args.direction == "gain" else True,
        "largest gain resolved to a negative move",
    )

    data: dict[str, object] = {"ticker": series.ticker, "direction": args.direction}
    data.update(top)
    if args.n > 1:
        data["ranked"] = entries

    out = OpOutput(
        data=data,
        evidence=Evidence(
            dataset=DATASET,
            method="close_t / close_(t-1) - 1 over consecutive trading days",
            records_used=int(returns.size),
            coverage=f"{iso(series.dates[i])} to {iso(series.dates[j])}",
        ).note("price_field", "close"),
    )
    out.warn(
        "single-day moves are between consecutive trading days, so a move may "
        "span a weekend or public holiday"
    )
    return out


# --------------------------------------------------------------- max_drawdown


class MaxDrawdownArgs(Args):
    ticker: str = Field(description="ASX ticker, e.g. BHP.AX")
    basis: Literal["close", "intraday"] = Field(
        default="close",
        description=(
            "close = peak and trough measured on closing prices; "
            "intraday = peak on highs, trough on lows (a deeper figure)"
        ),
    )
    start: str | None = Field(default=None, description="ISO window start")
    end: str | None = Field(default=None, description="ISO window end")


@register(
    "asx.max_drawdown",
    MaxDrawdownArgs,
    summary=(
        "Largest peak-to-trough decline for a ticker, with the peak and trough "
        "dates and prices, and the recovery date if the peak was regained."
    ),
    datasets=("asx",),
)
def max_drawdown(args: MaxDrawdownArgs, store: Store) -> OpOutput:
    series = store.ticker(args.ticker)
    i, j = _span(series, args.start, args.end)
    peaks_src = series.high if args.basis == "intraday" else series.close
    trough_src = series.low if args.basis == "intraday" else series.close

    peak_val = -np.inf
    peak_idx = i
    best = 0.0
    best_peak = best_trough = i

    # Running-peak scan: one pass, tracking the deepest decline seen so far.
    for k in range(i, j + 1):
        if peaks_src[k] > peak_val:
            peak_val = float(peaks_src[k])
            peak_idx = k
        if peak_val > 0:
            dd = float(trough_src[k]) / peak_val - 1.0
            if dd < best:
                best, best_peak, best_trough = dd, peak_idx, k

    check(best <= 0, "drawdown must be zero or negative")
    check(
        series.dates[best_trough] >= series.dates[best_peak],
        "trough must not precede its peak",
    )

    recovery_idx = None
    peak_price = float(peaks_src[best_peak])
    for k in range(best_trough + 1, j + 1):
        if float(series.close[k]) >= peak_price:
            recovery_idx = k
            break

    return OpOutput(
        data={
            "ticker": series.ticker,
            "basis": args.basis,
            "max_drawdown_pct": decimal_to_pct(best),
            "peak_date": iso(series.dates[best_peak]),
            "peak_price": round_price(peak_price),
            "trough_date": iso(series.dates[best_trough]),
            "trough_price": round_price(float(trough_src[best_trough])),
            "drawdown_days": (series.dates[best_trough] - series.dates[best_peak]).days,
            "recovery_date": (iso(series.dates[recovery_idx]) if recovery_idx else None),
        },
        evidence=Evidence(
            dataset=DATASET,
            method="running-peak scan; max of (price / running_peak) - 1",
            records_used=j - i + 1,
            coverage=f"{iso(series.dates[i])} to {iso(series.dates[j])}",
        ).note("basis", args.basis),
    )


# --------------------------------------------------------------- rank_returns


class RankReturnsArgs(Args):
    tickers: list[str] | None = Field(
        default=None, description="tickers to rank; omit for all available"
    )
    start: str = Field(description="ISO start date")
    end: str = Field(description="ISO end date")
    alignment: Alignment = Field(default=Alignment.NEAREST)


@register(
    "asx.rank_returns",
    RankReturnsArgs,
    summary=(
        "Rank tickers by close-to-close return over a window, best first, with "
        "each ticker's start and end closes."
    ),
    datasets=("asx",),
)
def rank_returns(args: RankReturnsArgs, store: Store) -> OpOutput:
    tickers = args.tickers or store.tickers
    rows: list[dict[str, object]] = []
    skipped: list[str] = []

    for symbol in tickers:
        series = store.ticker(symbol)
        try:
            i, start_day = _resolve_day(series, parse_date(args.start), args.alignment, "start")
            j, end_day = _resolve_day(series, parse_date(args.end), args.alignment, "end")
        except TFQLError:
            # Coverage gaps are reported, never silently dropped.
            skipped.append(symbol)
            continue
        decimal = float(series.close[j]) / float(series.close[i]) - 1.0
        rows.append(
            {
                "ticker": symbol,
                "return_pct": decimal_to_pct(decimal),
                "return_decimal": decimal,
                "start_close": round_price(float(series.close[i])),
                "end_close": round_price(float(series.close[j])),
                "resolved_start": iso(start_day),
                "resolved_end": iso(end_day),
            }
        )

    check_non_empty(rows, "no tickers had data covering the requested window")
    rows.sort(key=lambda r: (-float(r["return_decimal"]), str(r["ticker"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    out = OpOutput(
        data={
            "ranked": rows,
            "best": rows[0]["ticker"],
            "worst": rows[-1]["ticker"],
            "ticker_count": len(rows),
        },
        evidence=Evidence(
            dataset=DATASET,
            method="per-ticker close-to-close return, sorted descending",
            records_used=len(rows),
            coverage=store.asx_coverage().describe(),
        ).note("alignment", str(args.alignment)),
    )
    if skipped:
        out.warn(f"no coverage in this window for: {', '.join(sorted(skipped))}")
    return out


# ---------------------------------------------------------------- volume_rank


class VolumeRankArgs(Args):
    tickers: list[str] | None = Field(default=None)
    agg: Literal["total", "average"] = Field(
        default="total",
        description="total and average volume can rank tickers differently",
    )
    start: str | None = Field(default=None)
    end: str | None = Field(default=None)


@register(
    "asx.volume_rank",
    VolumeRankArgs,
    summary="Rank tickers by total or average daily traded volume over a window.",
    datasets=("asx",),
)
def volume_rank(args: VolumeRankArgs, store: Store) -> OpOutput:
    tickers = args.tickers or store.tickers
    rows: list[dict[str, object]] = []
    skipped: list[str] = []

    for symbol in tickers:
        series = store.ticker(symbol)
        try:
            i, j = _span(series, args.start, args.end)
        except TFQLError:
            skipped.append(symbol)
            continue
        window = series.volume[i : j + 1]
        value = float(np.nansum(window))
        if args.agg == "average":
            value = float(np.nanmean(window))
        rows.append(
            {
                "ticker": symbol,
                f"{args.agg}_volume": round(value),
                "trading_days": j - i + 1,
            }
        )

    check_non_empty(rows, "no tickers had data covering the requested window")
    rows.sort(key=lambda r: (-int(r[f"{args.agg}_volume"]), str(r["ticker"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    out = OpOutput(
        data={"agg": args.agg, "ranked": rows, "highest": rows[0]["ticker"]},
        evidence=Evidence(
            dataset=DATASET,
            method=f"{args.agg} of daily volume per ticker, sorted descending",
            records_used=sum(int(r["trading_days"]) for r in rows),
            coverage=store.asx_coverage().describe(),
        ),
    )
    if skipped:
        out.warn(f"no coverage in this window for: {', '.join(sorted(skipped))}")
    return out


# --------------------------------------------------------------- event_window


class EventWindowArgs(Args):
    ticker: str = Field(description="ASX ticker, e.g. BHP.AX")
    event_date: str = Field(description="ISO date of the event")
    pre_days: int = Field(default=3, ge=0, le=60, description="trading days before")
    post_days: int = Field(default=3, ge=0, le=60, description="trading days after")
    alignment: Alignment = Field(default=Alignment.NEAREST)


@register(
    "asx.event_window",
    EventWindowArgs,
    summary=(
        "Price behaviour around an event date: closes before and after, the "
        "return across the window, and the event-day close. Windows are counted "
        "in trading days, not calendar days."
    ),
    datasets=("asx",),
)
def event_window(args: EventWindowArgs, store: Store) -> OpOutput:
    series = store.ticker(args.ticker)
    requested = parse_date(args.event_date)
    idx, resolved = _resolve_day(series, requested, args.alignment, "event_date")

    lo = max(0, idx - args.pre_days)
    hi = min(len(series) - 1, idx + args.post_days)
    pre_close = float(series.close[lo])
    post_close = float(series.close[hi])
    decimal = post_close / pre_close - 1.0

    out = OpOutput(
        data={
            "ticker": series.ticker,
            "requested_event_date": iso(requested),
            "resolved_event_date": iso(resolved),
            "event_close": round_price(float(series.close[idx])),
            "window_start_date": iso(series.dates[lo]),
            "window_start_close": round_price(pre_close),
            "window_end_date": iso(series.dates[hi]),
            "window_end_close": round_price(post_close),
            "window_return_pct": decimal_to_pct(decimal),
            "trading_days": hi - lo + 1,
        },
        evidence=Evidence(
            dataset=DATASET,
            method="trading-day window around the event; "
            "(window_end_close / window_start_close) - 1",
            records_used=hi - lo + 1,
            coverage=series.coverage.describe(),
        )
        .note("price_field", "close")
        .note("alignment", str(args.alignment)),
    )
    note = _alignment_note(requested, resolved, str(args.alignment))
    if note:
        out.warn(note)
    if idx - args.pre_days < 0 or idx + args.post_days > len(series) - 1:
        out.warn("window was truncated at the edge of this ticker's coverage")
    return out


# -------------------------------------------------------- equal_weight_basket


class BasketArgs(Args):
    tickers: list[str] | None = Field(default=None)
    start: str = Field(description="ISO start date")
    end: str = Field(description="ISO end date")
    rebalance: Literal["none", "daily"] = Field(
        default="none",
        description=(
            "none = equal weights at inception, held; "
            "daily = rebalanced to equal weights each day. These give "
            "different figures."
        ),
    )
    alignment: Alignment = Field(default=Alignment.NEAREST)


@register(
    "asx.equal_weight_basket",
    BasketArgs,
    summary=(
        "Return of an equally weighted basket of tickers over a window, either "
        "held from inception or rebalanced daily."
    ),
    datasets=("asx",),
)
def equal_weight_basket(args: BasketArgs, store: Store) -> OpOutput:
    tickers = args.tickers or store.tickers
    start_day, end_day = parse_date(args.start), parse_date(args.end)
    validate_window(start_day, end_day)

    members: list[dict[str, object]] = []
    daily_series: list[np.ndarray] = []
    skipped: list[str] = []

    for symbol in tickers:
        series = store.ticker(symbol)
        try:
            i, _ = _resolve_day(series, start_day, args.alignment, "start")
            j, _ = _resolve_day(series, end_day, args.alignment, "end")
        except TFQLError:
            skipped.append(symbol)
            continue
        decimal = float(series.close[j]) / float(series.close[i]) - 1.0
        members.append({"ticker": symbol, "return_pct": decimal_to_pct(decimal)})
        daily_series.append(series.daily_return[i + 1 : j + 1])

    check_non_empty(members, "no tickers had data covering the requested window")

    if args.rebalance == "none":
        basket = float(np.mean([float(m["return_pct"]) for m in members])) / 100.0
    else:
        width = min(a.size for a in daily_series)
        stacked = np.vstack([a[:width] for a in daily_series])
        basket = float(np.prod(1.0 + np.nanmean(stacked, axis=0)) - 1.0)

    out = OpOutput(
        data={
            "return_pct": decimal_to_pct(basket),
            "return_decimal": basket,
            "rebalance": args.rebalance,
            "member_count": len(members),
            "members": members,
        },
        evidence=Evidence(
            dataset=DATASET,
            method=(
                "mean of member close-to-close returns"
                if args.rebalance == "none"
                else "compounded daily cross-sectional mean return"
            ),
            records_used=len(members),
            coverage=store.asx_coverage().describe(),
        ).note("rebalance", args.rebalance),
    )
    if skipped:
        out.warn(f"excluded for lack of coverage: {', '.join(sorted(skipped))}")
    return out


# --------------------------------------------------------------- summary_stat


class SummaryStatArgs(Args):
    tickers: list[str] | None = Field(default=None)
    field: PriceField = Field(default="close")
    agg: Literal["avg", "min", "max", "median", "stddev", "sum"] = Field(default="avg")
    start: str | None = Field(default=None)
    end: str | None = Field(default=None)
    compare_to: str | None = Field(
        default=None,
        description="a ticker to compare the others against, e.g. CBA.AX",
    )


@register(
    "asx.summary_stat",
    SummaryStatArgs,
    summary=(
        "Per-ticker aggregate of a price or volume series over a window, ranked. "
        "Set compare_to to split tickers into those above and below one ticker."
    ),
    datasets=("asx",),
)
def summary_stat(args: SummaryStatArgs, store: Store) -> OpOutput:
    tickers = args.tickers or store.tickers
    funcs = {
        "avg": np.nanmean,
        "min": np.nanmin,
        "max": np.nanmax,
        "median": np.nanmedian,
        "stddev": np.nanstd,
        "sum": np.nansum,
    }
    fn = funcs[args.agg]

    rows: list[dict[str, object]] = []
    for symbol in tickers:
        series = store.ticker(symbol)
        i, j = _span(series, args.start, args.end)
        value = float(fn(series.field(args.field)[i : j + 1]))
        rows.append(
            {
                "ticker": symbol,
                f"{args.field}_{args.agg}": round_price(value),
                "trading_days": j - i + 1,
            }
        )

    check_non_empty(rows, "no tickers had data covering the requested window")
    key = f"{args.field}_{args.agg}"
    rows.sort(key=lambda r: (-float(r[key]), str(r["ticker"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    data: dict[str, object] = {
        "field": args.field,
        "agg": args.agg,
        "ranked": rows,
        "highest": rows[0]["ticker"],
        "lowest": rows[-1]["ticker"],
    }

    if args.compare_to:
        anchor = next((r for r in rows if r["ticker"] == args.compare_to), None)
        if anchor is None:
            raise TFQLError(
                ErrorCode.UNKNOWN_TICKER,
                f"compare_to ticker {args.compare_to!r} is not in the result set",
                requested=args.compare_to,
                available=[str(r["ticker"]) for r in rows],
            )
        threshold = float(anchor[key])
        data["compare_to"] = args.compare_to
        data["compare_to_value"] = anchor[key]
        data["above"] = [r for r in rows if float(r[key]) > threshold]
        data["below"] = [r for r in rows if float(r[key]) < threshold]

    return OpOutput(
        data=data,
        evidence=Evidence(
            dataset=DATASET,
            method=f"{args.agg} of {args.field} per ticker, sorted descending",
            records_used=sum(int(r["trading_days"]) for r in rows),
            coverage=store.asx_coverage().describe(),
        ),
    )
