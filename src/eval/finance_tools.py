"""Real deterministic tools over the hackathon's RBA / ASX / AFR datasets.

Per the challenge brief: "Use structured parsing and deterministic
calculations for RBA and ASX data. Use ... full-text search ... for AFR
records" -- so every tool here does exact arithmetic or exact substring
matching against the parsed data in ``finance_data.py``, never an LLM
guess. The agent (``finance_agent.py``) is only supposed to *select* these
tools and *synthesize* their results, not compute the numbers itself.

AFR pattern matching follows the brief's non-negotiable methodology:
case-insensitive, whole-word (``\\bpattern\\b``), matched once per record
across ``HEADLINE + SUBHEAD + INTRO + TEXT`` combined -- never per-field,
never a bare substring search.
"""

from __future__ import annotations

import asyncio
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from langchain.tools import tool

# Make the sibling finance_data import resolve regardless of how this file
# is loaded -- see the matching note in finance_agent.py.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from finance_data import (  # noqa: E402
    combined_text,
    in_range,
    list_asx_tickers,
    load_afr,
    load_asx,
    load_rba,
)

_HEADLINE_PREVIEW_LIMIT = 200


def _parse_date(value: str | None):
    if not value:
        return None
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def _preview(text: str, limit: int = _HEADLINE_PREVIEW_LIMIT) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... (truncated, {len(text)} chars total)"


# ---------------------------------------------------------------------------
# RBA cash-rate decisions
# ---------------------------------------------------------------------------
@tool
async def rba_extreme_rate(mode: Literal["lowest", "highest"] = "lowest") -> str:
    """Find the lowest or highest RBA cash-rate target in the dataset.

    Args:
        mode: "lowest" or "highest".
    """
    rows = load_rba()
    target = min(r["rate"] for r in rows) if mode == "lowest" else max(r["rate"] for r in rows)
    matching = [r for r in rows if r["rate"] == target]
    first_date = min(r["date"] for r in matching)
    return (
        f"The {mode} cash-rate target in the dataset is {target}%, which "
        f"first took effect on {first_date.isoformat()}, across {len(matching)} "
        "decision record(s) at that rate."
    )


@tool
async def rba_longest_gap() -> str:
    """Find the longest stretch (in days) between two consecutive non-zero
    RBA cash-rate changes (i.e. the longest period the rate held steady
    between actual moves, ignoring "no change" decisions)."""
    nonzero = [r for r in load_rba() if r["change"] != 0]
    if len(nonzero) < 2:
        return "Not enough non-zero rate changes in the dataset to compute a gap."
    best = max(
        (
            ((b["date"] - a["date"]).days, a, b)
            for a, b in zip(nonzero, nonzero[1:])
        ),
        key=lambda item: item[0],
    )
    days, before, after = best
    return (
        f"The longest stretch between two non-zero rate changes was {days} "
        f"days, from {before['date'].isoformat()} to {after['date'].isoformat()}, "
        f"during which the rate held at {before['rate']}% before changing to "
        f"{after['rate']}%."
    )


@tool
async def rba_largest_cycle(direction: Literal["hikes", "cuts"] = "hikes") -> str:
    """Find the largest tightening (hikes) or loosening (cuts) cycle: the
    longest run of consecutive same-direction rate changes with no
    intervening reversal, and its cumulative change.

    Args:
        direction: "hikes" for consecutive increases, "cuts" for
            consecutive decreases.
    """
    rows = load_rba()
    nonzero = [r for r in rows if r["change"] != 0]
    wants_positive = direction == "hikes"

    best_cycle: list[dict] = []
    current: list[dict] = []
    for row in nonzero:
        matches = row["change"] > 0 if wants_positive else row["change"] < 0
        if matches:
            current.append(row)
        else:
            if len(current) > len(best_cycle):
                best_cycle = current
            current = []
    if len(current) > len(best_cycle):
        best_cycle = current

    if not best_cycle:
        return f"No {direction} cycle found in the dataset."

    start, end = best_cycle[0], best_cycle[-1]
    cumulative = round(sum(r["change"] for r in best_cycle), 2)
    before_idx = rows.index(start)
    before_rate = rows[before_idx - 1]["rate"] if before_idx > 0 else start["rate"]
    label = "hike" if wants_positive else "cut"
    return (
        f"The largest {direction} cycle ran from {start['date'].isoformat()} to "
        f"{end['date'].isoformat()} and comprised {len(best_cycle)} {label}(s), "
        f"for a cumulative change of {cumulative:+.2f} percentage points. The "
        f"target rate immediately before the first {label} was {before_rate}%, "
        f"and the rate reached by {end['date'].isoformat()} was {end['rate']}%."
    )


@tool
async def rba_rate_at_date(as_of: str) -> str:
    """Look up the RBA cash-rate target in effect on a given date (the most
    recent decision on or before that date).

    Args:
        as_of: Date in "YYYY-MM-DD" format.
    """
    target = _parse_date(as_of)
    rows = [r for r in load_rba() if r["date"] <= target]
    if not rows:
        return f"No RBA decision on or before {as_of} exists in the dataset."
    latest = max(rows, key=lambda r: r["date"])
    return (
        f"As of {as_of}, the RBA cash-rate target was {latest['rate']}%, set "
        f"effective {latest['date'].isoformat()}."
    )


@tool
async def rba_changes_in_period(start_date: str, end_date: str) -> str:
    """Count RBA rate increases and decreases within a date range, and the
    net change over that period.

    Args:
        start_date: Inclusive start date, "YYYY-MM-DD".
        end_date: Inclusive end date, "YYYY-MM-DD".
    """
    start, end = _parse_date(start_date), _parse_date(end_date)
    in_period = [r for r in load_rba() if start <= r["date"] <= end and r["change"] != 0]
    increases = sum(1 for r in in_period if r["change"] > 0)
    decreases = sum(1 for r in in_period if r["change"] < 0)
    net = round(sum(r["change"] for r in in_period), 2)
    return (
        f"Between {start_date} and {end_date}, the RBA changed the cash rate "
        f"{len(in_period)} time(s): {increases} increase(s) and {decreases} "
        f"decrease(s), for a net change of {net:+.2f} percentage points."
    )


# ---------------------------------------------------------------------------
# ASX company prices
# ---------------------------------------------------------------------------
@tool
async def asx_price_extremes(ticker: str) -> str:
    """Find a company's highest and lowest daily closing prices and the
    dates they occurred, over the full dataset period.

    Args:
        ticker: Company name or ASX code, e.g. "BHP" or "BHP.AX".
    """
    rows = load_asx(ticker)
    hi = max(rows, key=lambda r: r["close"])
    lo = min(rows, key=lambda r: r["close"])
    return (
        f"{ticker.upper()}'s highest close was ${hi['close']:.2f} on {hi['date']}, "
        f"and its lowest close was ${lo['close']:.2f} on {lo['date']} "
        f"(data spans {rows[0]['date']} to {rows[-1]['date']})."
    )


@tool
async def asx_single_day_move(
    ticker: str, direction: Literal["gain", "decline"] = "decline"
) -> str:
    """Find a company's largest single-day percentage price move (gain or
    decline) in closing price, and when it happened.

    Args:
        ticker: Company name or ASX code, e.g. "BHP" or "BHP.AX".
        direction: "gain" for the largest rise, "decline" for the largest fall.
    """
    rows = load_asx(ticker)
    moves = [
        (
            (rows[i]["close"] / rows[i - 1]["close"] - 1) * 100,
            rows[i]["date"],
            rows[i - 1]["close"],
            rows[i]["close"],
        )
        for i in range(1, len(rows))
    ]
    best = min(moves) if direction == "decline" else max(moves)
    pct, when, prev_close, new_close = best
    verb = "decline" if direction == "decline" else "gain"
    return (
        f"{ticker.upper()}'s largest single-day {verb} was {pct:+.2f}%, on "
        f"{when}, when the close moved from ${prev_close:.2f} to ${new_close:.2f}."
    )


@tool
async def asx_total_return(ticker: str) -> str:
    """Compute a company's total close-to-close return over the full
    dataset period.

    Args:
        ticker: Company name or ASX code, e.g. "BHP" or "BHP.AX".
    """
    rows = load_asx(ticker)
    first, last = rows[0], rows[-1]
    total_return = (last["close"] / first["close"] - 1) * 100
    return (
        f"{ticker.upper()}'s total return from {first['date']} to {last['date']} "
        f"was {total_return:+.2f}%, from ${first['close']:.2f} to ${last['close']:.2f}."
    )


@tool
async def asx_rank_by_return(ticker: str) -> str:
    """Rank one company's total close-to-close return against the full
    18-company ASX basket in this dataset.

    Args:
        ticker: Company name or ASX code, e.g. "BHP" or "BHP.AX".
    """
    all_tickers = list_asx_tickers()
    ranked = sorted(
        (
            (t, (lambda rows: (rows[-1]["close"] / rows[0]["close"] - 1) * 100)(load_asx(t)))
            for t in all_tickers
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    names = [name for name, _ in ranked]
    target = ticker.strip().upper().removesuffix(".AX")
    matches = [i for i, name in enumerate(names) if name.upper() == target]
    if not matches:
        return f"Unknown ASX ticker/company {ticker!r}. Available: {names}"
    rank = matches[0] + 1
    _, total_return = ranked[matches[0]]
    return (
        f"{ticker.upper()} ranks {rank} of {len(ranked)} companies in the ASX "
        f"basket by total close-to-close return, with a return of "
        f"{total_return:+.2f}% over the full dataset period."
    )


@tool
async def asx_top_performer(direction: Literal["highest", "lowest"] = "highest") -> str:
    """Find the single best- or worst-performing company in the ASX basket
    by total close-to-close return over the full dataset period.

    Args:
        direction: "highest" for the best performer, "lowest" for the worst.
    """
    all_tickers = list_asx_tickers()
    ranked = sorted(
        (
            (t, (lambda rows: (rows[-1]["close"] / rows[0]["close"] - 1) * 100)(load_asx(t)))
            for t in all_tickers
        ),
        key=lambda item: item[1],
        reverse=(direction == "highest"),
    )
    name, total_return = ranked[0]
    label = "best" if direction == "highest" else "worst"
    return (
        f"The {label}-performing company in the ASX basket was {name}, with a "
        f"total close-to-close return of {total_return:+.2f}% over the full "
        "dataset period."
    )


@tool
async def asx_volume_stats(ticker: str) -> str:
    """Find a company's highest single-day trading volume and its average
    daily volume over the full dataset period.

    Args:
        ticker: Company name or ASX code, e.g. "BHP" or "BHP.AX".
    """
    rows = load_asx(ticker)
    peak = max(rows, key=lambda r: r["volume"])
    average = sum(r["volume"] for r in rows) / len(rows)
    return (
        f"{ticker.upper()}'s highest single-day volume was {peak['volume']:,} "
        f"shares on {peak['date']}. Average daily volume over the full period "
        f"was {average:,.0f} shares."
    )


@tool
async def asx_drawdown(ticker: str) -> str:
    """Find a company's maximum peak-to-trough drawdown in closing price
    over the full dataset period.

    Args:
        ticker: Company name or ASX code, e.g. "BHP" or "BHP.AX".
    """
    rows = load_asx(ticker)
    peak_price = rows[0]["close"]
    peak_date = rows[0]["date"]
    worst = (0.0, peak_date, peak_price, peak_date, peak_price)
    for row in rows:
        if row["close"] > peak_price:
            peak_price, peak_date = row["close"], row["date"]
        drawdown = (row["close"] / peak_price - 1) * 100
        if drawdown < worst[0]:
            worst = (drawdown, peak_date, peak_price, row["date"], row["close"])
    pct, p_date, p_price, t_date, t_price = worst
    return (
        f"{ticker.upper()}'s maximum drawdown was {pct:.2f}%, from a peak "
        f"close of ${p_price:.2f} on {p_date} to a trough close of "
        f"${t_price:.2f} on {t_date}."
    )


# ---------------------------------------------------------------------------
# AFR articles
# ---------------------------------------------------------------------------
_afr_load_lock = asyncio.Lock()


async def _afr_records() -> list[dict]:
    # The first call parses ~780MB across ~85 files (~10-15s); run it off
    # the event loop so concurrent /query requests aren't blocked. The
    # brief requires handling 3 concurrent requests correctly -- without
    # this lock, concurrent cold-start calls would each kick off their own
    # redundant 10-15s load before finance_data's cache is populated. Once
    # loaded, later calls hit that cache and return near-instantly even
    # through the lock/to_thread hop.
    async with _afr_load_lock:
        return await asyncio.to_thread(load_afr)


@tool
async def afr_corpus_stats() -> str:
    """Report the total number of AFR articles in the dataset and the
    publication date range they span."""
    records = await _afr_records()
    dates = [r["publication_date"] for r in records if r["publication_date"]]
    return (
        f"There are {len(records)} articles in total, spanning publication "
        f"dates from {min(dates).isoformat()} to {max(dates).isoformat()}."
    )


@tool
async def afr_busiest_day(start_date: str | None = None, end_date: str | None = None) -> str:
    """Find the single publication date with the most AFR articles,
    optionally restricted to a date range.

    Args:
        start_date: Optional inclusive start date, "YYYY-MM-DD".
        end_date: Optional inclusive end date, "YYYY-MM-DD".
    """
    start, end = _parse_date(start_date), _parse_date(end_date)
    counts: dict = {}
    for record in await _afr_records():
        d = record["publication_date"]
        if d and in_range(d, start, end):
            counts[d] = counts.get(d, 0) + 1
    if not counts:
        return "No articles found in that date range."
    busiest_date, count = max(counts.items(), key=lambda item: item[1])
    return f"The busiest day was {busiest_date.isoformat()}, with {count} articles published."


@tool
async def afr_pattern_count(
    pattern: str, start_date: str | None = None, end_date: str | None = None
) -> str:
    """Count AFR articles that mention a whole-word, case-insensitive
    pattern anywhere across their headline, subhead, intro, and body text
    combined. Each article counts once even if the pattern appears more
    than once or in multiple fields.

    Args:
        pattern: The exact word or phrase to search for (e.g. "bank", "NAB").
            Matched as a whole word/phrase, not a substring.
        start_date: Optional inclusive start date, "YYYY-MM-DD".
        end_date: Optional inclusive end date, "YYYY-MM-DD".
    """
    start, end = _parse_date(start_date), _parse_date(end_date)
    regex = re.compile(rf"\b{re.escape(pattern)}\b", re.IGNORECASE)
    matched = 0
    for record in await _afr_records():
        if (start or end) and not in_range(record["publication_date"], start, end):
            continue
        if regex.search(combined_text(record)):
            matched += 1
    scope = f" between {start_date} and {end_date}" if (start or end) else ""
    return (
        f"{matched} articles mention {pattern!r} in their headline, subhead, "
        f"intro, or body text{scope}."
    )


@tool
async def afr_longest_headline(
    start_date: str | None = None, end_date: str | None = None
) -> str:
    """Find the longest article headline (by character count) in the
    dataset, optionally restricted to a date range.

    Args:
        start_date: Optional inclusive start date, "YYYY-MM-DD".
        end_date: Optional inclusive end date, "YYYY-MM-DD".
    """
    start, end = _parse_date(start_date), _parse_date(end_date)
    best = None
    for record in await _afr_records():
        if (start or end) and not in_range(record["publication_date"], start, end):
            continue
        headline = str(record.get("HEADLINE") or "")
        if best is None or len(headline) > len(best[0]):
            best = (headline, record["publication_date"])
    if best is None:
        return "No articles found in that date range."
    headline, when = best
    when_str = when.isoformat() if when else "unknown date"
    return (
        f"The longest headline is {len(headline)} characters, published "
        f"{when_str}: {_preview(headline)!r}"
    )


@tool
async def afr_articles_on_date(as_of: str) -> str:
    """List article headlines published on a specific date, as evidence
    for date-specific questions.

    Args:
        as_of: Date in "YYYY-MM-DD" format.
    """
    target = _parse_date(as_of)
    headlines = [
        str(r.get("HEADLINE") or "")
        for r in await _afr_records()
        if r["publication_date"] == target
    ]
    if not headlines:
        return f"No articles found on {as_of}."
    shown = headlines[:10]
    more = f" (+{len(headlines) - 10} more)" if len(headlines) > 10 else ""
    listed = "; ".join(_preview(h, 120) for h in shown)
    return f"{len(headlines)} article(s) on {as_of}: {listed}{more}"


ALL_TOOLS = [
    rba_extreme_rate,
    rba_longest_gap,
    rba_largest_cycle,
    rba_rate_at_date,
    rba_changes_in_period,
    asx_price_extremes,
    asx_single_day_move,
    asx_total_return,
    asx_rank_by_return,
    asx_top_performer,
    asx_volume_stats,
    asx_drawdown,
    afr_corpus_stats,
    afr_busiest_day,
    afr_pattern_count,
    afr_longest_headline,
    afr_articles_on_date,
]
