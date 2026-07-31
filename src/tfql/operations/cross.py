"""Cross-dataset operations.

Only two operations live here. Everything else that spans datasets goes through
the plan's ``depends_on`` edges instead, because the space of
"market event x rate question" combinations is combinatorially large and a
dependency edge covers it generically where an operation catalogue never could.

The one genuinely new concern at this layer is **calendar reconciliation**: RBA
effective dates are decision dates, ASX dates are trading days, and AFR dates
are publication dates. The three calendars do not align, so every join states
the alignment it applied and the coverage the datasets actually share.
"""

from __future__ import annotations

from pydantic import Field

from ..coverage import describe_overlap
from ..dates import Alignment, iso, parse_date, resolve, validate_window
from ..errors import TFQLError
from ..evidence import Evidence, OpOutput
from ..invariants import check_non_empty
from ..precision import bp_to_pct, decimal_to_pct, round_price
from ..registry import Args, register
from ..store import AFR_ALL_TEXT, AFR_TABLE, Store

DATASET = "cross"


def _rate_on(store: Store, day, label: str) -> tuple[str, float, float]:
    """The cash-rate target in effect on a date, aligned to the last decision."""
    rba = store.rba
    rba.coverage.require(day, label=label)
    idx, effective = resolve(rba.dates, day, Alignment.PREVIOUS, dataset="rba", label=label)
    return (
        iso(effective),
        bp_to_pct(int(rba.target_bp[idx])),
        bp_to_pct(int(rba.change_bp[idx])),
    )


# ------------------------------------------------- rate_event_market_return


class RateEventMarketReturnArgs(Args):
    event_date: str = Field(description="ISO date of the rate decision or other market event")
    tickers: list[str] | None = Field(
        default=None, description="tickers to measure; omit for all available"
    )
    pre_days: int = Field(default=3, ge=0, le=60, description="trading days before")
    post_days: int = Field(default=3, ge=0, le=60, description="trading days after")
    alignment: Alignment = Field(
        default=Alignment.NEAREST,
        description="how to map the event date onto trading days",
    )


@register(
    "cross.rate_event_market_return",
    RateEventMarketReturnArgs,
    summary=(
        "How ASX prices moved around a rate decision: the cash rate in effect, "
        "and each ticker's return over a trading-day window spanning the event."
    ),
    datasets=("rba", "asx"),
)
def rate_event_market_return(args: RateEventMarketReturnArgs, store: Store) -> OpOutput:
    event_day = parse_date(args.event_date)
    effective, rate_pct, change_pct = _rate_on(store, event_day, "event_date")

    rows: list[dict[str, object]] = []
    skipped: list[str] = []
    for symbol in args.tickers or store.tickers:
        series = store.ticker(symbol)
        try:
            idx, resolved = resolve(
                series.dates,
                event_day,
                args.alignment,
                dataset=f"asx:{symbol}",
                label="event_date",
            )
        except TFQLError:
            skipped.append(symbol)
            continue
        lo = max(0, idx - args.pre_days)
        hi = min(len(series) - 1, idx + args.post_days)
        decimal = float(series.close[hi]) / float(series.close[lo]) - 1.0
        rows.append(
            {
                "ticker": symbol,
                "resolved_event_date": iso(resolved),
                "event_close": round_price(float(series.close[idx])),
                "window_start_date": iso(series.dates[lo]),
                "window_end_date": iso(series.dates[hi]),
                "window_return_pct": decimal_to_pct(decimal),
                "return_decimal": decimal,
            }
        )

    check_non_empty(rows, "no ticker had price data around the event date")
    rows.sort(key=lambda r: (-float(r["return_decimal"]), str(r["ticker"])))

    out = OpOutput(
        data={
            "event_date": iso(event_day),
            "rate_effective_date": effective,
            "cash_rate_target_pct": rate_pct,
            "rate_change_pct_points": change_pct,
            "window_trading_days": args.pre_days + args.post_days + 1,
            "best_performer": rows[0]["ticker"],
            "worst_performer": rows[-1]["ticker"],
            "tickers": rows,
        },
        evidence=Evidence(
            dataset="rba+asx",
            method=(
                "cash rate resolved as-of the event date; per-ticker "
                "close-to-close return across a trading-day event window"
            ),
            records_used=len(rows),
            coverage=str(describe_overlap(store.rba.coverage, store.asx_coverage())["shared"]),
        )
        .note("alignment", str(args.alignment))
        .note("price_field", "close"),
    )
    if effective != iso(event_day):
        out.warn(f"no rate decision on {iso(event_day)}; used the rate set on {effective}")
    if skipped:
        out.warn(f"no price coverage at this date for: {', '.join(sorted(skipped))}")
    return out


# ---------------------------------------------------------- news_rate_context


class NewsRateContextArgs(Args):
    query: str = Field(description="search terms for the article retrieval")
    start: str | None = Field(default=None, description="ISO window start")
    end: str | None = Field(default=None, description="ISO window end")
    limit: int = Field(default=5, ge=1, le=15)
    excerpt_chars: int = Field(default=600, ge=100, le=2000)


@register(
    "cross.news_rate_context",
    NewsRateContextArgs,
    summary=(
        "Articles matching a query together with the cash rate in effect when "
        "each was published. This is the bundle to use for article-grounded "
        "sentiment questions."
    ),
    datasets=("afr", "rba"),
)
def news_rate_context(args: NewsRateContextArgs, store: Store) -> OpOutput:
    lo = parse_date(args.start) if args.start else None
    hi = parse_date(args.end) if args.end else None
    validate_window(lo, hi)
    lo, hi = store.afr_coverage.clamp(lo, hi)

    sql = (
        f"select headline, subhead, publication_date, "
        f"substr(coalesce(text, ''), 1, {args.excerpt_chars}), score from "
        f"(select *, fts_main_{AFR_TABLE}.match_bm25(article_id, ?) as score "
        f" from {AFR_TABLE}) "
        f"where score is not null and publication_date between ? and ? "
        f"order by score desc, publication_date, headline limit {args.limit}"
    )
    rows = store.query(sql, [args.query, lo, hi])

    if not rows:
        # Fall back to literal whole-word matching over all four fields when
        # the stemmed index finds nothing.
        sql = (
            f"select headline, subhead, publication_date, "
            f"substr(coalesce(text, ''), 1, {args.excerpt_chars}), null "
            f"from {AFR_TABLE} "
            f"where regexp_matches({AFR_ALL_TEXT}, ?) "
            f"and publication_date between ? and ? "
            f"order by publication_date, headline limit {args.limit}"
        )
        rows = store.query(sql, [f"(?i){args.query}", lo, hi])

    articles: list[dict[str, object]] = []
    for headline, subhead, published, excerpt, score in rows:
        effective, rate_pct, change_pct = _rate_on(store, published, "publication_date")
        entry: dict[str, object] = {
            "headline": headline,
            "subhead": subhead,
            "publication_date": iso(published),
            "excerpt": excerpt,
            "cash_rate_target_pct": rate_pct,
            "rate_effective_date": effective,
            "rate_change_pct_points": change_pct,
        }
        if score is not None:
            entry["relevance_score"] = round(float(score), 4)
        articles.append(entry)

    out = OpOutput(
        data={
            "query": args.query,
            "article_count": len(articles),
            "window_start": iso(lo),
            "window_end": iso(hi),
            "articles": articles,
        },
        evidence=Evidence(
            dataset="afr+rba",
            method=(
                "article retrieval, then the cash rate resolved as-of each "
                "article's publication date"
            ),
            records_used=len(articles),
            coverage=str(describe_overlap(store.afr_coverage, store.rba.coverage)["shared"]),
        ).note(
            "coverage_detail",
            describe_overlap(store.afr_coverage, store.rba.coverage, store.asx_coverage()),
        ),
    )
    if not articles:
        out.warn(f"no AFR articles matched {args.query!r} between {iso(lo)} and {iso(hi)}")
    return out
