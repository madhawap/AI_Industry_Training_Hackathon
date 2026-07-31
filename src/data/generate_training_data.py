"""Fine-tuning data generator: questions, grounded answers, and real tool traces.

Produces JSONL training examples in the spirit of
``Participant_Package/public_questions.jsonl``, but generated at scale by
actually running TFQL's ``execute_plan`` against the real warehouse built from
``data set/`` (RBA cash-rate decisions, ASX prices, AFR articles) -- never by
asking a language model to invent a number. This mirrors the project's core
rule (see ``src/tfql/README.md``): a wrong number is worth the same as no
number, so nothing here is allowed to guess.

Each record carries the exact fields the challenge brief's Required Response
needs (``answer``, ``steps``, ``tool_trace``) plus the public-questions-style
grading metadata, so the file doubles as fine-tuning input (question + verified
tool results -> answer) and as extra calibration cases.

Three categories, all evidence-grounded:

  answerable     -- single- and cross-dataset questions the data supports.
                    The plan always succeeds; the answer states exactly the
                    fields the operation returned.
  unanswerable   -- coverage gaps, unknown tickers, out-of-range dates. The
                    plan is run for real, the operation raises its real
                    TFQLError (DATE_OUTSIDE_COVERAGE, UNKNOWN_TICKER,
                    NO_MATCHING_RECORDS), and the answer states the refusal
                    using the *actual* coverage bounds read from the store --
                    not a guessed cutoff.
  extrapolation  -- prediction/forecast framing (future rates, future prices,
                    "will X happen"). The plan grounds the answer in the last
                    real observation the data contains, then explicitly
                    declines to invent a forecast, per the brief's rule:
                    "state the limitation clearly ... instead of ...
                    inventing a figure."

Usage (run from the repository root so ``src`` resolves as a package):

    python3 -m src.data.generate_training_data \\
        --warehouse ./warehouse.duckdb \\
        --out training/data/generated_questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

from src.tfql import PlanRequest, Store
from src.tfql import execute as execute_plan
from src.tfql.models import OperationResult, OperationRequest, PlanResult

TOLERANCE_NOTE = (
    "Dates, tickers, rankings, counts, RBA values, and AFR values are exact. "
    "Returns, drawdowns, volatility, and calculated percentage shares allow "
    "+/-0.02 percentage points; correlations allow +/-0.001; quoted closes "
    "allow +/-0.0001; calculated average volume allows +/-1 share."
)

Render = Callable[["PlanResult", Store], tuple[str, list[str]]]


@dataclass(slots=True)
class Case:
    id: str
    category: str  # answerable | unanswerable | extrapolation
    difficulty: str  # easy | medium | hard
    datasets: list[str]
    prompt: str
    ops: list[OperationRequest]
    render: Render
    derivation_methodology: str


# --------------------------------------------------------------- formatting


def pct(v: float) -> str:
    return f"{v:+.2f}%"


def price(v: float) -> str:
    return f"${v:,.2f}"


def count(v: int) -> str:
    return f"{int(v):,}"


def op(op_id: str, name: str, **args: Any) -> OperationRequest:
    return OperationRequest(id=op_id, op=name, args=args)


def data_of(bundle: PlanResult, op_id: str) -> dict[str, Any]:
    r = _result(bundle, op_id)
    if r.status != "ok":
        raise RuntimeError(f"expected {op_id!r} to succeed, got {r.status}: {r.error}")
    assert r.data is not None
    return r.data


def error_of(bundle: PlanResult, op_id: str) -> OperationResult:
    r = _result(bundle, op_id)
    if r.status == "ok":
        raise RuntimeError(f"expected {op_id!r} to fail, but it succeeded")
    return r


def _result(bundle: PlanResult, op_id: str) -> OperationResult:
    for r in bundle.results:
        if r.id == op_id:
            return r
    raise KeyError(op_id)


def run_case(store: Store, case_id: str, ops: list[OperationRequest]) -> PlanResult:
    return execute_plan(PlanRequest(operations=ops, request_id=case_id), store)


# ------------------------------------------------------------- RBA templates


def gen_rba(store: Store) -> list[Case]:
    cases: list[Case] = []

    for direction in ("highest", "lowest"):

        def render(bundle: PlanResult, _store: Store, direction=direction) -> tuple[str, list[str]]:
            d = data_of(bundle, "extreme")
            fact = (
                f"The {direction} cash-rate target in the RBA dataset was "
                f"{d['cash_rate_target_pct']}%, which first took effect on "
                f"{d['first_effective_date']}, and {d['record_count']} decision "
                f"records show that rate."
            )
            return fact, [fact]

        cases.append(
            Case(
                id=f"GEN-RBA-extreme-{direction}",
                category="answerable",
                difficulty="easy",
                datasets=["RBA"],
                prompt=(
                    f"What is the {direction} cash-rate target in the RBA dataset, when did it "
                    "first take effect, and how many decision records show that rate?"
                ),
                ops=[op("extreme", "rba.rate_extreme", direction=direction)],
                render=render,
                derivation_methodology=(
                    f"rba.rate_extreme(direction={direction}) over the full RBA series."
                ),
            )
        )

    windows = [
        ("2015-01-01", "2021-12-31", "between 2015 and 2021"),
        ("2019-01-01", "2019-12-31", "during 2019"),
        ("2020-01-01", "2020-12-31", "during 2020"),
        ("2011-01-01", "2013-12-31", "across the 2011-2013 easing period"),
        ("2022-01-01", "2023-12-31", "across the 2022-2023 tightening cycle"),
    ]
    for start, end, phrase in windows:

        def render(bundle: PlanResult, _store: Store) -> tuple[str, list[str]]:
            d = data_of(bundle, "summary")
            fact = (
                f"Of {d['record_count']} decision records, {d['changed_count']} changed the "
                f"rate ({d['increases']} increases and {d['decreases']} decreases), a cumulative "
                f"move of {d['cumulative_change_pct_points']:+.2f} percentage points, taking the "
                f"target from {d['rate_before_pct']}% to {d['rate_after_pct']}%."
            )
            return fact, [fact]

        cases.append(
            Case(
                id=f"GEN-RBA-summary-{start}-{end}",
                category="answerable",
                difficulty="medium",
                datasets=["RBA"],
                prompt=(
                    f"{phrase[0].upper()}{phrase[1:]}, how many RBA decisions changed the cash "
                    "rate, what was the increase/decrease split, and how far did the cumulative "
                    "move take the target?"
                ),
                ops=[op("summary", "rba.change_summary", start=start, end=end)],
                render=render,
                derivation_methodology=f"rba.change_summary(start={start}, end={end}).",
            )
        )

    for kind in ("any_change", "hike", "cut"):

        def render(bundle: PlanResult, _store: Store, kind=kind) -> tuple[str, list[str]]:
            d = data_of(bundle, "hold")
            label = {"any_change": "between any two rate moves", "hike": "between two hikes", "cut": "between two cuts"}[kind]
            fact = (
                f"The longest stretch {label} was {d['gap_days']} days, from {d['start_date']} to "
                f"{d['end_date']}, during which the rate held at {d['rate_during_pct']}% before "
                f"moving to {d['rate_after_pct']}%."
            )
            return fact, [fact]

        cases.append(
            Case(
                id=f"GEN-RBA-hold-{kind}",
                category="answerable",
                difficulty="medium",
                datasets=["RBA"],
                prompt=(
                    "What was the longest stretch "
                    + {"any_change": "between two non-zero RBA rate changes", "hike": "between two RBA rate hikes", "cut": "between two RBA rate cuts"}[kind]
                    + "?"
                ),
                ops=[op("hold", "rba.longest_hold", kind=kind)],
                render=render,
                derivation_methodology=f"rba.longest_hold(kind={kind}).",
            )
        )

    for cyc_dir in ("tightening", "easing"):

        def render(bundle: PlanResult, _store: Store, cyc_dir=cyc_dir) -> tuple[str, list[str]]:
            d = data_of(bundle, "cycle")
            fact = (
                f"The largest {cyc_dir} cycle ran from {d['start_date']} to {d['end_date']} "
                f"({d['move_count']} moves over {d['duration_days']} days), a cumulative change of "
                f"{d['cumulative_change_pct_points']:+.2f} percentage points, taking the rate from "
                f"{d['rate_before_pct']}% to {d['rate_after_pct']}%."
            )
            return fact, [fact]

        cases.append(
            Case(
                id=f"GEN-RBA-cycle-{cyc_dir}",
                category="answerable",
                difficulty="hard",
                datasets=["RBA"],
                prompt=(
                    f"Identify the RBA's largest {cyc_dir} cycle: how many moves it contained, "
                    "its cumulative change, its start and end dates, and the rate before and after."
                ),
                ops=[op("cycle", "rba.rate_cycle", direction=cyc_dir, select="largest")],
                render=render,
                derivation_methodology=f"rba.rate_cycle(direction={cyc_dir}, select=largest).",
            )
        )

    period_pairs = [
        ("2019-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
        ("2015-01-01", "2015-12-31", "2021-01-01", "2021-12-31"),
    ]
    for a_start, a_end, b_start, b_end in period_pairs:

        def render(bundle: PlanResult, _store: Store) -> tuple[str, list[str]]:
            d = data_of(bundle, "cmp")
            a, b = d["period_a"], d["period_b"]
            fact = (
                f"{a['start']}-{a['end']} moved {a['change_pct_points']:+.2f}pp across "
                f"{a['decision_count']} decisions (ending at {a['rate_at_end_pct']}%), while "
                f"{b['start']}-{b['end']} moved {b['change_pct_points']:+.2f}pp across "
                f"{b['decision_count']} decisions (ending at {b['rate_at_end_pct']}%); the "
                f"difference between the two periods is {d['difference_pct_points']:+.2f} "
                "percentage points."
            )
            return fact, [fact]

        cases.append(
            Case(
                id=f"GEN-RBA-periodcmp-{a_start}-vs-{b_start}",
                category="answerable",
                difficulty="hard",
                datasets=["RBA"],
                prompt=(
                    f"Compare RBA policy in {a_start[:4]} against {b_start[:4]}: the rate move, "
                    "decision count and ending rate for each year, and the difference between them."
                ),
                ops=[
                    op(
                        "cmp",
                        "rba.period_comparison",
                        period_a_start=a_start,
                        period_a_end=a_end,
                        period_b_start=b_start,
                        period_b_end=b_end,
                    )
                ],
                render=render,
                derivation_methodology="rba.period_comparison over the two named calendar years.",
            )
        )

    for target_date, resolution in [
        ("2020-03-19", "as_of"),
        ("2019-06-05", "exact"),
        ("2022-05-04", "exact"),
    ]:

        def render(bundle: PlanResult, _store: Store) -> tuple[str, list[str]]:
            d = data_of(bundle, "rate")
            fact = (
                f"The cash-rate target in effect on {d['requested_date']} was "
                f"{d['cash_rate_target_pct']}% (decision effective {d['effective_date']}, a change "
                f"of {d['change_pct_points']:+.2f} percentage points)."
            )
            return fact, [fact]

        cases.append(
            Case(
                id=f"GEN-RBA-rateat-{target_date}",
                category="answerable",
                difficulty="easy",
                datasets=["RBA"],
                prompt=f"What was the RBA cash-rate target in effect on {target_date}?",
                ops=[op("rate", "rba.rate_at_date", date=target_date, resolution=resolution)],
                render=render,
                derivation_methodology=f"rba.rate_at_date(date={target_date}, resolution={resolution}).",
            )
        )

    return cases


# ------------------------------------------------------------- ASX templates


FOCUS_TICKERS_HINT = ["BHP.AX", "CBA.AX", "AMP.AX", "QBE.AX", "RIO.AX", "NAB.AX", "ANZ.AX", "TPG.AX"]


def gen_asx(store: Store) -> list[Case]:
    cases: list[Case] = []
    tickers = [t for t in FOCUS_TICKERS_HINT if t in store.asx]
    all_tickers = store.tickers

    for ticker in tickers:
        for start, end, label in [
            ("2018-01-01", "2018-12-31", "2018"),
            ("2021-01-01", "2021-12-31", "2021"),
        ]:

            def render(bundle: PlanResult, _store: Store) -> tuple[str, list[str]]:
                d = data_of(bundle, "ret")
                fact = (
                    f"{d['ticker']} moved from {price(d['start_close'])} on {d['resolved_start']} to "
                    f"{price(d['end_close'])} on {d['resolved_end']}, a return of "
                    f"{pct(d['return_pct'])}."
                )
                return fact, [fact]

            cases.append(
                Case(
                    id=f"GEN-ASX-return-{ticker}-{label}",
                    category="answerable",
                    difficulty="easy",
                    datasets=["ASX"],
                    prompt=f"What was {ticker}'s first-to-last close return during {label}?",
                    ops=[op("ret", "asx.return", ticker=ticker, start=start, end=end)],
                    render=render,
                    derivation_methodology=f"asx.return(ticker={ticker}, start={start}, end={end}).",
                )
            )

        for field_name, direction in [("close", "highest"), ("close", "lowest")]:

            def render(bundle: PlanResult, _store: Store, field_name=field_name) -> tuple[str, list[str]]:
                d = data_of(bundle, "extreme")
                fact = f"{d['ticker']}'s {direction} {field_name} was {price(d[field_name])} on {d['date']}."
                return fact, [fact]

            cases.append(
                Case(
                    id=f"GEN-ASX-extreme-{ticker}-{field_name}-{direction}",
                    category="answerable",
                    difficulty="easy",
                    datasets=["ASX"],
                    prompt=f"What was {ticker}'s {direction} closing price over the full sample, and on what date?",
                    ops=[op("extreme", "asx.price_extreme", ticker=ticker, field=field_name, direction=direction)],
                    render=render,
                    derivation_methodology=f"asx.price_extreme(ticker={ticker}, field={field_name}, direction={direction}).",
                )
            )

        for direction in ("gain", "decline"):

            def render(bundle: PlanResult, _store: Store, direction=direction) -> tuple[str, list[str]]:
                d = data_of(bundle, "move")
                fact = (
                    f"{d['ticker']}'s largest single-day {direction} was {pct(d['pct_change'])} on "
                    f"{d['date']}, from {price(d['previous_close'])} on {d['previous_date']} to "
                    f"{price(d['close'])}."
                )
                return fact, [fact]

            cases.append(
                Case(
                    id=f"GEN-ASX-move-{ticker}-{direction}",
                    category="answerable",
                    difficulty="medium",
                    datasets=["ASX"],
                    prompt=f"What was {ticker}'s largest single-day {direction} in closing price, and on what date?",
                    ops=[op("move", "asx.biggest_move", ticker=ticker, direction=direction)],
                    render=render,
                    derivation_methodology=f"asx.biggest_move(ticker={ticker}, direction={direction}).",
                )
            )

        def render(bundle: PlanResult, _store: Store) -> tuple[str, list[str]]:
            d = data_of(bundle, "dd")
            recovery = f", recovering by {d['recovery_date']}" if d["recovery_date"] else " (never fully recovered in-sample)"
            fact = (
                f"{d['ticker']}'s largest peak-to-trough drawdown was {pct(d['max_drawdown_pct'])}, "
                f"from {price(d['peak_price'])} on {d['peak_date']} to {price(d['trough_price'])} on "
                f"{d['trough_date']}{recovery}."
            )
            return fact, [fact]

        cases.append(
            Case(
                id=f"GEN-ASX-drawdown-{ticker}",
                category="answerable",
                difficulty="hard",
                datasets=["ASX"],
                prompt=f"What was {ticker}'s largest peak-to-trough drawdown, with the peak and trough dates?",
                ops=[op("dd", "asx.max_drawdown", ticker=ticker)],
                render=render,
                derivation_methodology=f"asx.max_drawdown(ticker={ticker}, basis=close).",
            )
        )

    non_tabcorp = [t for t in all_tickers if t != "TAH.AX"]
    for label, tick_set, start, end in [
        ("all-2018", all_tickers, "2018-01-01", "2018-12-31"),
        ("nontabcorp-2018", non_tabcorp, "2018-01-01", "2018-12-31"),
        ("nontabcorp-2021", non_tabcorp, "2021-01-01", "2021-12-31"),
        ("nontabcorp-full", non_tabcorp, "2015-01-02", "2021-12-30"),
    ]:

        def render(bundle: PlanResult, _store: Store) -> tuple[str, list[str]]:
            d = data_of(bundle, "rank")
            best, worst = d["ranked"][0], d["ranked"][-1]
            fact = (
                f"Best: {best['ticker']} at {pct(best['return_pct'])}; worst: {worst['ticker']} at "
                f"{pct(worst['return_pct'])} (across {d['ticker_count']} tickers)."
            )
            return fact, [fact]

        scope = "excluding Tabcorp" if "nontabcorp" in label else "across all tickers"
        cases.append(
            Case(
                id=f"GEN-ASX-rank-{label}",
                category="answerable",
                difficulty="medium",
                datasets=["ASX"],
                prompt=f"{scope[0].upper()}{scope[1:]}, which ticker had the best and worst return from {start} to {end}?",
                ops=[op("rank", "asx.rank_returns", tickers=tick_set, start=start, end=end)],
                render=render,
                derivation_methodology=f"asx.rank_returns(tickers={label}, start={start}, end={end}).",
            )
        )

    for agg in ("total", "average"):

        def render(bundle: PlanResult, _store: Store) -> tuple[str, list[str]]:
            d = data_of(bundle, "vol")
            top = d["ranked"][0]
            volume_key = f"{d['agg']}_volume"
            fact = f"{top['ticker']} has the highest {d['agg']} daily volume at {count(top[volume_key])} shares."
            return fact, [fact]

        cases.append(
            Case(
                id=f"GEN-ASX-volumerank-{agg}",
                category="answerable",
                difficulty="medium",
                datasets=["ASX"],
                prompt=f"Excluding Tabcorp, which ticker has the highest {agg} daily volume over the full sample?",
                ops=[op("vol", "asx.volume_rank", tickers=non_tabcorp, agg=agg)],
                render=render,
                derivation_methodology=f"asx.volume_rank(tickers=non-Tabcorp, agg={agg}).",
            )
        )

    def render_basket(bundle: PlanResult, _store: Store) -> tuple[str, list[str]]:
        d = data_of(bundle, "basket")
        fact = f"The equally weighted non-Tabcorp basket returned {pct(d['return_pct'])} from 2015-01-02 to 2021-12-30."
        return fact, [fact]

    cases.append(
        Case(
            id="GEN-ASX-basket-nontabcorp-full",
            category="answerable",
            difficulty="medium",
            datasets=["ASX"],
            prompt="Excluding Tabcorp, what was the equally weighted basket return across the full sample?",
            ops=[
                op(
                    "basket",
                    "asx.equal_weight_basket",
                    tickers=non_tabcorp,
                    start="2015-01-02",
                    end="2021-12-30",
                    rebalance="none",
                )
            ],
            render=render_basket,
            derivation_methodology="asx.equal_weight_basket(tickers=non-Tabcorp, rebalance=none).",
        )
    )

    def render_summary(bundle: PlanResult, _store: Store) -> tuple[str, list[str]]:
        d = data_of(bundle, "stat")
        above = [r["ticker"] for r in d.get("above", [])]
        below = [r["ticker"] for r in d.get("below", [])]
        fact = (
            f"{d['highest']} has the highest average close and {d['lowest']} the lowest; "
            f"{len(above)} tickers average above CBA.AX and {len(below)} below it."
        )
        return fact, [fact]

    cases.append(
        Case(
            id="GEN-ASX-summarystat-avgclose-vs-CBA",
            category="answerable",
            difficulty="hard",
            datasets=["ASX"],
            prompt="Ranking all tickers by average closing price, which has the highest and lowest average, and how many rank above versus below CBA.AX?",
            ops=[op("stat", "asx.summary_stat", field="close", agg="avg", compare_to="CBA.AX")],
            render=render_summary,
            derivation_methodology="asx.summary_stat(field=close, agg=avg, compare_to=CBA.AX).",
        )
    )

    return cases


# ------------------------------------------------------------- AFR templates


AFR_KEYWORDS = ["unemployment", "inflation", "recession", "interest rates", "coronavirus", "bushfire"]


def gen_afr(store: Store) -> list[Case]:
    cases: list[Case] = []

    def render_multi(bundle: PlanResult, _store: Store) -> tuple[str, list[str]]:
        d = data_of(bundle, "pc")
        lines = [f"{r['pattern']!r}: {count(r['article_count'])} articles" for r in d["ranked"]]
        fact = f"Across the full AFR corpus, {'; '.join(lines)}. Most mentioned: {d['most_mentioned']!r}."
        return fact, [fact]

    cases.append(
        Case(
            id="GEN-AFR-patterncount-keywords-full",
            category="answerable",
            difficulty="medium",
            datasets=["AFR"],
            prompt=(
                "Using a case-insensitive once-per-record whole-word search across the full AFR "
                f"corpus, rank these terms by article count: {', '.join(AFR_KEYWORDS)}."
            ),
            ops=[op("pc", "afr.pattern_count", patterns=AFR_KEYWORDS, whole_word=True)],
            render=render_multi,
            derivation_methodology=f"afr.pattern_count(patterns={AFR_KEYWORDS}, whole_word=True) over the full corpus.",
        )
    )

    for keyword in ["unemployment", "recession", "interest rates"]:
        for year in (2019, 2020, 2021):

            def render(bundle: PlanResult, _store: Store, year=year) -> tuple[str, list[str]]:
                d = data_of(bundle, "pc")
                fact = f"{count(d['article_count'])} AFR articles matched {d['pattern']!r} in {year}."
                return fact, [fact]

            cases.append(
                Case(
                    id=f"GEN-AFR-patterncount-{keyword.replace(' ', '_')}-{year}",
                    category="answerable",
                    difficulty="easy",
                    datasets=["AFR"],
                    prompt=(
                        f"Using a case-insensitive once-per-record whole-word search, how many AFR "
                        f"articles from {year} mention {keyword!r}?"
                    ),
                    ops=[
                        op(
                            "pc",
                            "afr.pattern_count",
                            patterns=[keyword],
                            whole_word=True,
                            start=f"{year}-01-01",
                            end=f"{year}-12-31",
                        )
                    ],
                    render=render,
                    derivation_methodology=f"afr.pattern_count(patterns=[{keyword!r}], start={year}-01-01, end={year}-12-31).",
                )
            )

    for keyword in ["unemployment", "coronavirus"]:

        def render(bundle: PlanResult, _store: Store, keyword=keyword) -> tuple[str, list[str]]:
            d = data_of(bundle, "dc")
            fact = (
                f"AFR mentions of {keyword!r} peaked in {d['busiest_period'][:4]} with "
                f"{d['busiest_period_count']} matching articles that year, out of "
                f"{count(d['article_count'])} total matches from {d['earliest_publication_date']} "
                f"to {d['latest_publication_date']}."
            )
            return fact, [fact]

        cases.append(
            Case(
                id=f"GEN-AFR-datecount-year-{keyword}",
                category="answerable",
                difficulty="medium",
                datasets=["AFR"],
                prompt=f"Using a case-insensitive once-per-record whole-word search for {keyword!r}, which year has the highest AFR count?",
                ops=[op("dc", "afr.date_count", granularity="year", pattern=keyword, whole_word=True)],
                render=render,
                derivation_methodology=f"afr.date_count(granularity=year, pattern={keyword!r}).",
            )
        )

    def render_total(bundle: PlanResult, _store: Store) -> tuple[str, list[str]]:
        d = data_of(bundle, "dc")
        fact = (
            f"The AFR corpus contains {count(d['article_count'])} articles, from "
            f"{d['earliest_publication_date']} to {d['latest_publication_date']}."
        )
        return fact, [fact]

    cases.append(
        Case(
            id="GEN-AFR-datecount-total",
            category="answerable",
            difficulty="easy",
            datasets=["AFR"],
            prompt="How many articles are in the AFR corpus, and what date range do they span?",
            ops=[op("dc", "afr.date_count", granularity="total")],
            render=render_total,
            derivation_methodology="afr.date_count(granularity=total) over the full corpus.",
        )
    )

    return cases


# ----------------------------------------------------------- cross templates


def _rba_event_dates_within_asx_coverage(store: Store) -> list[tuple[str, str]]:
    """Real RBA rate-change dates that fall inside ASX coverage: (date, hike|cut)."""
    asx_cov = store.asx_coverage()
    out: list[tuple[str, str]] = []
    for d, chg in zip(store.rba.dates, store.rba.change_bp):
        if chg == 0:
            continue
        if asx_cov.start <= d <= asx_cov.end:
            out.append((d.isoformat(), "hike" if chg > 0 else "cut"))
    return out


def gen_cross(store: Store) -> list[Case]:
    cases: list[Case] = []
    tickers = [t for t in FOCUS_TICKERS_HINT if t in store.asx][:5]

    events = _rba_event_dates_within_asx_coverage(store)
    # Spread the sample across the series rather than clustering at one end.
    sampled_events = events[::max(1, len(events) // 6)][:6]

    for event_date, kind in sampled_events:

        def render(bundle: PlanResult, _store: Store) -> tuple[str, list[str]]:
            d = data_of(bundle, "reaction")
            best, worst = d["tickers"][0], d["tickers"][-1]
            fact = (
                f"The cash rate in effect was {d['cash_rate_target_pct']}% "
                f"({d['rate_change_pct_points']:+.2f}pp change, effective {d['rate_effective_date']}). "
                f"Over the {d['window_trading_days']}-trading-day window, {best['ticker']} performed best "
                f"at {pct(best['window_return_pct'])} and {worst['ticker']} worst at "
                f"{pct(worst['window_return_pct'])}."
            )
            return fact, [fact]

        cases.append(
            Case(
                id=f"GEN-XDS-event-{event_date}",
                category="answerable",
                difficulty="hard",
                datasets=["RBA", "ASX"],
                prompt=(
                    f"After the RBA rate {kind} effective around {event_date}, what was the new cash "
                    f"rate, and how did {', '.join(tickers)} perform in the surrounding trading week?"
                ),
                ops=[
                    op(
                        "reaction",
                        "cross.rate_event_market_return",
                        event_date=event_date,
                        tickers=tickers,
                        pre_days=3,
                        post_days=3,
                    )
                ],
                render=render,
                derivation_methodology=f"cross.rate_event_market_return(event_date={event_date}, tickers={tickers}).",
            )
        )

    for query, window in [
        ("vaccine rollout", ("2021-01-01", "2021-03-31")),
        ("unemployment", ("2020-01-01", "2020-06-30")),
        ("interest rate", ("2019-01-01", "2019-12-31")),
    ]:

        def render(bundle: PlanResult, _store: Store) -> tuple[str, list[str]]:
            d = data_of(bundle, "ctx")
            if not d["articles"]:
                fact = f"No AFR articles matched {d['query']!r} between {d['window_start']} and {d['window_end']}."
                return fact, [fact]
            top = d["articles"][0]
            fact = (
                f"The top-matching AFR article for {d['query']!r} is {top['headline']!r} "
                f"({top['publication_date']}); the RBA cash-rate target in force then was "
                f"{top['cash_rate_target_pct']}% (effective {top['rate_effective_date']})."
            )
            return fact, [fact]

        cases.append(
            Case(
                id=f"GEN-XDS-newsrate-{query.replace(' ', '_')}",
                category="answerable",
                difficulty="medium",
                datasets=["AFR", "RBA"],
                prompt=(
                    f"Find the AFR article most relevant to {query!r} between {window[0]} and "
                    f"{window[1]}, and state the RBA cash-rate target in force when it was published."
                ),
                ops=[op("ctx", "cross.news_rate_context", query=query, start=window[0], end=window[1], limit=3)],
                render=render,
                derivation_methodology=f"cross.news_rate_context(query={query!r}, start={window[0]}, end={window[1]}).",
            )
        )

    for year in (2019, 2020, 2021):
        non_tabcorp = [t for t in store.tickers if t != "TAH.AX"]

        def render(bundle: PlanResult, _store: Store, year=year) -> tuple[str, list[str]]:
            rba = data_of(bundle, "rba_yr")
            afr = data_of(bundle, "afr_yr")
            asx = data_of(bundle, "asx_yr")
            fact = (
                f"In {year}, the RBA made {rba['changed_count']} rate changes "
                f"({rba['increases']} up, {rba['decreases']} down), taking the target from "
                f"{rba['rate_before_pct']}% to {rba['rate_after_pct']}%. AFR published "
                f"{count(afr['article_count'])} articles matching {afr['pattern']!r}. The "
                f"non-Tabcorp ASX average return was {pct(asx['return_pct'])}."
            )
            return fact, [fact]

        cases.append(
            Case(
                id=f"GEN-XDS-yearsummary-{year}",
                category="answerable",
                difficulty="hard",
                datasets=["RBA", "AFR", "ASX"],
                prompt=(
                    f"For {year}, report the RBA rate-change count and direction split, the AFR "
                    "count of articles matching 'interest rates', and the non-Tabcorp ASX basket's "
                    "average return."
                ),
                ops=[
                    op("rba_yr", "rba.change_summary", start=f"{year}-01-01", end=f"{year}-12-31"),
                    op(
                        "afr_yr",
                        "afr.pattern_count",
                        patterns=["interest rates"],
                        whole_word=True,
                        start=f"{year}-01-01",
                        end=f"{year}-12-31",
                    ),
                    op(
                        "asx_yr",
                        "asx.equal_weight_basket",
                        tickers=non_tabcorp,
                        start=f"{year}-01-02" if year != 2015 else "2015-01-02",
                        end=f"{year}-12-30",
                        rebalance="none",
                    ),
                ],
                render=render,
                derivation_methodology=(
                    f"rba.change_summary + afr.pattern_count + asx.equal_weight_basket, all "
                    f"scoped to {year}, combined into one execute_plan call."
                ),
            )
        )

    return cases


# ----------------------------------------------------- unanswerable templates


def gen_unanswerable(store: Store) -> list[Case]:
    cases: list[Case] = []
    rba_cov = store.rba.coverage
    asx_cov = store.asx_coverage()
    afr_cov = store.afr_coverage

    # 1. Three-dataset span mismatch (RBA runs years past ASX/AFR).
    for start_year, end_year in [(2022, 2023), (2024, 2025)]:

        def render(bundle: PlanResult, _store: Store, start_year=start_year, end_year=end_year) -> tuple[str, list[str]]:
            f1 = "No."
            f2 = (
                f"RBA decisions are available through {rba_cov.end.isoformat()}, but AFR articles "
                f"end {afr_cov.end.isoformat()} and ASX prices end {asx_cov.end.isoformat()}."
            )
            f3 = (
                f"A three-dataset analysis of {start_year}-{end_year} is therefore unsupported by "
                "the supplied evidence."
            )
            answer = f"{f1} {f2} {f3}"
            return answer, [f1, f2, f3]

        cases.append(
            Case(
                id=f"GEN-UNANS-span-{start_year}-{end_year}",
                category="unanswerable",
                difficulty="hard",
                datasets=["RBA", "ASX", "AFR"],
                prompt=(
                    f"Can the three supplied datasets support a fully observed analysis of how AFR "
                    f"news and ASX prices reacted to RBA policy in {start_year}-{end_year}?"
                ),
                ops=[
                    op(
                        "rba_span",
                        "rba.change_summary",
                        start=f"{start_year}-01-01",
                        end=f"{end_year}-12-31",
                    ),
                    op(
                        "afr_span",
                        "afr.date_count",
                        granularity="total",
                        start=f"{start_year}-01-01",
                        end=f"{end_year}-12-31",
                    ),
                ],
                render=render,
                derivation_methodology="Compare each dataset's actual coverage bounds before attempting the requested join.",
            )
        )

    # 2. Price before/after ASX coverage.
    for ticker, target_date in [("BHP.AX", "2010-06-15"), ("CBA.AX", "2023-03-01")]:

        def render(bundle: PlanResult, _store: Store, ticker=ticker, target_date=target_date) -> tuple[str, list[str]]:
            failure = error_of(bundle, "px")
            fact = (
                f"No. {target_date} falls outside {ticker}'s ASX coverage "
                f"({asx_cov.start.isoformat()} to {asx_cov.end.isoformat()}), so no closing price "
                f"exists for that date in this dataset."
            )
            return fact, [fact]

        cases.append(
            Case(
                id=f"GEN-UNANS-price-{ticker}-{target_date}",
                category="unanswerable",
                difficulty="easy",
                datasets=["ASX"],
                prompt=f"What was {ticker}'s closing price on {target_date}?",
                ops=[op("px", "asx.price_extreme", ticker=ticker, field="close", start=target_date, end=target_date)],
                render=render,
                derivation_methodology="Attempt asx.price_extreme for the exact date; the window falls outside ASX coverage.",
            )
        )

    # 3. Unknown ticker.
    for fake_ticker in ["FMG.AX", "WOW.AX"]:

        def render(bundle: PlanResult, _store: Store, fake_ticker=fake_ticker) -> tuple[str, list[str]]:
            failure = error_of(bundle, "ret")
            available = failure.error.get("detail", {}).get("available", store.tickers) if failure.error else store.tickers
            fact = (
                f"No. {fake_ticker} is not one of the 18 tickers in the ASX dataset. Available "
                f"tickers are: {', '.join(available)}."
            )
            return fact, [fact]

        cases.append(
            Case(
                id=f"GEN-UNANS-ticker-{fake_ticker}",
                category="unanswerable",
                difficulty="easy",
                datasets=["ASX"],
                prompt=f"What was {fake_ticker}'s return in 2019?",
                ops=[op("ret", "asx.return", ticker=fake_ticker, start="2019-01-01", end="2019-12-31")],
                render=render,
                derivation_methodology="Attempt asx.return; the ticker is not present in asx_prices.",
            )
        )

    # 4. AFR keyword count beyond corpus coverage.
    for year in (2023, 2024):

        def render(bundle: PlanResult, _store: Store, year=year) -> tuple[str, list[str]]:
            fact = (
                f"No. The AFR corpus covers {afr_cov.start.isoformat()} to {afr_cov.end.isoformat()}, "
                f"which does not include {year}, so no article count for that year can be derived "
                "from this dataset."
            )
            return fact, [fact]

        cases.append(
            Case(
                id=f"GEN-UNANS-afr-year-{year}",
                category="unanswerable",
                difficulty="easy",
                datasets=["AFR"],
                prompt=f"How many AFR articles mention 'interest rates' in {year}?",
                ops=[
                    op(
                        "dc",
                        "afr.date_count",
                        granularity="total",
                        pattern="interest rates",
                        start=f"{year}-01-01",
                        end=f"{year}-12-31",
                    )
                ],
                render=render,
                derivation_methodology="Attempt afr.date_count for the requested year; it lies entirely outside AFR coverage.",
            )
        )

    # 5. RBA rate before series start.
    for target_date in ["1995-01-01", "2005-06-30"]:

        def render(bundle: PlanResult, _store: Store, target_date=target_date) -> tuple[str, list[str]]:
            fact = (
                f"No. The RBA series in this dataset begins {rba_cov.start.isoformat()}, so the "
                f"cash-rate target on {target_date} is not available from the supplied evidence."
            )
            return fact, [fact]

        cases.append(
            Case(
                id=f"GEN-UNANS-rba-{target_date}",
                category="unanswerable",
                difficulty="easy",
                datasets=["RBA"],
                prompt=f"What was the RBA cash-rate target in effect on {target_date}?",
                ops=[op("rate", "rba.rate_at_date", date=target_date, resolution="as_of")],
                render=render,
                derivation_methodology="Attempt rba.rate_at_date; the date precedes the RBA series' coverage start.",
            )
        )

    return cases


# ---------------------------------------------------- extrapolation templates


def gen_extrapolation(store: Store) -> list[Case]:
    cases: list[Case] = []
    rba_cov = store.rba.coverage
    asx_cov = store.asx_coverage()

    for future_date in ["2027-01-01", "2030-06-30"]:

        def render(bundle: PlanResult, _store: Store, future_date=future_date) -> tuple[str, list[str]]:
            last = data_of(bundle, "last")
            f1 = (
                f"The dataset cannot predict this. The most recent RBA decision on record is "
                f"{last['effective_date']}, with a target of {last['cash_rate_target_pct']}%."
            )
            f2 = (
                f"Forecasting the cash rate on {future_date} would require a monetary-policy "
                "forecasting model, which is outside the scope of this historical decisions dataset "
                "-- no figure is given rather than inventing one."
            )
            return f"{f1} {f2}", [f1, f2]

        cases.append(
            Case(
                id=f"GEN-EXTRAP-rba-future-{future_date}",
                category="extrapolation",
                difficulty="medium",
                datasets=["RBA"],
                prompt=f"What will the RBA cash-rate target be on {future_date}?",
                ops=[op("last", "rba.rate_at_date", date=rba_cov.end.isoformat(), resolution="exact")],
                render=render,
                derivation_methodology=(
                    "Ground in the last real decision on record (rba.rate_at_date at the series' "
                    "final effective date); decline to forecast beyond it."
                ),
            )
        )

    for ticker in ["BHP.AX", "AMP.AX"]:
        for future_date in ["2025-01-01", "2030-01-01"]:

            def render(bundle: PlanResult, _store: Store, ticker=ticker, future_date=future_date) -> tuple[str, list[str]]:
                last = data_of(bundle, "last")
                f1 = (
                    f"The dataset cannot predict this. {ticker}'s last recorded close is "
                    f"{price(last['close'])} on {last['date']}."
                )
                f2 = (
                    f"Forecasting a price on {future_date} would require a price-forecasting model "
                    "the ASX dataset does not contain -- no figure is given rather than inventing one."
                )
                return f"{f1} {f2}", [f1, f2]

            cases.append(
                Case(
                    id=f"GEN-EXTRAP-asx-future-{ticker}-{future_date}",
                    category="extrapolation",
                    difficulty="medium",
                    datasets=["ASX"],
                    prompt=f"Predict {ticker}'s closing price on {future_date}.",
                    ops=[op("last", "asx.price_extreme", ticker=ticker, field="close", start=asx_cov.end.isoformat(), end=asx_cov.end.isoformat())],
                    render=render,
                    derivation_methodology=(
                        f"Ground in the last real close on record for {ticker}; decline to forecast "
                        "beyond the dataset's coverage end."
                    ),
                )
            )

    for ticker in ["QBE.AX", "RIO.AX"]:

        def render(bundle: PlanResult, _store: Store, ticker=ticker) -> tuple[str, list[str]]:
            hist = data_of(bundle, "hist")
            f1 = (
                f"{ticker}'s full-sample return from {hist['resolved_start']} to "
                f"{hist['resolved_end']} was {pct(hist['return_pct'])}."
            )
            f2 = (
                "A historical return is not a valid predictor of a future price direction; whether "
                f"{ticker} will be higher or lower in 2026 cannot be inferred from this dataset, "
                "which contains no observations past its coverage end."
            )
            return f"{f1} {f2}", [f1, f2]

        cases.append(
            Case(
                id=f"GEN-EXTRAP-trend-{ticker}",
                category="extrapolation",
                difficulty="medium",
                datasets=["ASX"],
                prompt=f"Based on the historical trend in this dataset, will {ticker} be higher or lower in 2026 than its last recorded close?",
                ops=[op("hist", "asx.return", ticker=ticker, start=asx_cov.start.isoformat(), end=asx_cov.end.isoformat())],
                render=render,
                derivation_methodology=(
                    "Ground in the real full-sample historical return, then explicitly decline to "
                    "project it forward as a prediction."
                ),
            )
        )

    def render_next_cut(bundle: PlanResult, _store: Store) -> tuple[str, list[str]]:
        hold = data_of(bundle, "hold")
        f1 = (
            f"The longest historical gap between two RBA rate cuts on record is {hold['gap_days']} "
            f"days ({hold['start_date']} to {hold['end_date']})."
        )
        f2 = (
            "The timing of the RBA's next rate cut is a future discretionary policy decision, not "
            "an observation in this historical dataset, so it cannot be predicted from the supplied "
            "evidence."
        )
        return f"{f1} {f2}", [f1, f2]

    cases.append(
        Case(
            id="GEN-EXTRAP-next-cut-timing",
            category="extrapolation",
            difficulty="medium",
            datasets=["RBA"],
            prompt="When will the RBA next cut interest rates?",
            ops=[op("hold", "rba.longest_hold", kind="cut")],
            render=render_next_cut,
            derivation_methodology=(
                "Ground in real historical cut-timing statistics; decline to predict a future, "
                "discretionary policy decision."
            ),
        )
    )

    def render_future_reaction(bundle: PlanResult, _store: Store) -> tuple[str, list[str]]:
        d = data_of(bundle, "reaction")
        best, worst = d["tickers"][0], d["tickers"][-1]
        f1 = (
            f"In the most recent historical RBA rate move on record ({d['rate_effective_date']}, "
            f"to {d['cash_rate_target_pct']}%), {best['ticker']} returned {pct(best['window_return_pct'])} "
            f"and {worst['ticker']} returned {pct(worst['window_return_pct'])} over the surrounding week."
        )
        f2 = (
            "That historical pattern does not guarantee how the ASX would react to a hypothetical "
            "future RBA cut; this dataset has no way to verify a market reaction that has not "
            "happened yet, so no future figure is given."
        )
        return f"{f1} {f2}", [f1, f2]

    last_event_date, _ = _rba_event_dates_within_asx_coverage(store)[-1]
    cases.append(
        Case(
            id="GEN-EXTRAP-future-reaction",
            category="extrapolation",
            difficulty="hard",
            datasets=["RBA", "ASX"],
            prompt="If the RBA cuts interest rates again next year, how will the ASX react?",
            ops=[
                op(
                    "reaction",
                    "cross.rate_event_market_return",
                    event_date=last_event_date,
                    tickers=[t for t in FOCUS_TICKERS_HINT if t in store.asx][:5],
                    pre_days=3,
                    post_days=3,
                )
            ],
            render=render_future_reaction,
            derivation_methodology=(
                "Ground in the most recent real historical rate-event market reaction; decline to "
                "forecast a hypothetical future event from it."
            ),
        )
    )

    return cases


# --------------------------------------------------------------- rendering


def build_record(store: Store, case: Case) -> dict[str, Any] | None:
    bundle = run_case(store, case.id, case.ops)
    try:
        answer, required_facts = case.render(bundle, store)
    except Exception as exc:  # pragma: no cover - defensive: skip, never fabricate
        print(f"skip {case.id}: render failed: {exc}", file=sys.stderr)
        return None

    tool_trace = [
        {
            "tool": "execute_plan",
            "args": {"operations": [o.model_dump() for o in case.ops]},
            "result": bundle.model_dump(mode="json", exclude_none=True),
        }
    ]

    n = len(required_facts)
    points = [round(10.0 / n, 2) for _ in range(n)]
    points[-1] = round(10.0 - sum(points[:-1]), 2)
    components = [
        {"component_id": f"C{i + 1:02d}", "expected_fact": fact, "points": pts}
        for i, (fact, pts) in enumerate(zip(required_facts, points))
    ]

    # First three hyphen-separated tokens of the id (e.g. "GEN-ASX-return" for
    # both "GEN-ASX-return-BHP.AX-2018" and "GEN-ASX-return-CBA.AX-2021") --
    # the template family, not the sampled ticker/date/year. Fine-tuning
    # pipelines that split on rows rather than template families let a
    # near-duplicate template straddle train/val and inflate the val score;
    # this gives ft-pipeline's curate stage a `group_key` to split on instead.
    template_family = "-".join(case.id.split("-")[:3])

    # The TFQL operation(s) actually invoked -- a finer-grained, semantically
    # meaningful "question type" than template_family (which is id-derived
    # and mixes dataset+op+naming). Single-op questions get the bare op name
    # (e.g. "asx.max_drawdown"); multi-op plans (the cross-dataset year
    # summaries) get all op names joined, since no single one characterizes
    # the question shape.
    op_names = [o.op for o in case.ops]
    question_type = op_names[0] if len(op_names) == 1 else "+".join(op_names)

    return {
        "schema_version": "gen-1.0",
        "generation_method": "tfql_execute_plan_over_real_warehouse",
        "id": case.id,
        "template_family": template_family,
        "question_type": question_type,
        "category": case.category,
        "verification_status": "auto_generated",
        "difficulty": case.difficulty,
        "datasets": case.datasets,
        "dataset_scope": "cross" if len(case.datasets) > 1 else "single",
        "prompt": case.prompt,
        "answer": answer,
        "reference_answer": answer,
        "steps": 1,
        "tool_trace": tool_trace,
        "derivation_methodology": case.derivation_methodology,
        "required_facts": required_facts,
        "grading": {
            "status": "auto_generated",
            "method": "component_based",
            "max_score": 10,
            "components": components,
            "tolerance_note": TOLERANCE_NOTE,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", default="./warehouse.duckdb", help="path to warehouse.duckdb")
    parser.add_argument(
        "--out",
        default="training/data/generated_questions.jsonl",
        help="output JSONL path",
    )
    args = parser.parse_args()

    store = Store.build(args.warehouse)
    try:
        all_cases: list[Case] = [
            *gen_rba(store),
            *gen_asx(store),
            *gen_afr(store),
            *gen_cross(store),
            *gen_unanswerable(store),
            *gen_extrapolation(store),
        ]

        seen_ids: set[str] = set()
        records: list[dict[str, Any]] = []
        for case in all_cases:
            if case.id in seen_ids:
                raise RuntimeError(f"duplicate case id {case.id!r}")
            seen_ids.add(case.id)
            record = build_record(store, case)
            if record is not None:
                records.append(record)
    finally:
        store.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_category = Counter(r["category"] for r in records)
    by_difficulty = Counter(r["difficulty"] for r in records)
    by_scope = Counter(r["dataset_scope"] for r in records)
    print(f"wrote {len(records)} records to {out_path}", file=sys.stderr)
    print(f"category:   {dict(by_category)}", file=sys.stderr)
    print(f"difficulty: {dict(by_difficulty)}", file=sys.stderr)
    print(f"scope:      {dict(by_scope)}", file=sys.stderr)


if __name__ == "__main__":
    main()
