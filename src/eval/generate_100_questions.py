"""One-off generator: build a 100-question evaluation dataset for the
finance hackathon agent, with every answer/grading_component computed
directly from the real RBA/ASX/AFR data (same math as finance_tools.py,
so the generated ground truth always matches what the live tools return).

Writes ``AI_Industry_Training_Hackathon/data set/evals/questions.json`` in
the same flat schema as ``mock_questions.json`` (difficulty/question/answer/
grading_components), plus ``tools``/``tool_args`` for reference, so it's
directly consumable by ``llm_judge_grader.py --questions ...``.

Run once to (re)generate the file:

    .venv/bin/python evals-hackathon/generate_100_questions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from finance_data import (  # noqa: E402
    combined_text,
    in_range,
    list_asx_tickers,
    load_afr,
    load_asx,
    load_rba,
)

OUT_PATH = REPO_ROOT / "AI_Industry_Training_Hackathon" / "data set" / "evals" / "questions.json"

import re
from datetime import date, datetime


def _pd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


rows: list[dict[str, Any]] = []


def add(
    qid: str,
    difficulty: str,
    question: str,
    answer: str,
    components: list[str],
    tools: list[str],
    tool_args: list[dict[str, Any]],
) -> None:
    rows.append({
        "id": qid,
        "difficulty": difficulty,
        "question": question,
        "answer": answer,
        "grading_components": components,
        "tools": tools,
        "tool_args": tool_args,
    })


# ===========================================================================
# RBA (23 questions)
# ===========================================================================
rba_rows = load_rba()

for mode in ("lowest", "highest"):
    target = min(r["rate"] for r in rba_rows) if mode == "lowest" else max(r["rate"] for r in rba_rows)
    matching = [r for r in rba_rows if r["rate"] == target]
    first_date = min(r["date"] for r in matching)
    add(
        f"rba-{mode}-rate",
        "easy",
        f"What is the {mode} cash-rate target in the RBA dataset, when did it first "
        "take effect, and how many decision records show that rate?",
        f"The {mode} cash-rate target in the dataset is {target}%, which first took "
        f"effect on {first_date.isoformat()}, across {len(matching)} decision "
        "record(s) at that rate.",
        [str(target), first_date.isoformat(), str(len(matching))],
        ["rba_extreme_rate"],
        [{"mode": mode}],
    )

nonzero = [r for r in rba_rows if r["change"] != 0]
best_gap = max(((b["date"] - a["date"]).days, a, b) for a, b in zip(nonzero, nonzero[1:]))
days, before, after = best_gap
add(
    "rba-longest-gap", "medium",
    "What was the longest stretch between two non-zero RBA rate changes in this dataset?",
    f"The longest stretch between two non-zero rate changes was {days} days, from "
    f"{before['date'].isoformat()} to {after['date'].isoformat()}, during which the "
    f"rate held at {before['rate']}% before changing to {after['rate']}%.",
    [str(days), before["date"].isoformat(), after["date"].isoformat(), str(before["rate"]), str(after["rate"])],
    ["rba_longest_gap"], [{}],
)

for direction, wants_positive, label in (("hikes", True, "hike"), ("cuts", False, "cut")):
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
    start, end = best_cycle[0], best_cycle[-1]
    cumulative = round(sum(r["change"] for r in best_cycle), 2)
    before_idx = rba_rows.index(start)
    before_rate = rba_rows[before_idx - 1]["rate"] if before_idx > 0 else start["rate"]
    add(
        f"rba-largest-{direction}-cycle", "hard",
        f"Which {'tightening' if wants_positive else 'loosening'} cycle (consecutive "
        f"{label}s with no intervening {'cuts' if wants_positive else 'hikes'}) had the "
        "most " + label + "s, and what was the cumulative "
        + ("increase" if wants_positive else "decrease") + "?",
        f"The largest {direction} cycle ran from {start['date'].isoformat()} to "
        f"{end['date'].isoformat()} and comprised {len(best_cycle)} {label}(s), for a "
        f"cumulative change of {cumulative:+.2f} percentage points. The target rate "
        f"immediately before the first {label} was {before_rate}%, and the rate "
        f"reached by {end['date'].isoformat()} was {end['rate']}%.",
        [str(len(best_cycle)), start["date"].isoformat(), end["date"].isoformat(),
         f"{cumulative:+.2f}", str(before_rate), str(end["rate"])],
        ["rba_largest_cycle"], [{"direction": direction}],
    )

RATE_AT_DATES = [
    "2010-06-15", "2012-01-01", "2014-03-01", "2016-07-01", "2018-01-01",
    "2019-09-01", "2020-12-01", "2021-06-01", "2023-01-01", "2025-01-01",
]
for as_of in RATE_AT_DATES:
    target = _pd(as_of)
    candidates = [r for r in rba_rows if r["date"] <= target]
    latest = max(candidates, key=lambda r: r["date"])
    add(
        f"rba-rate-at-{as_of}", "easy",
        f"What was the RBA cash rate target in effect on {as_of}?",
        f"As of {as_of}, the RBA cash-rate target was {latest['rate']}%, set effective "
        f"{latest['date'].isoformat()}.",
        [as_of, str(latest["rate"]), latest["date"].isoformat()],
        ["rba_rate_at_date"], [{"as_of": as_of}],
    )

PERIODS = [
    ("2010-01-01", "2011-12-31"), ("2012-01-01", "2013-12-31"),
    ("2015-01-01", "2016-12-31"), ("2017-01-01", "2017-12-31"),
    ("2019-01-01", "2020-12-31"), ("2020-01-01", "2021-12-31"),
    ("2022-01-01", "2023-12-31"), ("2010-02-03", "2026-06-17"),
]
for start_s, end_s in PERIODS:
    start, end = _pd(start_s), _pd(end_s)
    in_period = [r for r in rba_rows if start <= r["date"] <= end and r["change"] != 0]
    inc = sum(1 for r in in_period if r["change"] > 0)
    dec = sum(1 for r in in_period if r["change"] < 0)
    net = round(sum(r["change"] for r in in_period), 2)
    add(
        f"rba-changes-{start_s}-{end_s}", "medium",
        f"Between {start_s} and {end_s}, how many times did the RBA change the cash "
        "rate, and how many were increases versus decreases?",
        f"Between {start_s} and {end_s}, the RBA changed the cash rate {len(in_period)} "
        f"time(s): {inc} increase(s) and {dec} decrease(s), for a net change of "
        f"{net:+.2f} percentage points.",
        [str(len(in_period)), str(inc), str(dec), f"{net:+.2f}"],
        ["rba_changes_in_period"], [{"start_date": start_s, "end_date": end_s}],
    )

# ===========================================================================
# ASX (50 questions)
# ===========================================================================
TICKERS = list_asx_tickers()  # 18 companies

for ticker in TICKERS:
    asx_rows = load_asx(ticker)
    hi = max(asx_rows, key=lambda r: r["close"])
    lo = min(asx_rows, key=lambda r: r["close"])
    add(
        f"asx-extremes-{ticker}", "easy",
        f"What were {ticker}'s highest and lowest closing prices in this dataset, "
        "and on what dates did they occur?",
        f"{ticker.upper()}'s highest close was ${hi['close']:.2f} on {hi['date']}, and "
        f"its lowest close was ${lo['close']:.2f} on {lo['date']} (data spans "
        f"{asx_rows[0]['date']} to {asx_rows[-1]['date']}).",
        [f"{hi['close']:.2f}", hi["date"], f"{lo['close']:.2f}", lo["date"]],
        ["asx_price_extremes"], [{"ticker": ticker}],
    )

for index, ticker in enumerate(TICKERS):
    direction = "decline" if index % 2 == 0 else "gain"
    asx_rows = load_asx(ticker)
    moves = [
        (
            (asx_rows[i]["close"] / asx_rows[i - 1]["close"] - 1) * 100,
            asx_rows[i]["date"], asx_rows[i - 1]["close"], asx_rows[i]["close"],
        )
        for i in range(1, len(asx_rows))
    ]
    best = min(moves) if direction == "decline" else max(moves)
    pct, when, prev_close, new_close = best
    add(
        f"asx-move-{ticker}", "medium",
        f"What was {ticker}'s largest single-day percentage {direction} in closing "
        f"price, and when did it happen?",
        f"{ticker.upper()}'s largest single-day {direction} was {pct:+.2f}%, on "
        f"{when}, when the close moved from ${prev_close:.2f} to ${new_close:.2f}.",
        [f"{pct:+.2f}%", when, f"{prev_close:.2f}", f"{new_close:.2f}"],
        ["asx_single_day_move"], [{"ticker": ticker, "direction": direction}],
    )

for direction in ("highest", "lowest"):
    ranked = sorted(
        ((t, (lambda r: (r[-1]["close"] / r[0]["close"] - 1) * 100)(load_asx(t))) for t in TICKERS),
        key=lambda item: item[1], reverse=(direction == "highest"),
    )
    name, total_return = ranked[0]
    label = "best" if direction == "highest" else "worst"
    add(
        f"asx-top-performer-{direction}", "hard",
        f"Which company was the {label}-performing in the ASX basket by total "
        "close-to-close return over the full dataset period, and what was its return?",
        f"The {label}-performing company in the ASX basket was {name}, with a total "
        f"close-to-close return of {total_return:+.2f}% over the full dataset period.",
        [name, f"{total_return:+.2f}%"],
        ["asx_top_performer"], [{"direction": direction}],
    )

RANK_TICKERS = ["BHP", "CBA", "AGL", "AMP", "NAB", "Rio"]
all_returns = sorted(
    ((t, (lambda r: (r[-1]["close"] / r[0]["close"] - 1) * 100)(load_asx(t))) for t in TICKERS),
    key=lambda item: item[1], reverse=True,
)
names_ranked = [t for t, _ in all_returns]
for ticker in RANK_TICKERS:
    idx = names_ranked.index(ticker)
    rank = idx + 1
    total_return = all_returns[idx][1]
    add(
        f"asx-rank-{ticker}", "hard",
        f"Where does {ticker} rank by total close-to-close return among the "
        f"{len(TICKERS)}-company ASX basket over the full dataset period, and what is "
        "its return?",
        f"{ticker.upper()} ranks {rank} of {len(TICKERS)} companies in the ASX basket "
        f"by total close-to-close return, with a return of {total_return:+.2f}% over "
        "the full dataset period.",
        [f"{rank} of {len(TICKERS)}", f"{total_return:+.2f}"],
        ["asx_rank_by_return"], [{"ticker": ticker}],
    )

RETURN_TICKERS = ["ANZ", "QBE", "IAG", "TPG", "GPT", "Suncorp"]
for ticker in RETURN_TICKERS:
    asx_rows = load_asx(ticker)
    first, last = asx_rows[0], asx_rows[-1]
    total_return = (last["close"] / first["close"] - 1) * 100
    add(
        f"asx-total-return-{ticker}", "medium",
        f"What is {ticker}'s total close-to-close return over the full dataset period?",
        f"{ticker.upper()}'s total return from {first['date']} to {last['date']} was "
        f"{total_return:+.2f}%, from ${first['close']:.2f} to ${last['close']:.2f}.",
        [f"{total_return:+.2f}%", f"{first['close']:.2f}", f"{last['close']:.2f}"],
        ["asx_total_return"], [{"ticker": ticker}],
    )

# ===========================================================================
# AFR (27 questions)
# ===========================================================================
afr_rows = load_afr()  # one-time ~10-15s parse of ~85 files

dates_all = [r["publication_date"] for r in afr_rows if r["publication_date"]]
add(
    "afr-corpus-stats", "easy",
    "How many articles are in the AFR dataset in total, and what date range do they span?",
    f"There are {len(afr_rows)} articles in total, spanning publication dates from "
    f"{min(dates_all).isoformat()} to {max(dates_all).isoformat()}.",
    [str(len(afr_rows)), min(dates_all).isoformat(), max(dates_all).isoformat()],
    ["afr_corpus_stats"], [{}],
)


def _busiest_day(start: date | None, end: date | None) -> tuple[date, int]:
    counts: dict[date, int] = {}
    for r in afr_rows:
        d = r["publication_date"]
        if d and in_range(d, start, end):
            counts[d] = counts.get(d, 0) + 1
    return max(counts.items(), key=lambda item: item[1])


busiest_date, count = _busiest_day(None, None)
add(
    "afr-busiest-day-overall", "medium",
    "Which single day had the most AFR articles published, and how many?",
    f"The busiest day was {busiest_date.isoformat()}, with {count} articles published.",
    [busiest_date.isoformat(), str(count)],
    ["afr_busiest_day"], [{}],
)

for year in range(2015, 2022):
    start, end = date(year, 1, 1), date(year, 12, 31)
    busiest_date, count = _busiest_day(start, end)
    add(
        f"afr-busiest-day-{year}", "medium",
        f"Which single day in {year} had the most AFR articles published, and how many?",
        f"The busiest day in {year} was {busiest_date.isoformat()}, with {count} "
        "articles published.",
        [busiest_date.isoformat(), str(count)],
        ["afr_busiest_day"], [{"start_date": start.isoformat(), "end_date": end.isoformat()}],
    )

KEYWORDS = [
    "bank", "NAB", "CBA", "ANZ", "Westpac", "RBA", "mining", "China",
    "iron ore", "tax", "budget", "election", "gigafactory",
]
for keyword in KEYWORDS:
    regex = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
    matched = sum(1 for r in afr_rows if regex.search(combined_text(r)))
    add(
        f"afr-pattern-{keyword.replace(' ', '-')}", "hard" if matched == 0 else "medium",
        f"How many AFR articles mention '{keyword}' (as a whole word) in their "
        "headline, subhead, intro, or body text?",
        f"{matched} articles mention '{keyword}' as a whole word across their "
        "headline, subhead, intro, or body text.",
        [str(matched)],
        ["afr_pattern_count"], [{"pattern": keyword}],
    )

longest = max(afr_rows, key=lambda r: len(str(r.get("HEADLINE") or "")))
headline = str(longest.get("HEADLINE") or "")
add(
    "afr-longest-headline-overall", "hard",
    "What is the longest article headline (by character count) in the entire AFR "
    "dataset, and how long is it?",
    f"The longest headline is {len(headline)} characters, published "
    f"{longest['publication_date'].isoformat()}: {headline[:80]!r}...",
    [str(len(headline)), longest["publication_date"].isoformat(), headline[:30]],
    ["afr_longest_headline"], [{}],
)

year = 2015
in_year = [r for r in afr_rows if r["publication_date"] and r["publication_date"].year == year]
longest_year = max(in_year, key=lambda r: len(str(r.get("HEADLINE") or "")))
headline_year = str(longest_year.get("HEADLINE") or "")
add(
    f"afr-longest-headline-{year}", "hard",
    f"What is the longest article headline (by character count) among AFR articles "
    f"published in {year}?",
    f"The longest headline in {year} is {len(headline_year)} characters, published "
    f"{longest_year['publication_date'].isoformat()}: {headline_year[:80]!r}...",
    [str(len(headline_year)), longest_year["publication_date"].isoformat()],
    ["afr_longest_headline"], [{"start_date": f"{year}-01-01", "end_date": f"{year}-12-31"}],
)

SAMPLE_DATES = ["2015-01-05", "2016-08-26", "2018-03-15", "2020-03-09"]
for as_of in SAMPLE_DATES:
    target = _pd(as_of)
    headlines = [str(r.get("HEADLINE") or "") for r in afr_rows if r["publication_date"] == target]
    add(
        f"afr-articles-on-{as_of}", "medium",
        f"How many AFR articles were published on {as_of}?",
        f"{len(headlines)} article(s) were published on {as_of}.",
        [str(len(headlines))],
        ["afr_articles_on_date"], [{"as_of": as_of}],
    )

print(f"RBA: {sum(1 for r in rows if r['id'].startswith('rba'))}")
print(f"ASX: {sum(1 for r in rows if r['id'].startswith('asx'))}")
print(f"AFR: {sum(1 for r in rows if r['id'].startswith('afr'))}")
print(f"TOTAL: {len(rows)}")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(rows)} questions to {OUT_PATH}")
