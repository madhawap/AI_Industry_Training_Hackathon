#!/usr/bin/env python3
"""
test_queries.py — Smoke-test the DuckDB warehouse produced by setup_duckdb.py.

Runs a battery of SQL queries against warehouse.duckdb and checks the results
against known-correct facts, independently re-derived from the RBA/ASX/AFR
source files under `data set/`. This catches regressions in the ingestion
pipeline — e.g. a bad date parse, a dropped row, or reintroduced data
corruption — not just "did the script run without crashing".

Also verifies the database supports concurrent reads (a project requirement:
at least 3 parallel requests).

Usage:
    python3 test_queries.py [db_path]

Default: db_path = ./warehouse.duckdb
Exit code: 0 if all tests pass, 1 otherwise.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

failures = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


def test_tables_exist(con):
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    check("tables exist", {"rba_rates", "asx_prices", "afr_articles"} <= tables, f"found {tables}")


def test_rba_shape(con):
    total, changes, lo, hi = con.execute("""
        SELECT count(*),
               count(*) FILTER (change_pct != 0),
               min(effective_date),
               max(effective_date)
        FROM rba_rates
    """).fetchone()
    check("rba_rates: 175 decision records", total == 175, f"got {total}")
    check("rba_rates: 41 rate changes", changes == 41, f"got {changes}")
    check("rba_rates: date range 2010-02-03 to 2026-06-17",
          str(lo) == "2010-02-03" and str(hi) == "2026-06-17", f"got {lo} to {hi}")


def test_rba_lowest_rate(con):
    rate, first_date, n = con.execute("""
        SELECT cash_rate_target, min(effective_date), count(*)
        FROM rba_rates
        WHERE cash_rate_target = (SELECT min(cash_rate_target) FROM rba_rates)
        GROUP BY cash_rate_target
    """).fetchone()
    check("rba lowest rate is 0.1%", rate == 0.1, f"got {rate}")
    check("rba lowest rate first effective 2020-11-04", str(first_date) == "2020-11-04", f"got {first_date}")
    check("rba lowest rate holds for 16 records", n == 16, f"got {n}")


def test_rba_longest_nonzero_gap(con):
    gap, d1, d2, rate_before, rate_after = con.execute("""
        WITH nonzero AS (
            SELECT effective_date, cash_rate_target,
                   lag(effective_date) OVER (ORDER BY effective_date) AS prev_date,
                   lag(cash_rate_target) OVER (ORDER BY effective_date) AS prev_rate
            FROM rba_rates
            WHERE change_pct != 0
        )
        SELECT date_diff('day', prev_date, effective_date) AS gap,
               prev_date, effective_date, prev_rate, cash_rate_target
        FROM nonzero
        ORDER BY gap DESC
        LIMIT 1
    """).fetchone()
    check("rba longest non-zero gap is 1036 days", gap == 1036, f"got {gap}")
    check("rba gap spans 2016-08-03 to 2019-06-05",
          str(d1) == "2016-08-03" and str(d2) == "2019-06-05", f"got {d1} to {d2}")
    check("rba rate held at 1.5 before changing to 1.25",
          rate_before == 1.5 and rate_after == 1.25, f"got {rate_before} -> {rate_after}")


def test_rba_tightening_cycle(con):
    n_hikes, cum, rate_before, rate_after = con.execute("""
        SELECT count(*), round(sum(change_pct), 2),
               (SELECT cash_rate_target FROM rba_rates WHERE effective_date < DATE '2022-05-04'
                ORDER BY effective_date DESC LIMIT 1),
               (SELECT cash_rate_target FROM rba_rates WHERE effective_date = DATE '2023-11-08')
        FROM rba_rates
        WHERE change_pct > 0
          AND effective_date BETWEEN DATE '2022-05-04' AND DATE '2023-11-08'
    """).fetchone()
    check("2022-2023 tightening cycle: 13 hikes", n_hikes == 13, f"got {n_hikes}")
    check("2022-2023 tightening cycle: +4.25pp cumulative", cum == 4.25, f"got {cum}")
    check("2022-2023 tightening cycle: 0.1 -> 4.35",
          rate_before == 0.1 and rate_after == 4.35, f"got {rate_before} -> {rate_after}")


def test_asx_bhp_high_low(con):
    hi, hi_date, lo, lo_date = con.execute("""
        SELECT
            (SELECT round(close, 2) FROM asx_prices WHERE ticker='BHP.AX' ORDER BY close DESC LIMIT 1),
            (SELECT date FROM asx_prices WHERE ticker='BHP.AX' ORDER BY close DESC LIMIT 1),
            (SELECT round(close, 2) FROM asx_prices WHERE ticker='BHP.AX' ORDER BY close ASC LIMIT 1),
            (SELECT date FROM asx_prices WHERE ticker='BHP.AX' ORDER BY close ASC LIMIT 1)
    """).fetchone()
    check("BHP highest close $33.72 on 2021-08-04", hi == 33.72 and str(hi_date) == "2021-08-04", f"got {hi} on {hi_date}")
    check("BHP lowest close $6.44 on 2016-01-21", lo == 6.44 and str(lo_date) == "2016-01-21", f"got {lo} on {lo_date}")


def test_asx_bhp_biggest_decline(con):
    pct, d, prev_close, close = con.execute("""
        WITH chg AS (
            SELECT date, close,
                   lag(close) OVER (ORDER BY date) AS prev_close
            FROM asx_prices WHERE ticker = 'BHP.AX'
        )
        SELECT round((close - prev_close) / prev_close * 100, 2), date, round(prev_close, 2), round(close, 2)
        FROM chg
        ORDER BY (close - prev_close) / prev_close
        LIMIT 1
    """).fetchone()
    check("BHP biggest single-day decline -14.41% on 2020-03-09",
          pct == -14.41 and str(d) == "2020-03-09", f"got {pct}% on {d}")
    check("BHP decline was $19.07 -> $16.32", prev_close == 19.07 and close == 16.32,
          f"got {prev_close} -> {close}")


def test_asx_volume_integrity(con):
    n_tickers, n_null_volume = con.execute("""
        SELECT count(DISTINCT ticker), count(*) FILTER (volume IS NULL) FROM asx_prices
    """).fetchone()
    check("asx_prices covers 18 tickers", n_tickers == 18, f"got {n_tickers}")
    check("asx_prices has no unparseable volume values", n_null_volume == 0, f"got {n_null_volume} nulls")


def test_afr_shape(con):
    total, lo, hi = con.execute("""
        SELECT count(*), min(publication_date), max(publication_date) FROM afr_articles
    """).fetchone()
    check("afr_articles: 219446 articles", total == 219446, f"got {total}")
    check("afr_articles: date range 2015-01-02 to 2021-12-29",
          str(lo) == "2015-01-02" and str(hi) == "2021-12-29", f"got {lo} to {hi}")


def test_afr_busiest_day(con):
    d, n = con.execute("""
        SELECT publication_date, count(*) AS n
        FROM afr_articles
        GROUP BY publication_date
        ORDER BY n DESC
        LIMIT 1
    """).fetchone()
    check("afr busiest day is 2016-08-26 with 314 articles", str(d) == "2016-08-26" and n == 314, f"got {d} with {n}")


def test_afr_bank_and_longest_headline(con):
    n_bank = con.execute("""
        SELECT count(*) FROM afr_articles
        WHERE headline ILIKE '%bank%' OR text ILIKE '%bank%'
    """).fetchone()[0]
    headline, length = con.execute("""
        SELECT headline, length(headline) AS len
        FROM afr_articles
        ORDER BY len DESC
        LIMIT 1
    """).fetchone()
    check("66105 afr articles mention 'bank'", n_bank == 66105, f"got {n_bank}")
    check("longest afr headline is 489 chars",
          length == 489 and headline.startswith("MINING & OILto close of business"),
          f"got {length} chars: {headline!r}")


def test_afr_fts_index(con):
    schema_exists = con.execute("""
        SELECT count(*) FROM information_schema.schemata WHERE schema_name = 'fts_main_afr_articles'
    """).fetchone()[0]
    check("fts_main_afr_articles schema exists", schema_exists == 1, f"got {schema_exists}")
    if not schema_exists:
        return

    con.execute("LOAD fts")
    rows = con.execute("""
        WITH scored AS (
            SELECT headline, fts_main_afr_articles.match_bm25(article_id, 'bank') AS score
            FROM afr_articles
        )
        SELECT * FROM scored WHERE score IS NOT NULL ORDER BY score DESC LIMIT 1
    """).fetchall()
    check("BM25 search for 'bank' returns ranked results", len(rows) == 1, f"got {rows}")


def test_concurrent_reads(db_path: Path, n: int = 3):
    def run_query(i):
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            return con.execute("SELECT count(*) FROM rba_rates").fetchone()[0]
        finally:
            con.close()

    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(run_query, range(n)))

    check(f"{n} concurrent read-only connections all succeeded",
          all(r == 175 for r in results), f"got {results}")


def main():
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./warehouse.duckdb")
    if not db_path.exists():
        print(f"Error: {db_path} not found. Run setup_duckdb.py first.", file=sys.stderr)
        sys.exit(1)

    con = duckdb.connect(str(db_path), read_only=True)
    test_tables_exist(con)
    test_rba_shape(con)
    test_rba_lowest_rate(con)
    test_rba_longest_nonzero_gap(con)
    test_rba_tightening_cycle(con)
    test_asx_bhp_high_low(con)
    test_asx_bhp_biggest_decline(con)
    test_asx_volume_integrity(con)
    test_afr_shape(con)
    test_afr_busiest_day(con)
    test_afr_bank_and_longest_headline(con)
    test_afr_fts_index(con)
    con.close()

    test_concurrent_reads(db_path, n=3)

    print(f"\n{len(failures)} failure(s)" if failures else "\nAll checks passed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
