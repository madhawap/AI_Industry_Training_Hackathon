#!/usr/bin/env python3
"""
setup_duckdb.py — Repeatable ingestion of the RBA/ASX/AFR datasets into DuckDB.

Pipeline (per dataset): normalized/*  ->  Parquet (Hive-partitioned)  ->  DuckDB tables.

Parquet is the intermediate storage format because it's what lets DuckDB stay
fast as the file counts grow (200k AFR files, N ASX company files): instead of
re-scanning thousands of small CSV/JSONL files on every query, DuckDB scans a
much smaller number of column-oriented, partition-pruned Parquet files.

Safe to re-run: every step uses CREATE OR REPLACE / OVERWRITE_OR_IGNORE, so
re-running after new source files land simply rebuilds everything from
normalized/ — no manual cleanup required.

Also builds a DuckDB FTS (full-text search) index on afr_articles
(headline/subhead/text), rebuilt with overwrite=1 every run since the table
itself is rebuilt every run. Query it via fts_main_afr_articles.match_bm25(...)
or, more conveniently, query_tools.full_text_search().

Usage:
    python3 setup_duckdb.py [source_dir] [parquet_dir] [db_path]

Defaults:
    source_dir  = ./normalized   (output of normalize_dates.py)
    parquet_dir = ./parquet
    db_path     = ./warehouse.duckdb
"""

import sys
from pathlib import Path

import duckdb


def glob(base: Path, pattern: str) -> str:
    return str((base / pattern).as_posix())


def ingest_rba(con: duckdb.DuckDBPyConnection, source_dir: Path, parquet_dir: Path) -> None:
    csv_glob = glob(source_dir, "RBA Rates/*.csv")
    out_dir = parquet_dir / "rba"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "rba_rates.parquet"

    con.execute(f"""
        COPY (
            SELECT
                CAST("Effective Date" AS DATE)      AS effective_date,
                CAST("Change % points" AS DOUBLE)   AS change_pct,
                CAST("Cash rate target%" AS DOUBLE) AS cash_rate_target
            FROM read_csv_auto('{csv_glob}', header = true)
        ) TO '{out_file.as_posix()}' (FORMAT PARQUET);
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE rba_rates AS
        SELECT * FROM read_parquet('{out_file.as_posix()}');
    """)
    n = con.execute("SELECT count(*) FROM rba_rates").fetchone()[0]
    print(f"rba_rates: {n} rows")


def ingest_asx(con: duckdb.DuckDBPyConnection, source_dir: Path, parquet_dir: Path) -> None:
    jsonl_glob = glob(source_dir, "ASX/*.jsonl")
    out_dir = parquet_dir / "asx"
    out_dir.mkdir(parents=True, exist_ok=True)

    # volume is TRY_CAST rather than CAST: one known source row has a corrupted
    # (non-numeric) volume value from a merged CSV line upstream; TRY_CAST turns
    # that single bad value into NULL instead of failing the whole ingest or
    # forcing the column to VARCHAR for every row.
    con.execute(f"""
        COPY (
            SELECT
                CAST(date AS DATE)        AS date,
                CAST(open AS DOUBLE)      AS open,
                CAST(high AS DOUBLE)      AS high,
                CAST(low AS DOUBLE)       AS low,
                CAST(close AS DOUBLE)     AS close,
                TRY_CAST(volume AS BIGINT) AS volume,
                ticker
            FROM read_json_auto('{jsonl_glob}', union_by_name = true)
        ) TO '{out_dir.as_posix()}' (FORMAT PARQUET, PARTITION_BY (ticker), OVERWRITE_OR_IGNORE true);
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE asx_prices AS
        SELECT * FROM read_parquet('{glob(out_dir, "*/*.parquet")}', hive_partitioning = true);
    """)
    n_rows, n_null_vol, n_tickers = con.execute("""
        SELECT count(*), count(*) FILTER (volume IS NULL), count(DISTINCT ticker)
        FROM asx_prices
    """).fetchone()
    print(f"asx_prices: {n_rows} rows, {n_tickers} tickers, {n_null_vol} rows with unparseable volume")


def ingest_afr(con: duckdb.DuckDBPyConnection, source_dir: Path, parquet_dir: Path) -> None:
    jsonl_glob = glob(source_dir, "AFR/*.jsonl")
    out_dir = parquet_dir / "afr"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Source records use upper-case, no-space keys (HEADLINE, PUBLICATIONDATE, ...).
    # DuckDB identifier matching is case-insensitive but not underscore-insensitive,
    # so publicationdate (source) must be referenced without the underscore that
    # the output column name uses.
    #
    # A small number of source articles (92 of 219,538, across 4 of the 85 files)
    # carry an empty PUBLICATIONDATE. TRY_CAST turns those into NULL and the WHERE
    # drops them, rather than failing the whole ingest — this table is partitioned
    # and ordered by date, so a row with no date has nowhere valid to go.
    dropped = con.execute(f"""
        SELECT count(*) FROM read_json_auto('{jsonl_glob}', union_by_name = true)
        WHERE TRY_CAST(publicationdate AS DATE) IS NULL
    """).fetchone()[0]
    if dropped:
        print(f"afr_articles: dropping {dropped} rows with unparseable/empty publication date")

    # Explicit "AS <lowercase>" aliases are required, not cosmetic: DuckDB resolves
    # unquoted identifiers case-insensitively but the output column keeps the
    # source's original case (HEADLINE, SUBHEAD, ...) unless re-aliased, which
    # would otherwise leave afr_articles the only table with upper-case columns.
    con.execute(f"""
        COPY (
            SELECT
                headline AS headline,
                subhead AS subhead,
                newspaper AS newspaper,
                TRY_CAST(publicationdate AS DATE) AS publication_date,
                text AS text,
                intro AS intro,
                year(TRY_CAST(publicationdate AS DATE))  AS year,
                month(TRY_CAST(publicationdate AS DATE)) AS month
            FROM read_json_auto('{jsonl_glob}', union_by_name = true)
            WHERE TRY_CAST(publicationdate AS DATE) IS NOT NULL
        ) TO '{out_dir.as_posix()}' (FORMAT PARQUET, PARTITION_BY (year, month), OVERWRITE_OR_IGNORE true);
    """)

    # article_id is added here (not in the Parquet layer) purely as a stable
    # row identifier for the FTS index below, which requires one.
    con.execute(f"""
        CREATE OR REPLACE TABLE afr_articles AS
        SELECT row_number() OVER (ORDER BY publication_date, headline) AS article_id, *
        FROM read_parquet('{glob(out_dir, "*/*/*.parquet")}', hive_partitioning = true);
    """)
    n = con.execute("SELECT count(*) FROM afr_articles").fetchone()[0]
    print(f"afr_articles: {n} rows")

    con.execute("INSTALL fts; LOAD fts;")
    con.execute("""
        PRAGMA create_fts_index('afr_articles', 'article_id', 'headline', 'subhead', 'text', overwrite=1);
    """)
    print("afr_articles: FTS index built (fts_main_afr_articles) on headline/subhead/text")


def verify_concurrent_reads(db_path: Path, n: int = 3) -> None:
    """Open n simultaneous read-only connections and run a query on each,
    to confirm the database supports the required parallel-read access."""
    conns = [duckdb.connect(str(db_path), read_only=True) for _ in range(n)]
    try:
        for i, c in enumerate(conns):
            result = c.execute("SELECT count(*) FROM rba_rates").fetchone()[0]
            print(f"  connection {i + 1}/{n}: OK ({result} rows visible)")
    finally:
        for c in conns:
            c.close()


def main():
    args = sys.argv[1:]
    source_dir = Path(args[0]) if len(args) > 0 else Path("./normalized")
    parquet_dir = Path(args[1]) if len(args) > 1 else Path("./parquet")
    db_path = Path(args[2]) if len(args) > 2 else Path("./warehouse.duckdb")

    if not source_dir.is_dir():
        print(f"Error: source directory not found: {source_dir}", file=sys.stderr)
        print("Run normalize_dates.py first to produce it.", file=sys.stderr)
        sys.exit(1)

    parquet_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    print(f"Ingesting {source_dir} -> {parquet_dir} -> {db_path}\n")
    ingest_rba(con, source_dir, parquet_dir)
    ingest_asx(con, source_dir, parquet_dir)
    ingest_afr(con, source_dir, parquet_dir)
    con.close()

    print(f"\nVerifying {3} concurrent read-only connections against {db_path}:")
    verify_concurrent_reads(db_path, n=3)

    print(f"\nDone. Query it with: duckdb {db_path}")


if __name__ == "__main__":
    main()
