#!/usr/bin/env python3
"""
query_tools.py — Generic, schema-validated query functions over warehouse.duckdb.

These are meant to become callable tools for an agentic system: each function
takes plain arguments (table/column names as strings, filters as plain dicts),
does its own schema validation, and returns JSON-serializable data (list of
dicts / dict) — no DuckDB objects, no cursors, nothing an agent framework
would need to know how to serialize.

Design rules that matter for safety, since table/column names here may come
from an LLM rather than a trusted caller:
  - Every table and column name is checked against information_schema before
    being interpolated into SQL (identifiers can't be parameterized with `?`
    in SQL, so whitelisting is the alternative to string-splicing raw input).
  - Every filter VALUE is passed as a bound parameter (`?`), never interpolated.
  - Connections are opened read_only=True, so even a validation gap can't
    mutate the warehouse.
  - run_readonly_sql() is a guarded escape hatch for questions the structured
    functions below don't cover — it still runs read-only and rejects
    anything that isn't a single SELECT statement, but it's inherently less
    safe than the structured functions and should be preferred last.

Usage as a library:
    from query_tools import get_min, get_max, aggregate, filter_rows, ...
    get_min("rba_rates", "cash_rate_target")

Usage as a CLI (runs a demo of every function against warehouse.duckdb):
    python3 query_tools.py [db_path]
"""

import re
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

DEFAULT_DB = "./warehouse.duckdb"

_OPS = {
    "=": ("= ?", 1),
    "!=": ("!= ?", 1),
    "<": ("< ?", 1),
    "<=": ("<= ?", 1),
    ">": ("> ?", 1),
    ">=": (">= ?", 1),
    "like": ("LIKE ?", 1),
    "ilike": ("ILIKE ?", 1),
    "in": (None, "*"),  # special-cased below (variable arity)
    "is_null": ("IS NULL", 0),
    "is_not_null": ("IS NOT NULL", 0),
}

_AGGS = {"min", "max", "avg", "sum", "count", "count_distinct", "median", "stddev"}


class QueryError(ValueError):
    """Raised for invalid table/column/operator/aggregate names or malformed filters."""


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------

def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    if not Path(db_path).exists():
        raise QueryError(f"database not found: {db_path}")
    return duckdb.connect(db_path, read_only=True)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _valid_tables(con) -> set:
    return {r[0] for r in con.execute("SHOW TABLES").fetchall()}


def _valid_columns(con, table: str) -> dict:
    rows = con.execute(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ?",
        [table],
    ).fetchall()
    return {name: dtype for name, dtype in rows}


def _require_table(con, table: str) -> dict:
    tables = _valid_tables(con)
    if table not in tables:
        raise QueryError(f"unknown table {table!r}; available tables: {sorted(tables)}")
    return _valid_columns(con, table)


def _require_column(columns: dict, column: str, table: str) -> None:
    if column not in columns:
        raise QueryError(f"unknown column {column!r} on table {table!r}; available columns: {sorted(columns)}")


def _build_where(filters: list[dict] | None, columns: dict, table: str) -> tuple[str, list]:
    """filters: [{"column": str, "op": str, "value": Any}, ...] combined with AND."""
    if not filters:
        return "", []

    clauses = []
    params: list = []
    for f in filters:
        col = f.get("column")
        op = f.get("op", "=")
        if col is None:
            raise QueryError(f"filter missing 'column': {f}")
        _require_column(columns, col, table)
        op_key = op.lower()
        if op_key not in _OPS:
            raise QueryError(f"unsupported filter op {op!r}; supported: {sorted(_OPS)}")

        if op_key == "in":
            values = f.get("value")
            if not isinstance(values, (list, tuple)) or not values:
                raise QueryError(f"filter op 'in' requires a non-empty list value: {f}")
            placeholders = ", ".join(["?"] * len(values))
            clauses.append(f"{_quote(col)} IN ({placeholders})")
            params.extend(values)
            continue

        template, arity = _OPS[op_key]
        clauses.append(f"{_quote(col)} {template}")
        if arity == 1:
            if "value" not in f:
                raise QueryError(f"filter op {op!r} requires a 'value': {f}")
            params.append(f["value"])

    return "WHERE " + " AND ".join(clauses), params


def _rows_to_dicts(cur) -> list[dict]:
    cols = [d[0] for d in cur.description]
    out = []
    for row in cur.fetchall():
        record = {}
        for c, v in zip(cols, row):
            if isinstance(v, (date, datetime)):
                v = v.isoformat()
            elif isinstance(v, Decimal):
                v = float(v)
            record[c] = v
        out.append(record)
    return out


# --------------------------------------------------------------------------
# schema discovery — call this first if you don't already know what's queryable
# --------------------------------------------------------------------------

def get_schema(db_path: str = DEFAULT_DB) -> dict:
    """Return every table's columns/types and row count, for tool discovery."""
    con = _connect(db_path)
    try:
        schema = {}
        for table in sorted(_valid_tables(con)):
            columns = _valid_columns(con, table)
            n = con.execute(f"SELECT count(*) FROM {_quote(table)}").fetchone()[0]
            schema[table] = {"row_count": n, "columns": columns}
        return schema
    finally:
        con.close()


def list_distinct_values(
    table: str,
    column: str,
    filters: list[dict] | None = None,
    limit: int = 100,
    db_path: str = DEFAULT_DB,
) -> list:
    """Distinct values of one column, e.g. discovering the tickers or newspapers present."""
    con = _connect(db_path)
    try:
        columns = _require_table(con, table)
        _require_column(columns, column, table)
        where_sql, params = _build_where(filters, columns, table)
        sql = f"SELECT DISTINCT {_quote(column)} FROM {_quote(table)} {where_sql} ORDER BY 1 LIMIT ?"
        return [r[0] for r in con.execute(sql, params + [limit]).fetchall()]
    finally:
        con.close()


# --------------------------------------------------------------------------
# row retrieval
# --------------------------------------------------------------------------

def filter_rows(
    table: str,
    filters: list[dict] | None = None,
    columns: list[str] | None = None,
    order_by: str | None = None,
    ascending: bool = True,
    limit: int = 50,
    db_path: str = DEFAULT_DB,
) -> list[dict]:
    """Fetch rows matching filters. `filters` is a list of
    {"column", "op", "value"} dicts ANDed together. `op` is one of:
    =, !=, <, <=, >, >=, like, ilike, in, is_null, is_not_null.
    """
    con = _connect(db_path)
    try:
        table_columns = _require_table(con, table)
        select_cols = "*"
        if columns:
            for c in columns:
                _require_column(table_columns, c, table)
            select_cols = ", ".join(_quote(c) for c in columns)
        where_sql, params = _build_where(filters, table_columns, table)
        order_sql = ""
        if order_by:
            _require_column(table_columns, order_by, table)
            order_sql = f"ORDER BY {_quote(order_by)} {'ASC' if ascending else 'DESC'}"
        sql = f"SELECT {select_cols} FROM {_quote(table)} {where_sql} {order_sql} LIMIT ?"
        return _rows_to_dicts(con.execute(sql, params + [limit]))
    finally:
        con.close()


def count_matching(table: str, filters: list[dict] | None = None, db_path: str = DEFAULT_DB) -> int:
    """Count rows matching filters (same filter syntax as filter_rows)."""
    con = _connect(db_path)
    try:
        columns = _require_table(con, table)
        where_sql, params = _build_where(filters, columns, table)
        sql = f"SELECT count(*) FROM {_quote(table)} {where_sql}"
        return con.execute(sql, params).fetchone()[0]
    finally:
        con.close()


# --------------------------------------------------------------------------
# extremes / aggregation
# --------------------------------------------------------------------------

def extremes(
    table: str,
    column: str,
    direction: str = "max",
    n: int = 1,
    filters: list[dict] | None = None,
    return_columns: list[str] | None = None,
    db_path: str = DEFAULT_DB,
) -> list[dict]:
    """Return the n rows with the highest ('max') or lowest ('min') value of
    `column`, optionally filtered. Returns full rows (or `return_columns` if
    given), not just the extreme value — use aggregate() if you only need
    the scalar.
    """
    if direction not in ("max", "min"):
        raise QueryError("direction must be 'max' or 'min'")
    return filter_rows(
        table,
        filters=filters,
        columns=return_columns,
        order_by=column,
        ascending=(direction == "min"),
        limit=n,
        db_path=db_path,
    )


def get_min(table: str, column: str, filters: list[dict] | None = None,
            return_columns: list[str] | None = None, db_path: str = DEFAULT_DB) -> dict | None:
    """Convenience wrapper: the single row with the lowest `column` value."""
    rows = extremes(table, column, "min", n=1, filters=filters, return_columns=return_columns, db_path=db_path)
    return rows[0] if rows else None


def get_max(table: str, column: str, filters: list[dict] | None = None,
            return_columns: list[str] | None = None, db_path: str = DEFAULT_DB) -> dict | None:
    """Convenience wrapper: the single row with the highest `column` value."""
    rows = extremes(table, column, "max", n=1, filters=filters, return_columns=return_columns, db_path=db_path)
    return rows[0] if rows else None


def aggregate(
    table: str,
    column: str,
    agg: str = "count",
    filters: list[dict] | None = None,
    group_by: list[str] | None = None,
    order_by_agg: str = "none",
    limit: int = 100,
    db_path: str = DEFAULT_DB,
) -> list[dict]:
    """Compute an aggregate (min, max, avg, sum, count, count_distinct, median,
    stddev) over `column`, optionally grouped by other columns. Without
    group_by, returns a single-row list. `order_by_agg`: 'none' | 'asc' | 'desc'
    (sorts groups by the aggregate value — useful for "top N by X").
    """
    agg_key = agg.lower()
    if agg_key not in _AGGS:
        raise QueryError(f"unsupported aggregate {agg!r}; supported: {sorted(_AGGS)}")
    if order_by_agg not in ("none", "asc", "desc"):
        raise QueryError("order_by_agg must be 'none', 'asc', or 'desc'")

    con = _connect(db_path)
    try:
        columns = _require_table(con, table)
        _require_column(columns, column, table)
        agg_sql = {
            "count_distinct": f"count(DISTINCT {_quote(column)})",
            "median": f"median({_quote(column)})",
        }.get(agg_key, f"{agg_key}({_quote(column)})")

        result_alias = f"{column}_{agg_key}"
        select_parts = []
        if group_by:
            for g in group_by:
                _require_column(columns, g, table)
            select_parts.extend(_quote(g) for g in group_by)
        select_parts.append(f"{agg_sql} AS {_quote(result_alias)}")

        where_sql, params = _build_where(filters, columns, table)
        group_sql = f"GROUP BY {', '.join(_quote(g) for g in group_by)}" if group_by else ""
        order_sql = ""
        if order_by_agg != "none":
            order_sql = f"ORDER BY {_quote(result_alias)} {'ASC' if order_by_agg == 'asc' else 'DESC'}"
        sql = f"""
            SELECT {', '.join(select_parts)}
            FROM {_quote(table)}
            {where_sql}
            {group_sql}
            {order_sql}
            LIMIT ?
        """
        return _rows_to_dicts(con.execute(sql, params + [limit]))
    finally:
        con.close()


def date_range(table: str, date_column: str, filters: list[dict] | None = None,
               db_path: str = DEFAULT_DB) -> dict:
    """Earliest/latest value of a date column, plus the matching row count."""
    con = _connect(db_path)
    try:
        columns = _require_table(con, table)
        _require_column(columns, date_column, table)
        where_sql, params = _build_where(filters, columns, table)
        sql = f"""
            SELECT min({_quote(date_column)}), max({_quote(date_column)}), count(*)
            FROM {_quote(table)} {where_sql}
        """
        lo, hi, n = con.execute(sql, params).fetchone()
        return {
            "earliest": lo.isoformat() if lo is not None else None,
            "latest": hi.isoformat() if hi is not None else None,
            "row_count": n,
        }
    finally:
        con.close()


# --------------------------------------------------------------------------
# time-series specific: movers and gaps
# --------------------------------------------------------------------------

def period_over_period_change(
    table: str,
    value_column: str,
    date_column: str,
    partition_by: str | None = None,
    direction: str = "desc",
    n: int = 1,
    filters: list[dict] | None = None,
    db_path: str = DEFAULT_DB,
) -> list[dict]:
    """Biggest movers: the n rows with the largest (direction='desc') or
    smallest (direction='asc') period-over-period percentage change in
    `value_column`, ordered by `date_column`. `partition_by` restarts the
    comparison per group (e.g. per ticker) so periods aren't compared across
    unrelated series.
    """
    if direction not in ("asc", "desc"):
        raise QueryError("direction must be 'asc' or 'desc'")
    con = _connect(db_path)
    try:
        columns = _require_table(con, table)
        _require_column(columns, value_column, table)
        _require_column(columns, date_column, table)
        partition_sql = ""
        if partition_by:
            _require_column(columns, partition_by, table)
            partition_sql = f"PARTITION BY {_quote(partition_by)}"
        where_sql, params = _build_where(filters, columns, table)
        sql = f"""
            WITH chg AS (
                SELECT *,
                       lag({_quote(value_column)}) OVER ({partition_sql} ORDER BY {_quote(date_column)}) AS _prev_value
                FROM {_quote(table)}
                {where_sql}
            )
            SELECT *,
                   round(({_quote(value_column)} - _prev_value) / _prev_value * 100, 4) AS pct_change
            FROM chg
            WHERE _prev_value IS NOT NULL
            ORDER BY pct_change {'DESC' if direction == 'desc' else 'ASC'}
            LIMIT ?
        """
        return _rows_to_dicts(con.execute(sql, params + [n]))
    finally:
        con.close()


def longest_gap(
    table: str,
    date_column: str,
    filters: list[dict] | None = None,
    partition_by: str | None = None,
    n: int = 1,
    db_path: str = DEFAULT_DB,
) -> list[dict]:
    """Among rows matching `filters` (e.g. change_pct != 0), find the n
    longest gaps in days between consecutive matching rows, ordered by
    `date_column`. This is the generalized version of "longest stretch
    between qualifying events" (e.g. between non-zero RBA rate changes).
    `partition_by` restarts gap tracking per group (e.g. per ticker).
    """
    con = _connect(db_path)
    try:
        columns = _require_table(con, table)
        _require_column(columns, date_column, table)
        partition_sql = ""
        if partition_by:
            _require_column(columns, partition_by, table)
            partition_sql = f"PARTITION BY {_quote(partition_by)}"
        where_sql, params = _build_where(filters, columns, table)
        sql = f"""
            WITH matching AS (
                SELECT * FROM {_quote(table)} {where_sql}
            ),
            gaps AS (
                SELECT *,
                       lag({_quote(date_column)}) OVER ({partition_sql} ORDER BY {_quote(date_column)}) AS _prev_date
                FROM matching
            )
            SELECT *,
                   date_diff('day', _prev_date, {_quote(date_column)}) AS gap_days
            FROM gaps
            WHERE _prev_date IS NOT NULL
            ORDER BY gap_days DESC
            LIMIT ?
        """
        return _rows_to_dicts(con.execute(sql, params + [n]))
    finally:
        con.close()


# --------------------------------------------------------------------------
# text search
# --------------------------------------------------------------------------

def text_search(
    table: str,
    columns: list[str],
    keyword: str,
    match_all: bool = False,
    filters: list[dict] | None = None,
    return_columns: list[str] | None = None,
    limit: int = 50,
    db_path: str = DEFAULT_DB,
) -> list[dict]:
    """Case-insensitive substring search for `keyword` across one or more
    text columns. By default any column matching is enough (OR, e.g.
    headline-or-text); `match_all=True` requires every listed column to
    contain the keyword (AND). Any extra `filters` are always ANDed on top.
    """
    con = _connect(db_path)
    try:
        table_columns = _require_table(con, table)
        for c in columns:
            _require_column(table_columns, c, table)
        select_cols = "*"
        if return_columns:
            for c in return_columns:
                _require_column(table_columns, c, table)
            select_cols = ", ".join(_quote(c) for c in return_columns)

        joiner = " AND " if match_all else " OR "
        search_clause = joiner.join(f"{_quote(c)} ILIKE ?" for c in columns)
        params = [f"%{keyword}%" for _ in columns]

        where_extra, extra_params = _build_where(filters, table_columns, table)
        if where_extra:
            where_sql = f"WHERE ({search_clause}) AND " + where_extra[len("WHERE "):]
        else:
            where_sql = f"WHERE {search_clause}"
        params = params + extra_params

        sql = f"SELECT {select_cols} FROM {_quote(table)} {where_sql} LIMIT ?"
        return _rows_to_dicts(con.execute(sql, params + [limit]))
    finally:
        con.close()


def full_text_search(
    table: str,
    id_column: str,
    query: str,
    filters: list[dict] | None = None,
    return_columns: list[str] | None = None,
    limit: int = 20,
    db_path: str = DEFAULT_DB,
) -> list[dict]:
    """BM25-ranked full-text search using a DuckDB FTS index (see
    setup_duckdb.py, which builds fts_main_afr_articles on afr_articles's
    headline/subhead/text). Unlike text_search()'s plain substring match,
    this ranks results by relevance and tokenizes/stems the query, so e.g.
    "banking" and "banks" also match a query for "bank" — at the cost of
    requiring a prebuilt index (raises QueryError if none exists for
    `table`/`id_column`). Prefer this over text_search() when a query has
    many hits and you need the most relevant ones, not just any match.
    """
    con = _connect(db_path)
    try:
        columns = _require_table(con, table)
        _require_column(columns, id_column, table)
        con.execute("LOAD fts")

        fts_schema = f"fts_main_{table}"
        exists = con.execute(
            "SELECT count(*) FROM information_schema.schemata WHERE schema_name = ?", [fts_schema]
        ).fetchone()[0]
        if not exists:
            raise QueryError(
                f"no FTS index found for table {table!r} (expected schema {fts_schema!r}); "
                "build one with PRAGMA create_fts_index(...) — see setup_duckdb.py"
            )

        select_cols = "*"
        if return_columns:
            for c in return_columns:
                _require_column(columns, c, table)
            select_cols = ", ".join(_quote(c) for c in return_columns)

        where_extra, extra_params = _build_where(filters, columns, table)
        extra_clause = (" AND " + where_extra[len("WHERE "):]) if where_extra else ""

        sql = f"""
            WITH scored AS (
                SELECT {select_cols}, {fts_schema}.match_bm25({_quote(id_column)}, ?) AS _bm25_score
                FROM {_quote(table)}
            )
            SELECT * FROM scored
            WHERE _bm25_score IS NOT NULL {extra_clause}
            ORDER BY _bm25_score DESC
            LIMIT ?
        """
        return _rows_to_dicts(con.execute(sql, [query] + extra_params + [limit]))
    finally:
        con.close()


# --------------------------------------------------------------------------
# guarded escape hatch
# --------------------------------------------------------------------------

_SELECT_ONLY = re.compile(r"^\s*(WITH\b.+?\)\s*)?SELECT\b", re.IGNORECASE | re.DOTALL)
_FORBIDDEN = re.compile(r";|--|/\*|\b(insert|update|delete|drop|alter|attach|copy|pragma|create|call)\b", re.IGNORECASE)


def run_readonly_sql(sql: str, params: list | None = None, db_path: str = DEFAULT_DB) -> list[dict]:
    """Escape hatch for read-only queries the structured functions above
    don't cover. Rejects anything that isn't a single SELECT/WITH...SELECT
    statement; the underlying connection is also opened read_only=True as a
    second line of defense. Prefer the structured functions when they fit —
    this one has no per-column validation.
    """
    if not _SELECT_ONLY.match(sql) or _FORBIDDEN.search(sql):
        raise QueryError("only a single read-only SELECT (optionally with a WITH clause) is allowed")
    con = _connect(db_path)
    try:
        return _rows_to_dicts(con.execute(sql, params or []))
    finally:
        con.close()


# --------------------------------------------------------------------------
# demo / self-check
# --------------------------------------------------------------------------

def _demo(db_path: str) -> None:
    import json

    def show(label, value):
        print(f"\n-- {label} --")
        print(json.dumps(value, indent=2, default=str))

    show("schema", get_schema(db_path))
    show("distinct tickers", list_distinct_values("asx_prices", "ticker", db_path=db_path))
    show("lowest RBA cash rate (get_min)", get_min("rba_rates", "cash_rate_target", db_path=db_path))
    show(
        "BHP highest close (get_max, filtered)",
        get_max("asx_prices", "close", filters=[{"column": "ticker", "op": "=", "value": "BHP.AX"}], db_path=db_path),
    )
    show(
        "count of RBA hikes in the 2022-2023 cycle",
        count_matching(
            "rba_rates",
            filters=[
                {"column": "change_pct", "op": ">", "value": 0},
                {"column": "effective_date", "op": ">=", "value": "2022-05-04"},
                {"column": "effective_date", "op": "<=", "value": "2023-11-08"},
            ],
            db_path=db_path,
        ),
    )
    show(
        "avg close per ticker (aggregate + group_by)",
        aggregate("asx_prices", "close", "avg", group_by=["ticker"], order_by_agg="desc", db_path=db_path),
    )
    show("RBA decision date range", date_range("rba_rates", "effective_date", db_path=db_path))
    show(
        "BHP biggest single-day % decline (period_over_period_change)",
        period_over_period_change(
            "asx_prices", "close", "date",
            filters=[{"column": "ticker", "op": "=", "value": "BHP.AX"}],
            direction="asc", n=1, db_path=db_path,
        ),
    )
    show(
        "longest gap between non-zero RBA changes (longest_gap)",
        longest_gap("rba_rates", "effective_date", filters=[{"column": "change_pct", "op": "!=", "value": 0}], db_path=db_path),
    )
    show(
        "AFR articles mentioning 'bank' (text_search, substring)",
        text_search("afr_articles", ["headline", "text"], "bank",
                    return_columns=["headline", "publication_date"], limit=3, db_path=db_path),
    )
    show(
        "AFR articles most relevant to 'bank' (full_text_search, BM25-ranked)",
        full_text_search("afr_articles", "article_id", "bank",
                          return_columns=["headline", "publication_date"], limit=3, db_path=db_path),
    )
    show(
        "escape hatch: run_readonly_sql",
        run_readonly_sql("SELECT count(*) AS n FROM afr_articles WHERE year = ?", [2015], db_path=db_path),
    )


if __name__ == "__main__":
    _demo(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB)
