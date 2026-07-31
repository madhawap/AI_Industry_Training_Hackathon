# Tools catalog

Functions in [query_tools.py](query_tools.py), documented as tools for an agentic system to call. Every function takes plain arguments and returns JSON-serializable data (dicts/lists/scalars) — no DuckDB objects.

```python
from query_tools import get_min, get_max, aggregate, filter_rows, ...
get_min("rba_rates", "cash_rate_target")
```

## Safety model

These functions may end up driven by an LLM rather than a trusted caller, so:

- **Identifiers are whitelisted, not interpolated.** Every table and column name is checked against `information_schema` before being used in a query. SQL doesn't support parameterizing identifiers (`?` only works for values), so whitelisting is the substitute — an unknown table/column raises `QueryError` naming the valid options instead of running.
- **Values are always bound parameters.** Filter values, search keywords, etc. go through `?` placeholders, never string interpolation.
- **Connections are read-only.** Every function opens its own `duckdb.connect(db_path, read_only=True)`, so even a gap in the validation above can't mutate the warehouse.
- **The one escape hatch is explicitly guarded.** `run_readonly_sql()` exists for questions the structured functions don't cover. It rejects anything that isn't a single `SELECT` (or `WITH ... SELECT`) statement — no semicolons, comments, or DDL/DML keywords — on top of the read-only connection. Prefer a structured function when one fits; it has per-identifier validation this one doesn't.

Verified against six injection attempts (bad table/column names, a bogus filter op, and three `run_readonly_sql` bypass attempts including `DROP TABLE` and a comment-based bypass) — all six raised `QueryError` rather than executing.

## The `filters` argument

Every function that takes `filters` uses the same shape — a list of conditions ANDed together:

```python
filters = [
    {"column": "ticker", "op": "=", "value": "BHP.AX"},
    {"column": "close", "op": ">", "value": 20},
]
```

`op` is one of: `=`, `!=`, `<`, `<=`, `>`, `>=`, `like`, `ilike`, `in` (value must be a non-empty list), `is_null`, `is_not_null` (no value needed).

## Current schema

Call `get_schema()` for the live version of this; as of writing:

| table | columns | rows |
|---|---|---|
| `rba_rates` | `effective_date` (DATE), `change_pct` (DOUBLE), `cash_rate_target` (DOUBLE) | 175 |
| `asx_prices` | `date` (DATE), `open`/`high`/`low`/`close` (DOUBLE), `volume` (BIGINT), `ticker` (VARCHAR) | 31,932 (18 tickers) |
| `afr_articles` | `article_id` (BIGINT), `headline`/`subhead`/`newspaper`/`text`/`intro` (VARCHAR), `publication_date` (DATE), `year`/`month` (BIGINT) | 219,446 |

`afr_articles` also has a DuckDB FTS index (`fts_main_afr_articles`) over `headline`/`subhead`/`text`, built by `setup_duckdb.py` — this is what `full_text_search()` uses.

---

## `get_schema()`

Every table's columns, types, and row count. Call this first to discover what's queryable — it's the schema-discovery tool for everything else below.

**Args:** `db_path` (default `./warehouse.duckdb`)
**Returns:** `{table_name: {"row_count": int, "columns": {col_name: sql_type}}}`

```python
get_schema()
# {
#   "afr_articles": {"row_count": 219446, "columns": {"headline": "VARCHAR", ...}},
#   "asx_prices":   {"row_count": 31932,  "columns": {"date": "DATE", ...}},
#   "rba_rates":    {"row_count": 175,    "columns": {"effective_date": "DATE", ...}}
# }
```

## `list_distinct_values(table, column, filters=None, limit=100)`

Distinct values of one column — e.g. discovering which tickers or newspapers are present before filtering on them.

**Returns:** a plain list of values.

```python
list_distinct_values("asx_prices", "ticker")
# ["AGL.AX", "AMP.AX", "ANZ.AX", "AZJ.AX", "BHP.AX", "CBA.AX", "CMW.AX", "GPT.AX",
#  "IAG.AX", "NAB.AX", "QAN.AX", "QBE.AX", "RIO.AX", "SGP.AX", "SUN.AX", "TAH.AX",
#  "TCL.AX", "TPG.AX"]
```

## `filter_rows(table, filters=None, columns=None, order_by=None, ascending=True, limit=50)`

Generic filtered row fetch — the general-purpose "give me rows matching X" tool. `columns` restricts which fields come back (default all).

**Returns:** list of row dicts.

```python
filter_rows("rba_rates", filters=[{"column": "change_pct", "op": ">", "value": 0}],
            order_by="effective_date", limit=3)
# [{"effective_date": "2010-03-03", "change_pct": 0.25, "cash_rate_target": 4.0}, ...]
```

## `count_matching(table, filters=None)`

Count of rows matching filters — same filter syntax as `filter_rows`, but returns just the count. Useful when the agent only needs "how many", not the rows themselves.

**Returns:** `int`.

```python
count_matching("rba_rates", filters=[
    {"column": "change_pct", "op": ">", "value": 0},
    {"column": "effective_date", "op": ">=", "value": "2022-05-04"},
    {"column": "effective_date", "op": "<=", "value": "2023-11-08"},
])
# 13
```

## `get_min(table, column, filters=None, return_columns=None)` / `get_max(...)`

Convenience wrappers: the single row with the lowest/highest value of `column`, optionally filtered. This is the "minimum value in a table, filterable by another column" case directly.

**Returns:** a single row dict, or `None` if no rows match.

```python
get_min("rba_rates", "cash_rate_target")
# {"effective_date": "2020-11-04", "change_pct": -0.15, "cash_rate_target": 0.1}

get_max("asx_prices", "close", filters=[{"column": "ticker", "op": "=", "value": "BHP.AX"}])
# {"date": "2021-08-04", "open": 33.5, "high": 33.91, "low": 33.44, "close": 33.72,
#  "volume": 5971761, "ticker": "BHP.AX"}
```

## `extremes(table, column, direction="max", n=1, filters=None, return_columns=None)`

The general form behind `get_min`/`get_max`: the *n* rows with the highest/lowest value of `column`. Use this instead of the wrappers when you want more than one row (e.g. "the 5 biggest closes").

**Returns:** list of row dicts (length ≤ n).

## `aggregate(table, column, agg="count", filters=None, group_by=None, order_by_agg="none", limit=100)`

Compute an aggregate over `column`: `min`, `max`, `avg`, `sum`, `count`, `count_distinct`, `median`, `stddev`. Without `group_by`, returns one row. With it, one row per group; `order_by_agg` (`"asc"`/`"desc"`) sorts groups by the aggregate value — the "top N by X" pattern.

**Returns:** list of dicts, each with the group-by columns plus `{column}_{agg}`.

```python
aggregate("asx_prices", "close", "avg", group_by=["ticker"], order_by_agg="desc")
# [{"ticker": "CBA.AX", "close_avg": 58.33}, {"ticker": "RIO.AX", "close_avg": 48.44},
#  {"ticker": "NAB.AX", "close_avg": 17.95}, ... 15 more tickers ...,
#  {"ticker": "TAH.AX", "close_avg": 0.21}]
```

## `date_range(table, date_column, filters=None)`

Earliest/latest value of a date column plus the matching row count — the general form of "what date range does this dataset span".

**Returns:** `{"earliest": iso_date, "latest": iso_date, "row_count": int}`.

```python
date_range("rba_rates", "effective_date")
# {"earliest": "2010-02-03", "latest": "2026-06-17", "row_count": 175}
```

## `period_over_period_change(table, value_column, date_column, partition_by=None, direction="desc", n=1, filters=None)`

Biggest movers: the *n* rows with the largest (`direction="desc"`) or smallest (`"asc"`) period-over-period **percentage** change in `value_column`, ordered by `date_column`. `partition_by` restarts the day-over-day comparison per group (e.g. per `ticker`) so unrelated series aren't compared against each other. Built on a `lag()` window function.

**Returns:** list of row dicts, each with the original columns plus `pct_change` (and an internal `_prev_value`).

```python
period_over_period_change("asx_prices", "close", "date",
                           filters=[{"column": "ticker", "op": "=", "value": "BHP.AX"}],
                           direction="asc", n=1)
# [{"date": "2020-03-09", "open": 17.16, ..., "close": 16.32, "ticker": "BHP.AX",
#   "_prev_value": 19.07, "pct_change": -14.4144}]
```

## `longest_gap(table, date_column, filters=None, partition_by=None, n=1)`

Among rows matching `filters` (e.g. "non-zero rate changes only"), the *n* longest gaps in days between **consecutive matching rows**. This generalizes "longest stretch between qualifying events" — the RBA example below, but works for any table/predicate. `partition_by` restarts gap tracking per group.

**Returns:** list of row dicts, each with the original columns plus `gap_days` (and an internal `_prev_date`).

```python
longest_gap("rba_rates", "effective_date", filters=[{"column": "change_pct", "op": "!=", "value": 0}])
# [{"effective_date": "2019-06-05", "change_pct": -0.25, "cash_rate_target": 1.25,
#   "_prev_date": "2016-08-03", "gap_days": 1036}]
```

## `text_search(table, columns, keyword, match_all=False, filters=None, return_columns=None, limit=50)`

Case-insensitive **substring** search across one or more text columns. By default any listed column matching is enough (OR — e.g. headline-or-text); `match_all=True` requires every listed column to contain the keyword (AND). Any extra `filters` are ANDed on top regardless. No index required — a full scan every call, fine at hundreds of rows, slow at `afr_articles`' 200k+, where `full_text_search()`'s FTS index is the better fit.

**Returns:** list of row dicts.

```python
text_search("afr_articles", ["headline", "text"], "bank",
            return_columns=["headline", "publication_date"], limit=3)
# [{"headline": "A fresh game plan to grow the economy in 2015", "publication_date": "2015-01-02"},
#  {"headline": "BIG PICTURE", "publication_date": "2015-01-02"},
#  {"headline": "Big four shrug off the need for more capital", "publication_date": "2015-01-02"}]
```

## `full_text_search(table, id_column, query, filters=None, return_columns=None, limit=20)`

BM25-**relevance-ranked** full-text search using a prebuilt DuckDB FTS index (`setup_duckdb.py` builds `fts_main_afr_articles` on `afr_articles`'s `headline`/`subhead`/`text` every ingest). Tokenizes and stems the query, so "bank" also matches "banking"/"banks" — and ranks the most relevant hits first, unlike `text_search`'s unordered substring match. Raises `QueryError` if no index exists for the given `table`/`id_column` (check `information_schema.schemata` for `fts_main_<table>`, or just build one — see `setup_duckdb.py`).

**Returns:** list of row dicts, each with an added `_bm25_score` (higher = more relevant), ordered by that score descending.

**When to use this vs. `text_search`:** reach for this one when a query could have many hits and you need the *best* matches, not just any match — e.g. "which articles are most about banking" rather than "which articles mention banking at all". `text_search` needs no setup and works on any table; this one needs an index but ranks and stems.

```python
full_text_search("afr_articles", "article_id", "bank",
                  return_columns=["headline", "publication_date"], limit=3)
# [{"headline": "New focus on small-business banking", "publication_date": "2020-06-30", "_bm25_score": 1.206},
#  {"headline": "Bank tax battle: proposal has history", "publication_date": "2017-05-18", "_bm25_score": 1.204},
#  {"headline": "Bank of Queensland set to snare ME Bank for $1.325b", "publication_date": "2021-02-19", "_bm25_score": 1.203}]
```

## `run_readonly_sql(sql, params=None)`

Guarded escape hatch for questions the structured functions above don't cover. Only a single `SELECT` (optionally `WITH ... SELECT`) is accepted — semicolons, comments, and DDL/DML keywords (`insert`, `update`, `delete`, `drop`, `alter`, `attach`, `copy`, `pragma`, `create`, `call`) are all rejected, and the connection is read-only regardless. Has **no per-column validation**, so prefer a structured function when one fits.

**Returns:** list of row dicts.

```python
run_readonly_sql("SELECT count(*) AS n FROM afr_articles WHERE year = ?", [2015])
# [{"n": 29516}]
```

---

## Running the demo / self-check

```
python3 query_tools.py [db_path]
```

Runs every function above against the real warehouse and prints its output — useful both as a smoke test and as a live reference for what each call returns.
