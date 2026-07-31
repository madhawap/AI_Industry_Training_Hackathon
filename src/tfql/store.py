"""Read-only data access and the startup precompute.

TFQL reads the shared DuckDB warehouse. Two consequences shape this module:

  * **The warehouse is not ours to write.** Anything that would normally be a
    materialised column -- notably the AFR four-field concatenation -- is built
    as a SQL expression at query time instead. Correct, and free at mock scale;
    see AFR_ALL_TEXT for the note on real-corpus sizing.

  * **Requests are on a 60-second clock, startup is not.** RBA and ASX are small
    enough (175 and ~31.5k rows) to load entirely into memory once, so every
    numeric operation afterwards is array work with no database round trip.
    AFR stays in DuckDB, where it can use the engine's vectorised text scan.

The built Store is treated as immutable. Nothing mutates it after ``build()``,
which is what makes it safe to share across concurrent requests without locks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

import duckdb
import numpy as np

from .coverage import Coverage
from .errors import ErrorCode, TFQLError
from .precision import pct_to_bp

DEFAULT_WAREHOUSE: Final[str] = "/home/datasets/cognitivo_hackathon/mock_data/warehouse.duckdb"

RBA_TABLE: Final[str] = "rba_rates"
ASX_TABLE: Final[str] = "asx_prices"
AFR_TABLE: Final[str] = "afr_articles"

AFR_FIELDS: Final[tuple[str, ...]] = ("headline", "subhead", "intro", "text")
"""The four fields every AFR pattern count must search.

Setup Instructions call this non-negotiable: searching only the headline or only
the body produces counts that will not match the reference answers. The
warehouse's own FTS index covers only headline/subhead/text -- intro is missing
-- which is why counting operations never touch that index.
"""

AFR_ALL_TEXT: Final[str] = " || ' ' || ".join(f"coalesce({f}, '')" for f in AFR_FIELDS)
"""SQL expression concatenating the four searchable AFR fields.

Defined once so no operation can accidentally search a narrower scope. On the
real ~200k-article corpus this concatenation should be materialised at ingest
rather than evaluated per query; see README limitations.
"""


@dataclass(frozen=True, slots=True)
class RbaSeries:
    """The RBA decision series, sorted ascending by effective date."""

    dates: list[date]
    change_bp: np.ndarray  # int64 basis points, signed
    target_bp: np.ndarray  # int64 basis points
    coverage: Coverage

    def __len__(self) -> int:
        return len(self.dates)


@dataclass(frozen=True, slots=True)
class TickerSeries:
    """One ticker's price history, sorted ascending by trading date."""

    ticker: str
    dates: list[date]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    daily_return: np.ndarray  # close-to-close, index 0 is nan
    coverage: Coverage

    def __len__(self) -> int:
        return len(self.dates)

    def field(self, name: str) -> np.ndarray:
        try:
            return {
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
            }[name]
        except KeyError:
            raise TFQLError(
                ErrorCode.INVALID_ARGUMENT,
                f"unknown price field {name!r}",
                accepted=["open", "high", "low", "close", "volume"],
            ) from None


class Store:
    """Immutable, preloaded view over the warehouse."""

    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        rba: RbaSeries,
        asx: dict[str, TickerSeries],
        afr_coverage: Coverage,
    ) -> None:
        self._con = connection
        self.rba = rba
        self.asx = asx
        self.afr_coverage = afr_coverage

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, warehouse: str | Path | None = None) -> Store:
        """Open the warehouse read-only and precompute every in-memory series.

        Called once during application startup, before ``/health`` reports
        ready. Everything expensive happens here, where no clock is running.
        """
        path = Path(warehouse or os.environ.get("TFQL_WAREHOUSE", DEFAULT_WAREHOUSE))
        if not path.exists():
            raise TFQLError(
                ErrorCode.NO_MATCHING_RECORDS,
                f"warehouse not found at {path}",
                path=str(path),
                hint="set TFQL_WAREHOUSE to the warehouse.duckdb location",
            )
        con = duckdb.connect(str(path), read_only=True)
        cls._verify_schema(con)
        return cls(
            connection=con,
            rba=cls._load_rba(con),
            asx=cls._load_asx(con),
            afr_coverage=cls._load_afr_coverage(con),
        )

    @staticmethod
    def _verify_schema(con: duckdb.DuckDBPyConnection) -> None:
        """Fail loudly at startup if the warehouse is not the expected shape."""
        expected = {
            RBA_TABLE: {"effective_date", "change_pct", "cash_rate_target"},
            ASX_TABLE: {"date", "open", "high", "low", "close", "volume", "ticker"},
            AFR_TABLE: {"article_id", "publication_date", *AFR_FIELDS},
        }
        rows = con.execute(
            "select table_name, column_name from information_schema.columns"
        ).fetchall()
        actual: dict[str, set[str]] = {}
        for table, column in rows:
            actual.setdefault(table, set()).add(column)
        for table, columns in expected.items():
            missing = columns - actual.get(table, set())
            if missing:
                raise TFQLError(
                    ErrorCode.NO_MATCHING_RECORDS,
                    f"warehouse table {table!r} is missing columns: {sorted(missing)}",
                    table=table,
                    missing=sorted(missing),
                    found=sorted(actual.get(table, set())),
                )

    @staticmethod
    def _load_rba(con: duckdb.DuckDBPyConnection) -> RbaSeries:
        rows = con.execute(
            f"select effective_date, change_pct, cash_rate_target "
            f"from {RBA_TABLE} order by effective_date"
        ).fetchall()
        if not rows:
            raise TFQLError(ErrorCode.NO_MATCHING_RECORDS, "RBA table is empty")
        dates = [r[0] for r in rows]
        # Rates become integer basis points here, once, at the boundary.
        change_bp = np.array([pct_to_bp(r[1] or 0.0) for r in rows], dtype=np.int64)
        target_bp = np.array([pct_to_bp(r[2]) for r in rows], dtype=np.int64)
        return RbaSeries(
            dates=dates,
            change_bp=change_bp,
            target_bp=target_bp,
            coverage=Coverage("rba", dates[0], dates[-1], len(dates)),
        )

    @staticmethod
    def _load_asx(con: duckdb.DuckDBPyConnection) -> dict[str, TickerSeries]:
        rows = con.execute(
            f"select ticker, date, open, high, low, close, volume "
            f"from {ASX_TABLE} order by ticker, date"
        ).fetchall()
        if not rows:
            raise TFQLError(ErrorCode.NO_MATCHING_RECORDS, "ASX table is empty")

        grouped: dict[str, list[tuple]] = {}
        for row in rows:
            grouped.setdefault(row[0], []).append(row)

        series: dict[str, TickerSeries] = {}
        for ticker, ticker_rows in grouped.items():
            dates = [r[1] for r in ticker_rows]
            close = np.array([r[5] for r in ticker_rows], dtype=np.float64)
            # Daily close-to-close returns, precomputed so "biggest single-day
            # move" is an argmin on an array that already exists.
            daily = np.full(close.shape, np.nan, dtype=np.float64)
            if close.size > 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    daily[1:] = close[1:] / close[:-1] - 1.0
            series[ticker] = TickerSeries(
                ticker=ticker,
                dates=dates,
                open=np.array([r[2] for r in ticker_rows], dtype=np.float64),
                high=np.array([r[3] for r in ticker_rows], dtype=np.float64),
                low=np.array([r[4] for r in ticker_rows], dtype=np.float64),
                close=close,
                volume=np.array(
                    [(r[6] if r[6] is not None else np.nan) for r in ticker_rows],
                    dtype=np.float64,
                ),
                daily_return=daily,
                coverage=Coverage(f"asx:{ticker}", dates[0], dates[-1], len(dates)),
            )
        return series

    @staticmethod
    def _load_afr_coverage(con: duckdb.DuckDBPyConnection) -> Coverage:
        start, end, count = con.execute(
            f"select min(publication_date), max(publication_date), count(*) from {AFR_TABLE}"
        ).fetchone()
        if not count:
            raise TFQLError(ErrorCode.NO_MATCHING_RECORDS, "AFR table is empty")
        return Coverage("afr", start, end, count)

    # ----------------------------------------------------------------- use

    def query(self, sql: str, params: list[Any] | None = None) -> list[tuple]:
        """Run a read-only query on an independent cursor.

        DuckDB connections are not safe to share across threads, but
        ``cursor()`` yields an independent one over the same database, so
        concurrent requests never contend.
        """
        return self._con.cursor().execute(sql, params or []).fetchall()

    def ticker(self, symbol: str) -> TickerSeries:
        """Look up a ticker, raising UNKNOWN_TICKER with the valid symbols."""
        try:
            return self.asx[symbol]
        except KeyError:
            raise TFQLError(
                ErrorCode.UNKNOWN_TICKER,
                f"unknown ticker {symbol!r}",
                requested=symbol,
                available=sorted(self.asx),
            ) from None

    @property
    def tickers(self) -> list[str]:
        return sorted(self.asx)

    def asx_coverage(self) -> Coverage:
        """Coverage across all tickers combined."""
        starts = [s.coverage.start for s in self.asx.values()]
        ends = [s.coverage.end for s in self.asx.values()]
        total = sum(len(s) for s in self.asx.values())
        return Coverage("asx", min(starts), max(ends), total)

    def close(self) -> None:
        self._con.close()
