"""Loaders for the real hackathon datasets under ``AI_Industry_Training_Hackathon``.

Three datasets, three shapes (see the brief for the full schema):

* RBA cash-rate decisions -- one small CSV/JSONL, ``RBA-rates.csv``.
  Fields: ``Effective Date`` (``d Mon YYYY``), ``Change % points``
  (signed, e.g. ``+0.25``), ``Cash rate target%``.
* ASX prices -- one JSONL file per company under ``ASX/``.
  Fields: ``ticker`` (e.g. ``BHP.AX``), ``date`` (ISO ``YYYY-MM-DD``),
  ``open``/``high``/``low``/``close``/``volume``.
* AFR articles -- ~85 monthly JSONL files under ``AFR/`` (~220k articles,
  ~780MB total). Fields: ``HEADLINE``, ``SUBHEAD``, ``INTRO``, ``TEXT``,
  ``NEWSPAPER``, ``PUBLICATIONDATE`` (``YYYYMMDD`` string).

All three are parsed with the standard library only (csv/json), matching the
brief's "structured parsing and deterministic calculations" requirement --
no pandas, no external index. Each loader is a module-level cache: the
first call parses from disk, later calls in the same process reuse the
parsed result. RBA and ASX are small (175 rows; ~32k rows across 18 files)
and load in well under a second. AFR is the expensive one (~13s to scan all
85 files) -- callers doing this from an async tool should run the first
load in a thread (see ``finance_tools.py``) so it doesn't block the event
loop for concurrent requests.
"""

from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "AI_Industry_Training_Hackathon" / "data set"
RBA_CSV = DATA_ROOT / "RBA Rates" / "RBA-rates.csv"
ASX_DIR = DATA_ROOT / "ASX"
AFR_DIR = DATA_ROOT / "AFR"


class DatasetNotFoundError(RuntimeError):
    """Raised when the real hackathon dataset isn't present at DATA_ROOT."""


def _require(path: Path) -> Path:
    if not path.exists():
        raise DatasetNotFoundError(
            f"Expected hackathon dataset at {path}, but it does not exist. "
            "Check that AI_Industry_Training_Hackathon/data set is present "
            "at the repo root."
        )
    return path


# ---------------------------------------------------------------------------
# RBA cash-rate decisions
# ---------------------------------------------------------------------------
_rba_cache: list[dict[str, Any]] | None = None


def load_rba() -> list[dict[str, Any]]:
    """Return all RBA decision rows sorted by effective date (ascending).

    Each row: ``{"date": date, "change": float, "rate": float}``.
    """
    global _rba_cache
    if _rba_cache is not None:
        return _rba_cache

    _require(RBA_CSV)
    rows: list[dict[str, Any]] = []
    with open(RBA_CSV, newline="", encoding="utf-8-sig") as f:
        for record in csv.DictReader(f):
            effective = datetime.strptime(
                record["Effective Date"].strip(), "%d %b %Y"
            ).date()
            rows.append({
                "date": effective,
                "change": float(record["Change % points"].replace("+", "")),
                "rate": float(record["Cash rate target%"]),
            })
    rows.sort(key=lambda r: r["date"])
    _rba_cache = rows
    return rows


# ---------------------------------------------------------------------------
# ASX company prices
# ---------------------------------------------------------------------------
_asx_ticker_index: dict[str, Path] | None = None
_asx_cache: dict[str, list[dict[str, Any]]] = {}


def _build_asx_ticker_index() -> dict[str, Path]:
    _require(ASX_DIR)
    index: dict[str, Path] = {}
    for path in sorted(ASX_DIR.glob("*.jsonl")):
        company = path.stem.split("-ASX-")[0].upper()
        with open(path, encoding="utf-8-sig") as f:
            first = json.loads(f.readline())
        ticker = str(first.get("ticker", "")).upper()
        for alias in {company, ticker, ticker.removesuffix(".AX")}:
            if alias:
                index[alias] = path
    return index


def resolve_asx_ticker(name: str) -> Path:
    """Resolve a user-supplied company/ticker string to its data file.

    Accepts company names (``"BHP"``), Yahoo-style tickers (``"BHP.AX"``),
    or bare codes (``"bhp"``) case-insensitively.
    """
    global _asx_ticker_index
    if _asx_ticker_index is None:
        _asx_ticker_index = _build_asx_ticker_index()
    key = name.strip().upper().removesuffix(".AX")
    path = _asx_ticker_index.get(key) or _asx_ticker_index.get(f"{key}.AX")
    if path is None:
        available = sorted({p.stem.split("-ASX-")[0] for p in _asx_ticker_index.values()})
        raise ValueError(
            f"Unknown ASX ticker/company {name!r}. Available: {available}"
        )
    return path


def load_asx(ticker: str) -> list[dict[str, Any]]:
    """Return one company's daily OHLCV rows sorted by date (ascending).

    Each row: ``{"date": "YYYY-MM-DD", "open", "high", "low", "close",
    "volume"}``. Dates are kept as ISO strings (already sortable/comparable
    as-is) rather than parsed, since nothing here needs date arithmetic.
    """
    path = resolve_asx_ticker(ticker)
    cache_key = str(path)
    if cache_key in _asx_cache:
        return _asx_cache[cache_key]

    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["date"])
    _asx_cache[cache_key] = rows
    return rows


def list_asx_tickers() -> list[str]:
    global _asx_ticker_index
    if _asx_ticker_index is None:
        _asx_ticker_index = _build_asx_ticker_index()
    return sorted({p.stem.split("-ASX-")[0] for p in _asx_ticker_index.values()})


# ---------------------------------------------------------------------------
# AFR articles
# ---------------------------------------------------------------------------
_afr_cache: list[dict[str, Any]] | None = None


def load_afr() -> list[dict[str, Any]]:
    """Return every AFR article across all monthly files (~220k rows).

    Each row keeps the original fields plus a parsed ``publication_date``
    (``date`` object, from the raw ``YYYYMMDD`` string).

    This is the expensive loader (~10-15s to scan ~780MB across ~85 files).
    It's cached at module scope after the first call. Callers on an async
    hot path should run the *first* call via ``asyncio.to_thread`` so it
    doesn't block the event loop -- see ``finance_tools.py``.
    """
    global _afr_cache
    if _afr_cache is not None:
        return _afr_cache

    _require(AFR_DIR)
    rows: list[dict[str, Any]] = []
    for path in sorted(AFR_DIR.glob("*.jsonl")):
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                raw_date = str(record.get("PUBLICATIONDATE", "")).strip()
                record["publication_date"] = (
                    datetime.strptime(raw_date, "%Y%m%d").date()
                    if len(raw_date) == 8 and raw_date.isdigit()
                    else None
                )
                rows.append(record)
    _afr_cache = rows
    return rows


def combined_text(record: dict[str, Any]) -> str:
    """The four searchable fields joined, per the brief's search scope."""
    return " ".join(
        str(record.get(field) or "")
        for field in ("HEADLINE", "SUBHEAD", "INTRO", "TEXT")
    )


def in_range(d: date | None, start: date | None, end: date | None) -> bool:
    if d is None:
        return False
    if start is not None and d < start:
        return False
    if end is not None and d > end:
        return False
    return True
