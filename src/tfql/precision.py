"""Numeric precision rules.

Interest rates are stored and compared as integer **basis points** (1 bp =
0.01%), never as floats. Three reasons:

  * ``0.1 + 0.25 != 0.35`` in binary floating point, so summing a cycle's rate
    changes accumulates error and the reconciliation invariant in
    ``rba.rate_cycle`` fails spuriously.
  * ``cash_rate_target == 0.1`` is unsafe as a float equality test, and
    ``rba.rate_extreme`` depends on exactly that comparison to count how many
    records share the extreme rate.
  * The graded worked example turned on matching an exact rate and date.

Prices stay float64 -- they are genuine measurements, not exact decimals --
but are only rounded at the formatting boundary, never mid-calculation.
"""

from __future__ import annotations

from typing import Final

BP_PER_PERCENT: Final[int] = 100
"""One percentage point is 100 basis points."""

_PRICE_DP: Final[int] = 4
_PCT_DP: Final[int] = 4


def pct_to_bp(pct: float) -> int:
    """Percent (or percentage points) -> integer basis points. 4.35 -> 435."""
    return round(pct * BP_PER_PERCENT)


def bp_to_pct(bp: int) -> float:
    """Integer basis points -> percent. 435 -> 4.35."""
    return bp / BP_PER_PERCENT


def round_pct(value: float) -> float:
    """Round a percentage for output. Applied once, at the formatting step."""
    return round(value, _PCT_DP)


def round_price(value: float) -> float:
    """Round a price for output. Applied once, at the formatting step."""
    return round(value, _PRICE_DP)


def decimal_to_pct(decimal_return: float) -> float:
    """0.057984 -> 5.7984.

    Emitted alongside the decimal form so the synthesiser never multiplies.
    Any arithmetic left undone here is arithmetic the language model attempts.
    """
    return round_pct(decimal_return * 100.0)
