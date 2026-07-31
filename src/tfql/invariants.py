"""Financial invariant checks.

The step that separates a TFQL operation from an ordinary function. Each
operation asserts identities that must hold for its result to be arithmetically
coherent -- a cycle's individual changes must sum to its net move, a drawdown
must be negative and its trough must follow its peak.

An operation that fails its own invariant raises INVARIANT_FAILED rather than
returning a plausible number, because a confidently wrong figure is worth the
same as no figure and is far harder to notice.
"""

from __future__ import annotations

from typing import Any

from .errors import ErrorCode, TFQLError


def check(condition: bool, message: str, **detail: Any) -> None:
    """Assert an invariant, raising INVARIANT_FAILED when it does not hold."""
    if not condition:
        raise TFQLError(ErrorCode.INVARIANT_FAILED, message, **detail)


def check_equal(actual: Any, expected: Any, message: str, **detail: Any) -> None:
    """Exact equality -- use for integers (basis points, counts), never floats."""
    check(
        actual == expected,
        f"{message} (got {actual!r}, expected {expected!r})",
        **detail,
    )


def check_close(
    actual: float,
    expected: float,
    message: str,
    *,
    tolerance: float = 1e-9,
    **detail: Any,
) -> None:
    """Float equality within a tolerance, for price-derived quantities."""
    check(
        abs(actual - expected) <= tolerance,
        f"{message} (got {actual!r}, expected {expected!r}, tol {tolerance})",
        **detail,
    )


def check_non_empty(rows: Any, message: str, **detail: Any) -> None:
    """Guard an empty result set with NO_MATCHING_RECORDS, not an IndexError."""
    if not rows:
        raise TFQLError(ErrorCode.NO_MATCHING_RECORDS, message, **detail)
