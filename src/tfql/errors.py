"""Structured errors for TFQL.

Every failure path in the system raises a TFQLError carrying one of the codes
below. Nothing in TFQL is allowed to substitute a plausible value for a failure
-- a wrong number scores the same as no number, but only the error is
diagnosable afterwards.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    # envelope / plan level
    UNKNOWN_OPERATION = "UNKNOWN_OPERATION"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    UNKNOWN_ARGUMENT = "UNKNOWN_ARGUMENT"
    PLAN_CYCLE = "PLAN_CYCLE"
    PLAN_TOO_COMPLEX = "PLAN_TOO_COMPLEX"
    UNRESOLVED_REFERENCE = "UNRESOLVED_REFERENCE"

    # data level
    DATE_RANGE_INVALID = "DATE_RANGE_INVALID"
    DATE_OUTSIDE_COVERAGE = "DATE_OUTSIDE_COVERAGE"
    UNKNOWN_TICKER = "UNKNOWN_TICKER"
    NO_MATCHING_RECORDS = "NO_MATCHING_RECORDS"

    # execution level
    INVARIANT_FAILED = "INVARIANT_FAILED"
    OPERATION_TIMEOUT = "OPERATION_TIMEOUT"
    DEPENDENCY_FAILED = "DEPENDENCY_FAILED"


class TFQLError(Exception):
    """An error with a machine-readable code and structured detail."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        **detail: Any,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": str(self.code), "message": self.message}
        if self.detail:
            out["detail"] = self.detail
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TFQLError({self.code}: {self.message})"
