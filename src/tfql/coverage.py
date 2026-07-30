"""Dataset coverage intervals.

The challenge brief requires cross-dataset answers to "respect the overlapping
date coverage and clearly identify missing coverage". That is only possible if
every operation knows the interval it is working inside, so coverage is
computed once at startup and threaded into every result's evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .dates import iso
from .errors import ErrorCode, TFQLError


@dataclass(frozen=True, slots=True)
class Coverage:
    """The inclusive date span of a dataset, plus how many records it holds."""

    dataset: str
    start: date
    end: date
    record_count: int

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end

    def require(self, day: date, *, label: str = "date") -> None:
        """Raise DATE_OUTSIDE_COVERAGE if ``day`` falls outside the interval."""
        if not self.contains(day):
            raise TFQLError(
                ErrorCode.DATE_OUTSIDE_COVERAGE,
                f"{label}={iso(day)} is outside {self.dataset} coverage "
                f"({iso(self.start)} to {iso(self.end)})",
                dataset=self.dataset,
                requested=iso(day),
                available=self.describe(),
            )

    def clamp(self, start: date | None, end: date | None) -> tuple[date, date]:
        """Intersect a requested window with this coverage.

        Missing bounds default to the coverage edges. Raises when the request
        misses the dataset entirely rather than silently returning nothing.
        """
        lo = self.start if start is None else max(start, self.start)
        hi = self.end if end is None else min(end, self.end)
        if lo > hi:
            raise TFQLError(
                ErrorCode.DATE_OUTSIDE_COVERAGE,
                f"requested window does not overlap {self.dataset} coverage ({self.describe()})",
                dataset=self.dataset,
                requested=(f"{iso(start) if start else '-'} to {iso(end) if end else '-'}"),
                available=self.describe(),
            )
        return lo, hi

    def describe(self) -> str:
        return f"{iso(self.start)} to {iso(self.end)}"

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "start": iso(self.start),
            "end": iso(self.end),
            "record_count": self.record_count,
        }


def overlap(*coverages: Coverage) -> tuple[date, date] | None:
    """The interval common to every supplied coverage, or None if disjoint.

    Used to state, rather than silently span, the gaps in cross-dataset
    questions -- the AFR corpus in particular covers far less ground than the
    RBA series.
    """
    if not coverages:
        return None
    lo = max(c.start for c in coverages)
    hi = min(c.end for c in coverages)
    return (lo, hi) if lo <= hi else None


def describe_overlap(*coverages: Coverage) -> dict[str, object]:
    """A JSON-ready summary of shared coverage, for cross-dataset evidence."""
    shared = overlap(*coverages)
    return {
        "datasets": {c.dataset: c.describe() for c in coverages},
        "shared": (f"{iso(shared[0])} to {iso(shared[1])}" if shared else None),
    }
