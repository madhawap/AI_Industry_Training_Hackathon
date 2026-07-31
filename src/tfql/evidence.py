"""Evidence bundles.

Every operation returns computed values *plus* a small record of how they were
obtained. The bundle is not decoration -- it is the fine-tuned model's input.
The synthesiser writes the final answer from the bundle alone, so:

  * every fact the answer needs must be present, and
  * every fact the answer states must be traceable to a field here.

That second property is what the rubric's "avoidance of unsupported claims"
criterion is testing, and it is enforced by construction rather than prompting.

Bundles stay small. Never raw rows -- computed values, method, coverage,
warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Evidence:
    """How a single operation reached its result."""

    dataset: str
    method: str
    records_used: int = 0
    coverage: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def note(self, key: str, value: Any) -> Evidence:
        """Record a resolution decision (alignment applied, field used, ...)."""
        self.notes[key] = value
        return self

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "dataset": self.dataset,
            "method": self.method,
            "records_used": self.records_used,
        }
        if self.coverage:
            out["coverage"] = self.coverage
        if self.notes:
            out.update(self.notes)
        return out


@dataclass(slots=True)
class OpOutput:
    """What an operation body returns, before the executor wraps it."""

    data: dict[str, Any]
    evidence: Evidence
    warnings: list[str] = field(default_factory=list)

    def warn(self, message: str) -> OpOutput:
        self.warnings.append(message)
        return self
