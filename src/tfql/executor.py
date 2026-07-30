"""Plan validation and execution.

Two properties matter more than speed here.

**Validate the whole plan before executing any of it.** An unknown operation, a
hallucinated argument, a cyclic dependency or a dangling reference is caught
while there is still budget to repair the call, rather than surfacing as a
stack trace halfway through.

**A failed operation never fails the plan.** The rubric awards points per
correct component, so three good results and one error are worth strictly more
than nothing. Failures become structured entries the synthesiser can describe as
a stated limitation.

Operations run sequentially in dependency order. That is deliberate: the data
work is microseconds against model calls measured in seconds, so parallelising
it would buy nothing measurable while costing determinism in error ordering.
"""

from __future__ import annotations

import re
from typing import Any

from . import registry
from .errors import ErrorCode, TFQLError
from .models import (
    MAX_DEPENDENCY_DEPTH,
    MAX_OPERATIONS,
    OperationRequest,
    OperationResult,
    PlanRequest,
    PlanResult,
)
from .store import Store

REFERENCE = re.compile(r"^\$\{([A-Za-z0-9_.-]+)\}$")
"""A whole-value reference to an earlier result, e.g. ``${peak.data.date}``."""


def _find_references(value: Any) -> list[str]:
    """Every ``${...}`` reference appearing anywhere in an argument tree."""
    found: list[str] = []
    if isinstance(value, str):
        match = REFERENCE.match(value)
        if match:
            found.append(match.group(1))
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_find_references(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_find_references(item))
    return found


def _dig(result: OperationResult, path: str) -> Any:
    """Walk a dotted path into a completed result."""
    current: Any = result.model_dump()
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise TFQLError(
                ErrorCode.UNRESOLVED_REFERENCE,
                f"{result.id!r} has no value at path {path!r}",
                operation_id=result.id,
                path=path,
            )
    return current


def _substitute(value: Any, results: dict[str, OperationResult]) -> Any:
    """Replace every ``${id.path}`` reference with the value it points at."""
    if isinstance(value, str):
        match = REFERENCE.match(value)
        if not match:
            return value
        ref = match.group(1)
        op_id, _, path = ref.partition(".")
        if op_id not in results:
            raise TFQLError(
                ErrorCode.UNRESOLVED_REFERENCE,
                f"reference {ref!r} points at unknown operation {op_id!r}",
                reference=ref,
            )
        return _dig(results[op_id], path) if path else results[op_id].data
    if isinstance(value, dict):
        return {k: _substitute(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, results) for v in value]
    return value


def _edges(operations: list[OperationRequest]) -> dict[str, set[str]]:
    """Dependency edges, combining explicit depends_on with implied references."""
    edges: dict[str, set[str]] = {}
    for op in operations:
        deps = set(op.depends_on)
        for ref in _find_references(op.args):
            deps.add(ref.split(".", 1)[0])
        edges[op.id] = deps
    return edges


def _order(operations: list[OperationRequest]) -> list[OperationRequest]:
    """Topologically sort the plan, raising on cycles or dangling references."""
    by_id = {op.id: op for op in operations}
    edges = _edges(operations)

    for op_id, deps in edges.items():
        for dep in deps:
            if dep not in by_id:
                raise TFQLError(
                    ErrorCode.UNRESOLVED_REFERENCE,
                    f"operation {op_id!r} depends on unknown operation {dep!r}",
                    operation_id=op_id,
                    missing=dep,
                )

    ordered: list[OperationRequest] = []
    state: dict[str, int] = {}  # 0 = visiting, 1 = done

    def visit(op_id: str, depth: int) -> None:
        if state.get(op_id) == 1:
            return
        if state.get(op_id) == 0:
            raise TFQLError(
                ErrorCode.PLAN_CYCLE,
                f"dependency cycle involving {op_id!r}",
                operation_id=op_id,
            )
        if depth > MAX_DEPENDENCY_DEPTH:
            raise TFQLError(
                ErrorCode.PLAN_TOO_COMPLEX,
                f"dependency chain deeper than {MAX_DEPENDENCY_DEPTH}",
                operation_id=op_id,
                max_depth=MAX_DEPENDENCY_DEPTH,
            )
        state[op_id] = 0
        for dep in sorted(edges[op_id]):
            visit(dep, depth + 1)
        state[op_id] = 1
        ordered.append(by_id[op_id])

    for op in operations:
        visit(op.id, 0)
    return ordered


def validate_plan(plan: PlanRequest) -> list[OperationRequest]:
    """Check the whole plan before any data is touched.

    Returns the operations in execution order. Raises TFQLError for anything
    the planner can fix and retry.
    """
    if len(plan.operations) > MAX_OPERATIONS:
        raise TFQLError(
            ErrorCode.PLAN_TOO_COMPLEX,
            f"{len(plan.operations)} operations exceeds the limit of {MAX_OPERATIONS}",
            requested=len(plan.operations),
            maximum=MAX_OPERATIONS,
        )

    seen: set[str] = set()
    for op in plan.operations:
        if op.id in seen:
            raise TFQLError(
                ErrorCode.INVALID_ARGUMENT,
                f"duplicate operation id {op.id!r}",
                operation_id=op.id,
            )
        seen.add(op.id)
        # Resolves the name now so UNKNOWN_OPERATION surfaces at plan time.
        registry.get(op.op)

    return _order(plan.operations)


def execute(plan: PlanRequest, store: Store) -> PlanResult:
    """Validate and run a plan, always returning an evidence bundle."""
    try:
        ordered = validate_plan(plan)
    except TFQLError as exc:
        # Plan-level failures still produce a bundle, so the synthesiser can
        # state the limitation instead of the request collapsing.
        return PlanResult(
            request_id=plan.request_id,
            status="error",
            results=[
                OperationResult(id=op.id, op=op.op, status="skipped", error=exc.to_dict())
                for op in plan.operations
            ],
        )

    edges = _edges(plan.operations)
    results: dict[str, OperationResult] = {}

    for op in ordered:
        failed_deps = sorted(
            dep for dep in edges[op.id] if dep in results and not results[dep].succeeded
        )
        if failed_deps:
            results[op.id] = OperationResult(
                id=op.id,
                op=op.op,
                status="skipped",
                error=TFQLError(
                    ErrorCode.DEPENDENCY_FAILED,
                    f"skipped because {', '.join(failed_deps)} did not succeed",
                    depends_on=failed_deps,
                ).to_dict(),
            )
            continue

        try:
            spec = registry.get(op.op)
            resolved_args = _substitute(op.args, results)
            args = registry.parse_args(spec, resolved_args)
            output = spec.fn(args, store)
            results[op.id] = OperationResult(
                id=op.id,
                op=op.op,
                status="ok",
                data=output.data,
                evidence=output.evidence.to_dict(),
                warnings=output.warnings,
            )
        except TFQLError as exc:
            results[op.id] = OperationResult(
                id=op.id, op=op.op, status="error", error=exc.to_dict()
            )
        except Exception as exc:  # noqa: BLE001 - a crash here would zero the whole plan
            results[op.id] = OperationResult(
                id=op.id,
                op=op.op,
                status="error",
                error=TFQLError(
                    ErrorCode.INVARIANT_FAILED,
                    f"unexpected failure in {op.op}: {exc}",
                ).to_dict(),
            )

    # Preserve the planner's original ordering in the response.
    ordered_results = [results[op.id] for op in plan.operations]
    succeeded = sum(1 for r in ordered_results if r.succeeded)
    status = "ok" if succeeded == len(ordered_results) else "error" if succeeded == 0 else "partial"
    return PlanResult(request_id=plan.request_id, status=status, results=ordered_results)
