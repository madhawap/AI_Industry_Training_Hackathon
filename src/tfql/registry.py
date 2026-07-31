"""Operation registry.

The registry is the closed vocabulary that Qwen plans against. An operation not
registered here cannot be requested, which is what turns "don't invent metrics"
from a prompt instruction into a structural guarantee.

It also generates the planner-facing catalogue, so the prompt and the executable
code can never drift apart.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from .errors import ErrorCode, TFQLError
from .evidence import OpOutput


class Args(BaseModel):
    """Base class for operation argument models.

    ``extra="forbid"`` is what produces UNKNOWN_ARGUMENT: a hallucinated
    parameter is rejected at plan-validation time, before any data is touched,
    while there is still budget to repair the call.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


OpFn = Callable[[Any, Any], OpOutput]


@dataclass(frozen=True, slots=True)
class OperationSpec:
    name: str
    args_model: type[Args]
    fn: OpFn
    summary: str
    datasets: tuple[str, ...]


_REGISTRY: dict[str, OperationSpec] = {}


def register(
    name: str,
    args_model: type[Args],
    *,
    summary: str,
    datasets: tuple[str, ...],
) -> Callable[[OpFn], OpFn]:
    """Decorator registering an operation under its dotted TFQL name."""

    def decorator(fn: OpFn) -> OpFn:
        if name in _REGISTRY:
            raise RuntimeError(f"operation {name!r} is already registered")
        _REGISTRY[name] = OperationSpec(
            name=name,
            args_model=args_model,
            fn=fn,
            summary=summary,
            datasets=datasets,
        )
        return fn

    return decorator


def get(name: str) -> OperationSpec:
    """Look up an operation, raising UNKNOWN_OPERATION with the valid names."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise TFQLError(
            ErrorCode.UNKNOWN_OPERATION,
            f"unknown operation {name!r}",
            requested=name,
            available=sorted(_REGISTRY),
        ) from None


def names() -> list[str]:
    return sorted(_REGISTRY)


def all_specs() -> list[OperationSpec]:
    return [_REGISTRY[n] for n in sorted(_REGISTRY)]


def parse_args(spec: OperationSpec, raw: dict[str, Any]) -> Args:
    """Validate raw JSON arguments against an operation's schema.

    Pydantic's structured errors are translated into TFQL codes so the planner
    sees UNKNOWN_ARGUMENT vs INVALID_ARGUMENT rather than a stack trace.
    """
    try:
        return spec.args_model.model_validate(raw)
    except ValidationError as exc:
        problems = exc.errors()
        code = (
            ErrorCode.UNKNOWN_ARGUMENT
            if any(p["type"] == "extra_forbidden" for p in problems)
            else ErrorCode.INVALID_ARGUMENT
        )
        raise TFQLError(
            code,
            f"invalid arguments for {spec.name}: "
            + "; ".join(f"{'.'.join(str(x) for x in p['loc'])}: {p['msg']}" for p in problems),
            operation=spec.name,
            accepted=sorted(spec.args_model.model_fields),
        ) from None


def _resolve_refs(node: Any, defs: dict[str, Any]) -> Any:
    """Inline ``$ref`` targets so enum values survive into the catalogue.

    Pydantic emits enums as ``{"$ref": "#/$defs/HoldKind"}``. Left unresolved,
    the planner never learns that ``kind`` accepts only any_change/hike/cut, and
    invents a value that fails validation -- costing a repair round trip.
    """
    if isinstance(node, dict):
        if "$ref" in node:
            name = str(node["$ref"]).rsplit("/", 1)[-1]
            target = _resolve_refs(defs.get(name, {}), defs)
            merged = {k: v for k, v in node.items() if k != "$ref"}
            return {**target, **merged}
        return {k: _resolve_refs(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_refs(v, defs) for v in node]
    return node


def catalogue() -> list[dict[str, Any]]:
    """Machine-readable catalogue for the planner system prompt.

    Generated from the registry so the prompt cannot describe an operation that
    does not exist, or omit one that does. Enum references are inlined, because
    the tool schema types ``args`` as a free-form object -- this catalogue is
    the planner's only source of truth for what each operation accepts.
    """
    out = []
    for spec in all_specs():
        schema = spec.args_model.model_json_schema()
        defs = schema.get("$defs", {})
        properties = {
            name: _resolve_refs(prop, defs)
            for name, prop in schema.get("properties", {}).items()
        }
        out.append(
            {
                "op": spec.name,
                "summary": spec.summary,
                "datasets": list(spec.datasets),
                "args": properties,
                "required": schema.get("required", []),
            }
        )
    return out
