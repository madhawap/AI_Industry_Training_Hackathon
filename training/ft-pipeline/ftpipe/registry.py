"""Name -> implementation lookup.

Every pluggable piece (adapter, renderer, grader, policy, train backend) is
registered under a string name and selected from config. Swapping the prompt
format or the grader is a config edit, not a code change.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

_REGISTRY: dict[str, dict[str, Any]] = {
    "adapter": {},
    "renderer": {},
    "grader": {},
    "policy": {},
    "backend": {},
}

# Modules to import so their @register decorators run. Add new implementations
# here (or ship them as plugins and import before calling `get`).
_MODULES = [
    "ftpipe.adapters.stub",
    "ftpipe.adapters.jsonl_generic",
    "ftpipe.renderers.chat",
    "ftpipe.graders.exact_match",
    "ftpipe.graders.component_match",
    "ftpipe.graders.format_health",
    "ftpipe.graders.reference_overlap",
    "ftpipe.graders.llm_judge",
    "ftpipe.policies.guardrailed",
    "ftpipe.backends",
]

_loaded = False


def register(kind: str, name: str) -> Callable:
    if kind not in _REGISTRY:
        raise KeyError(f"unknown plugin kind {kind!r}; expected one of {sorted(_REGISTRY)}")

    def decorator(obj):
        _REGISTRY[kind][name] = obj
        return obj

    return decorator


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    for module in _MODULES:
        importlib.import_module(module)
    _loaded = True


def get(kind: str, name: str) -> Any:
    _ensure_loaded()
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        available = sorted(_REGISTRY.get(kind, {}))
        raise KeyError(f"no {kind} named {name!r}; available: {available}") from None


def available(kind: str) -> list[str]:
    _ensure_loaded()
    return sorted(_REGISTRY[kind])
