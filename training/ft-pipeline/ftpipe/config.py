"""Config loading. One YAML file is the only place decisions live.

`null` is a first-class state: it means "not decided yet". Stages that cannot
proceed without a decision raise `UndecidedError` naming the exact key, rather
than silently guessing. That is how the pipeline stays runnable end-to-end
before the contract exists, without hiding the gaps.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class UndecidedError(RuntimeError):
    """A stage needs a config value that is still null."""


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load(path, *overlays) -> dict:
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    for overlay in overlays:
        if overlay:
            cfg = _deep_merge(cfg, yaml.safe_load(Path(overlay).read_text()) or {})
    return cfg


def get(cfg: dict, dotted: str, default: Any = "__raise__") -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default == "__raise__":
                raise KeyError(f"config key {dotted!r} is missing")
            return default
        node = node[part]
    return node


def require(cfg: dict, dotted: str, why: str = "") -> Any:
    """Fetch a value that must have been decided. Null -> loud, specific error."""
    value = get(cfg, dotted, None)
    if value is None:
        hint = f" — {why}" if why else ""
        raise UndecidedError(
            f"config key {dotted!r} is still null{hint}.\n"
            f"This is an undecided contract detail: set it in your config and re-run."
        )
    return value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """CLI `--set a.b=c` overrides, parsed as YAML scalars."""
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out
