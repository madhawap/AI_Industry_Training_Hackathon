"""Checkpoint selection: maximise one primary metric, subject to guardrails.

Guardrails exist because "best average score" is the wrong shipping rule when
one of the failure modes is confident fabrication. A checkpoint that scores
higher while inventing numbers must not win.

Also reports the *earliest* checkpoint within `tie_tolerance` of the best —
smaller adapters that reach the same quality earlier are usually the safer
ship, and the organisers' own baseline notes that an early checkpoint may
already be good enough.
"""

from __future__ import annotations

from ftpipe.registry import register


class Guardrailed:
    def __init__(self, cfg: dict):
        self.primary = cfg.get("primary", "component_recall")
        self.guardrails = cfg.get("guardrails", {}) or {}
        self.tie_tolerance = float(cfg.get("tie_tolerance", 0.005))
        self.prefer_earliest = bool(cfg.get("prefer_earliest_within_tolerance", True))

    def choose(self, metrics: list[dict]) -> dict:
        """metrics: [{"checkpoint": str, "step": int, "metrics": {name: value}}, ...]"""
        evaluated = []
        for entry in metrics:
            values = entry["metrics"]
            violations = []
            for name, rule in self.guardrails.items():
                value = values.get(name)
                if value is None:
                    violations.append(f"{name}: not measured")
                    continue
                if "max" in rule and value > rule["max"]:
                    violations.append(f"{name}={value:.4g} > max {rule['max']}")
                if "min" in rule and value < rule["min"]:
                    violations.append(f"{name}={value:.4g} < min {rule['min']}")
            evaluated.append({**entry, "violations": violations,
                              "primary_value": values.get(self.primary)})

        eligible = [e for e in evaluated if not e["violations"] and e["primary_value"] is not None]
        if not eligible:
            return {
                "chosen": None,
                "reason": "no checkpoint passed the guardrails",
                "candidates": evaluated,
            }

        best = max(eligible, key=lambda e: e["primary_value"])
        chosen = best
        if self.prefer_earliest:
            near = [e for e in eligible if best["primary_value"] - e["primary_value"] <= self.tie_tolerance]
            chosen = min(near, key=lambda e: e.get("step", 0))

        return {
            "chosen": chosen["checkpoint"],
            "step": chosen.get("step"),
            "primary": self.primary,
            "primary_value": chosen["primary_value"],
            "reason": (
                f"best {self.primary}={best['primary_value']:.4g} at {best['checkpoint']}; "
                + (f"shipping earliest within {self.tie_tolerance} -> {chosen['checkpoint']}"
                   if chosen is not best else "best is also earliest")
            ),
            "candidates": evaluated,
        }


@register("policy", "guardrailed")
def build(cfg: dict) -> Guardrailed:
    return Guardrailed(cfg)
