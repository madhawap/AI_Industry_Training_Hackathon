"""Format health — behavioural guardrails that need no reference answer.

These are the failure modes a component-based judge punishes hardest:
preamble ("I will use the retrieve tool to..."), hedging ("approximately",
"around"), and numbers that appear in the answer but nowhere in the inputs
(i.e. invented). Cheap to compute, and they make good selection guardrails.

`hallucinated_number_rate` is the important one: it is the direct measure of
"did the model state a figure that was not given to it".
"""

from __future__ import annotations

import json
import re

from ftpipe.registry import register

_NUM = re.compile(r"-?\d+(?:\.\d+)?")

_DEFAULT_HEDGES = [
    "approximately", "roughly", "around about", "it seems", "i think",
    "probably", "may have", "might have", "appears to be", "estimated",
]
_DEFAULT_PREAMBLES = [
    "i will", "i need to", "let me", "first, i", "the user is asking",
    "to answer this", "i'll use", "based on my search",
]


class FormatHealth:
    name = "format_health"

    def __init__(self, cfg: dict):
        self.hedges = [h.lower() for h in cfg.get("hedge_phrases", _DEFAULT_HEDGES)]
        self.preambles = [p.lower() for p in cfg.get("preamble_phrases", _DEFAULT_PREAMBLES)]
        self.max_chars = int(cfg.get("max_chars", 400))

    def score(self, rec, prediction: str) -> dict[str, float]:
        low = prediction.lower()

        # Numbers available to the model = everything in `inputs`, flattened.
        available = set(_NUM.findall(json.dumps(rec.inputs, default=str)))
        used = set(_NUM.findall(prediction))
        invented = {n for n in used if n not in available and _normalise(n, available) is None}

        return {
            "hedge_rate": float(any(h in low for h in self.hedges)),
            "preamble_rate": float(any(low.startswith(p) or f". {p}" in low for p in self.preambles)),
            "over_length_rate": float(len(prediction) > self.max_chars),
            "answer_chars": float(len(prediction)),
            "hallucinated_number_rate": float(bool(invented)),
        }


def _normalise(number: str, available: set[str]):
    """Treat 0.10 == 0.1 and 16.0 == 16 as available, not invented."""
    try:
        value = float(number)
    except ValueError:
        return None
    for candidate in available:
        try:
            if abs(float(candidate) - value) < 1e-9:
                return candidate
        except ValueError:
            continue
    return None


@register("grader", "format_health")
def build(cfg: dict) -> FormatHealth:
    return FormatHealth(cfg)
