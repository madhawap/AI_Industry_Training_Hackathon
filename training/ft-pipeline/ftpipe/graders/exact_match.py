"""Exact / normalised match against `rec.target`. The dumbest possible grader,
useful as the smoke-test grader for the walking skeleton and as a sanity floor."""

from __future__ import annotations

import re

from ftpipe.registry import register


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().rstrip("."))


class ExactMatch:
    name = "exact_match"

    def __init__(self, cfg: dict):
        self.normalise = bool(cfg.get("normalise", True))

    def score(self, rec, prediction: str) -> dict[str, float]:
        a, b = (prediction, rec.target)
        if self.normalise:
            a, b = _norm(a), _norm(b)
        return {"exact_match": float(a == b)}


@register("grader", "exact_match")
def build(cfg: dict) -> ExactMatch:
    return ExactMatch(cfg)
