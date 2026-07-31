"""Component recall — the grader that mirrors a component-based judge.

Each expected component is checked independently against the prediction and
scored yes/no, so partial credit falls out naturally. Equivalence rules
(date formats, numeric tolerance) are config, because the exact tolerance
contract is not decided yet.

Reads only `rec.eval["components"]`. If the eventual contract calls that field
something else, the adapter maps it — this grader does not change.
"""

from __future__ import annotations

import re
from datetime import datetime

from ftpipe.registry import register

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_DATE_FORMATS = ["%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%d/%m/%Y", "%b %d, %Y", "%B %d, %Y"]


def _as_date(text: str):
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _dates_in(text: str) -> set:
    found = set()
    for match in re.finditer(r"\d{4}-\d{2}-\d{2}|\d{1,2} \w+ \d{4}|\w+ \d{1,2}, \d{4}", text):
        parsed = _as_date(match.group())
        if parsed:
            found.add(parsed)
    return found


def _matches(component: str, prediction: str, *, tolerance: float, date_equiv: bool) -> bool:
    comp = str(component).strip()
    pred = prediction.strip()

    # 1. plain case-insensitive containment
    if comp.lower() in pred.lower():
        return True

    # 2. date equivalence across formats
    if date_equiv:
        comp_date = _as_date(comp)
        if comp_date and comp_date in _dates_in(pred):
            return True

    # 3. numeric equivalence within tolerance (handles 0.1 vs 0.10 vs 10%)
    comp_nums = _NUM.findall(comp)
    if len(comp_nums) == 1:
        try:
            target = float(comp_nums[0])
        except ValueError:
            return False
        for candidate in _NUM.findall(pred):
            try:
                if abs(float(candidate) - target) <= tolerance:
                    return True
            except ValueError:
                continue
    return False


class ComponentMatch:
    name = "component_match"

    def __init__(self, cfg: dict):
        self.tolerance = float(cfg.get("numeric_tolerance", 0.0))
        self.date_equiv = bool(cfg.get("date_equivalence", True))

    def score(self, rec, prediction: str) -> dict[str, float]:
        components = rec.eval.get("components") or []
        if not components:
            return {}  # nothing to grade; excluded from the mean
        hits = [
            _matches(c, prediction, tolerance=self.tolerance, date_equiv=self.date_equiv)
            for c in components
        ]
        return {
            "component_recall": sum(hits) / len(hits),
            "all_components": float(all(hits)),
            "n_components": float(len(hits)),
        }


@register("grader", "component_match")
def build(cfg: dict) -> ComponentMatch:
    return ComponentMatch(cfg)
