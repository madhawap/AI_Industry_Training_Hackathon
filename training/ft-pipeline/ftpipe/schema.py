"""The canonical record — the one shape the whole pipeline agrees on.

Everything upstream (data sources) adapts *into* this; everything downstream
(prompting, training, grading) reads *out of* it. Deliberately minimal: the
fields we are confident about are typed, and the ones that depend on decisions
we have not made yet are opaque dicts.

`inputs` and `eval` are opaque ON PURPOSE. When the tool-result schema and the
grading contract are finally decided, only an adapter, a renderer, and a grader
need to change. Nothing in curate/train/predict/select ever looks inside them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass
class Record:
    # --- committed structure -------------------------------------------------
    id: str  # stable, content-derived; see `make_id`
    task: str  # slice label, e.g. "rba_single_fact" — used for stratification
    target: str  # the gold output text the model should produce

    # --- deliberately opaque -------------------------------------------------
    inputs: dict[str, Any] = field(default_factory=dict)
    # Whatever the eventual prompt needs. The renderer is the only code that may
    # inspect this. Today it might be {"question", "context"}; tomorrow anything.

    eval: dict[str, Any] = field(default_factory=dict)
    # Grader-specific payload, e.g. {"components": [...]} or {"reference": "..."}.
    # The grader is the only code that may inspect this.

    meta: dict[str, Any] = field(default_factory=dict)
    # Provenance + grouping. `group_key` matters: curate splits on it so that
    # near-duplicate records cannot straddle train/val and inflate the score.

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Record":
        known = {"id", "task", "target", "inputs", "eval", "meta"}
        unknown = set(d) - known
        if unknown:
            raise SchemaError(f"unknown field(s) {sorted(unknown)}; canonical fields are {sorted(known)}")
        return Record(
            id=d["id"],
            task=d["task"],
            target=d["target"],
            inputs=d.get("inputs", {}),
            eval=d.get("eval", {}),
            meta=d.get("meta", {}),
        )


class SchemaError(ValueError):
    """Raised when a record violates the canonical shape."""


def make_id(*parts: Any) -> str:
    """Content-derived id, so the same logical example keeps the same id across
    regenerations and two runs can be compared record-by-record."""
    blob = "\x1f".join(json.dumps(p, sort_keys=True, ensure_ascii=False) for p in parts)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def validate(rec: Record) -> Record:
    """Structural checks only — we never assert on the *contents* of `inputs`
    or `eval`, because that is exactly what we have not decided yet."""
    if not rec.id:
        raise SchemaError("record has empty id")
    if not rec.task:
        raise SchemaError(f"{rec.id}: empty task label (needed for stratification)")
    if not isinstance(rec.target, str) or not rec.target.strip():
        raise SchemaError(f"{rec.id}: target must be a non-empty string")
    for name in ("inputs", "eval", "meta"):
        if not isinstance(getattr(rec, name), dict):
            raise SchemaError(f"{rec.id}: {name} must be a dict")
    return rec


def read_jsonl(path) -> list[Record]:
    records = []
    with open(path) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(validate(Record.from_dict(json.loads(line))))
            except (json.JSONDecodeError, SchemaError) as exc:
                raise SchemaError(f"{path}:{line_no}: {exc}") from exc
    return records


def write_jsonl(records: Iterable[Record], path) -> int:
    count = 0
    with open(path, "w") as fh:
        for rec in records:
            fh.write(validate(rec).to_json() + "\n")
            count += 1
    return count
