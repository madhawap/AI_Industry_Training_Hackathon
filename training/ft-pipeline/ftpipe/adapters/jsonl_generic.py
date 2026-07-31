"""Generic JSONL adapter — maps an arbitrary JSON/JSONL file into canonical
records via *configured field names*, so it needs no knowledge of the schema.

This is the bridge for real data before anyone has agreed a contract: point it
at a file, say which keys hold the question / context / answer / grading info,
and you get canonical records. When the real schema is fixed, either keep using
this with a settled field map or write a purpose-built adapter.

Config:
    adapter:
      name: jsonl_generic
      path: /path/to/questions.json        # .json (list) or .jsonl
      task_field: difficulty               # optional; falls back to task_default
      task_default: default
      target_field: answer                 # required
      input_fields: [question, query_context]   # copied into `inputs` verbatim
      eval_field: grading_components       # optional -> eval.components
      group_field: null                    # optional -> meta.group_key
      meta_fields: [difficulty, question_type]  # optional -> meta.<field>, verbatim

`task_field` is singular (one stratification label), so `meta_fields` exists
for everything else worth slicing metrics by later without overloading
`task` -- e.g. `evaluate.breakdown_by` (see stages/evaluate.py) reads
`rec.meta` for exactly these keys.
"""

from __future__ import annotations

import json
from pathlib import Path

from ftpipe.registry import register
from ftpipe.schema import Record, SchemaError, make_id


def _load_rows(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict):  # {"questions": [...]} style
        for value in data.values():
            if isinstance(value, list):
                return value
        raise SchemaError(f"{path}: object contains no list of records")
    return data


@register("adapter", "jsonl_generic")
def load(cfg: dict) -> list[Record]:
    path = Path(cfg["path"])
    if not path.is_file():
        raise FileNotFoundError(f"adapter.path does not exist: {path}")

    target_field = cfg.get("target_field")
    if not target_field:
        raise SchemaError("adapter.target_field must be set (which key holds the gold answer)")
    input_fields = cfg.get("input_fields") or []
    if not input_fields:
        raise SchemaError("adapter.input_fields must list at least one key to feed the prompt")

    task_field = cfg.get("task_field")
    task_default = cfg.get("task_default", "default")
    eval_field = cfg.get("eval_field")
    group_field = cfg.get("group_field")
    meta_fields = cfg.get("meta_fields") or []

    rows = _load_rows(path)
    records = []
    for i, row in enumerate(rows):
        if target_field not in row:
            raise SchemaError(f"{path}[{i}]: missing target_field {target_field!r}; keys are {sorted(row)}")
        inputs = {k: row[k] for k in input_fields if k in row}
        if not inputs:
            raise SchemaError(f"{path}[{i}]: none of input_fields {input_fields} present; keys are {sorted(row)}")

        evaluation = {}
        if eval_field and eval_field in row:
            evaluation["components"] = row[eval_field]

        records.append(
            Record(
                id=make_id(str(path), i, row.get(target_field)),
                task=str(row.get(task_field, task_default)) if task_field else task_default,
                target=str(row[target_field]),
                inputs=inputs,
                eval=evaluation,
                meta={
                    "source": str(path),
                    "row_index": i,
                    # Without a group field every row is its own group, which is
                    # the safe default (no accidental leakage, just less pooling).
                    "group_key": str(row.get(group_field)) if group_field and group_field in row else f"row:{i}",
                    **{f: row[f] for f in meta_fields if f in row},
                },
            )
        )
    return records
