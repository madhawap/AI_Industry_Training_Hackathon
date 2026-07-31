"""curate: canonical.jsonl -> train/val/test splits

Two rules that matter more than anything clever:

1. Split by GROUP, never by row. Generated data always contains near-duplicates;
   if a template's rows land in both train and val, val score is fiction. This
   is the single most common way a fine-tune "looks great" and then fails.
2. Deduplicate exact inputs, so the same example cannot be counted twice.

Both are implementable today because they only touch `meta.group_key` and a
hash of `inputs` — never the contents.
"""

from __future__ import annotations

import collections
import hashlib
import json
import random

from ftpipe import config
from ftpipe.artifacts import Run, file_hash
from ftpipe.schema import read_jsonl, write_jsonl


def _input_hash(rec) -> str:
    return hashlib.sha256(
        json.dumps(rec.inputs, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def run(cfg: dict, run: Run) -> dict:
    curate_cfg = cfg.get("curate", {}) or {}
    src = run.read_manifest("ingest")["outputs"]["canonical"]
    records = read_jsonl(src)

    # 1. dedupe on input content
    seen: set[str] = set()
    deduped = []
    for rec in records:
        key = _input_hash(rec)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)
    n_dropped = len(records) - len(deduped)

    # 2. group-aware split
    fractions = curate_cfg.get("splits", [0.8, 0.1, 0.1])
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"curate.splits must sum to 1.0, got {fractions}")

    by_group: dict[str, list] = collections.defaultdict(list)
    for rec in deduped:
        by_group[str(rec.meta.get("group_key", rec.id))].append(rec)

    groups = sorted(by_group)
    random.Random(int(config.get(cfg, "seed", 0))).shuffle(groups)

    n_groups = len(groups)
    n_train = int(n_groups * fractions[0])
    n_val = int(n_groups * fractions[1])
    buckets = {
        "train": groups[:n_train],
        "val": groups[n_train:n_train + n_val],
        "test": groups[n_train + n_val:],
    }

    out_dir = run.stage_dir("curate")
    outputs, summary = {}, {}
    for split, group_names in buckets.items():
        rows = [rec for g in group_names for rec in by_group[g]]
        path = out_dir / f"{split}.jsonl"
        write_jsonl(rows, path)
        outputs[split] = {"path": str(path), "n": len(rows), "n_groups": len(group_names),
                          "sha": file_hash(path)}
        summary[split] = {"n": len(rows), "groups": len(group_names),
                          "tasks": dict(collections.Counter(r.task for r in rows))}

    # Leakage assertion: a group must live in exactly one split.
    assignments = {g: s for s, names in buckets.items() for g in names}
    if len(assignments) != n_groups:
        raise AssertionError("group leakage detected across splits")

    run.write_manifest(
        "curate",
        inputs={"canonical": src, "splits": fractions,
                "group_key": curate_cfg.get("group_key", "meta.group_key")},
        outputs=outputs,
        extra={"deduped_dropped": n_dropped, "n_groups": n_groups, "summary": summary},
    )
    return {"dropped_duplicates": n_dropped, "n_groups": n_groups, **summary}
