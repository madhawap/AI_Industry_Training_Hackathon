"""ingest: raw source -> canonical.jsonl

The only stage that knows where data comes from. Everything after this point
sees canonical records and nothing else.
"""

from __future__ import annotations

import collections

from ftpipe import config, registry
from ftpipe.artifacts import Run, file_hash
from ftpipe.schema import write_jsonl


def run(cfg: dict, run: Run) -> dict:
    adapter_cfg = cfg.get("adapter", {}) or {}
    name = config.require(
        cfg, "adapter.name",
        "which source adapter to use (try 'stub' to exercise the pipeline without real data)",
    )
    loader = registry.get("adapter", name)

    records = list(loader(adapter_cfg))
    if not records:
        raise RuntimeError(f"adapter {name!r} produced no records")

    out_path = run.stage_dir("ingest") / "canonical.jsonl"
    count = write_jsonl(records, out_path)

    tasks = collections.Counter(r.task for r in records)
    groups = len({r.meta.get("group_key") for r in records})
    run.write_manifest(
        "ingest",
        inputs={"adapter": name, "adapter_cfg": adapter_cfg},
        outputs={"canonical": str(out_path), "sha": file_hash(out_path), "n": count},
        extra={"tasks": dict(tasks), "n_groups": groups},
    )
    return {"n": count, "tasks": dict(tasks), "n_groups": groups, "path": str(out_path)}
