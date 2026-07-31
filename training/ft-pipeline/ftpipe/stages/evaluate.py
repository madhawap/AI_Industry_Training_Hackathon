"""evaluate: predictions x graders -> metrics.json

Fast and CPU-only, so grader definitions can be changed and re-run without
regenerating a single token. Metrics are also broken down per `task`, because
an average hides which slice regressed.

`evaluate.breakdown_by` (a list of `rec.meta` key names, e.g.
`[difficulty, question_type]`) adds the same per-slice breakdown for any
adapter-supplied meta field -- `task` is one committed stratification label,
but a fine-tune can regress on "hard questions" or "asx.max_drawdown
questions" specifically without regressing on average, and only meta fields
the adapter actually populated (see adapters/jsonl_generic.py's `meta_fields`)
can be sliced this way.
"""

from __future__ import annotations

import collections
import json
import statistics

from ftpipe import registry
from ftpipe.artifacts import Run
from ftpipe.schema import read_jsonl


def _mean_by(bucket: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
    return {
        key: {m: round(statistics.fmean(v), 4) for m, v in metrics.items() if v}
        for key, metrics in bucket.items()
    }


def run(cfg: dict, run: Run) -> dict:
    eval_cfg = cfg.get("evaluate", {}) or {}
    grader_names = eval_cfg.get("graders") or ["exact_match"]
    graders = [registry.get("grader", name)(eval_cfg.get(name, {}) or {}) for name in grader_names]
    breakdown_fields = eval_cfg.get("breakdown_by") or []

    predict_manifest = run.read_manifest("predict")
    split = predict_manifest["inputs"]["split"]
    canonical = {rec.id: rec for rec in read_jsonl(run.read_manifest("curate")["outputs"][split]["path"])}

    results = []
    for variant in predict_manifest["outputs"]["predictions"]:
        totals: dict[str, list[float]] = collections.defaultdict(list)
        per_task: dict[str, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
        # One bucket per configured meta field, e.g. per_breakdown["difficulty"]["hard"]["component_recall"].
        per_breakdown: dict[str, dict[str, dict[str, list[float]]]] = {
            field: collections.defaultdict(lambda: collections.defaultdict(list)) for field in breakdown_fields
        }

        with open(variant["path"]) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                rec = canonical.get(row["id"])
                if rec is None:
                    continue
                for grader in graders:
                    for metric, value in grader.score(rec, row["prediction"]).items():
                        totals[metric].append(value)
                        per_task[rec.task][metric].append(value)
                        for field in breakdown_fields:
                            slice_key = str(rec.meta.get(field, "unknown"))
                            per_breakdown[field][slice_key][metric].append(value)

        summary = {m: round(statistics.fmean(v), 4) for m, v in totals.items() if v}
        results.append({
            "checkpoint": variant["checkpoint"],
            "step": variant["step"],
            "label": variant["label"],
            "metrics": summary,
            "per_task": _mean_by(per_task),
            "per_breakdown": {field: _mean_by(buckets) for field, buckets in per_breakdown.items()},
            "seconds_per_item": variant.get("seconds_per_item"),
        })

    results.sort(key=lambda r: r["step"])
    path = run.stage_dir("evaluate") / "metrics.json"
    path.write_text(json.dumps(results, indent=2))

    run.write_manifest(
        "evaluate",
        inputs={"graders": grader_names, "split": split, "breakdown_by": breakdown_fields},
        outputs={"metrics": str(path)},
        extra={"results": results},
    )
    return {"graders": grader_names, "results": [
        {"label": r["label"], **r["metrics"]} for r in results
    ]}
