"""evaluate: predictions x graders -> metrics.json

Fast and CPU-only, so grader definitions can be changed and re-run without
regenerating a single token. Metrics are also broken down per `task`, because
an average hides which slice regressed.
"""

from __future__ import annotations

import collections
import json
import statistics

from ftpipe import registry
from ftpipe.artifacts import Run
from ftpipe.schema import read_jsonl


def run(cfg: dict, run: Run) -> dict:
    eval_cfg = cfg.get("evaluate", {}) or {}
    grader_names = eval_cfg.get("graders") or ["exact_match"]
    graders = [registry.get("grader", name)(eval_cfg.get(name, {}) or {}) for name in grader_names]

    predict_manifest = run.read_manifest("predict")
    split = predict_manifest["inputs"]["split"]
    canonical = {rec.id: rec for rec in read_jsonl(run.read_manifest("curate")["outputs"][split]["path"])}

    results = []
    for variant in predict_manifest["outputs"]["predictions"]:
        totals: dict[str, list[float]] = collections.defaultdict(list)
        per_task: dict[str, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))

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

        summary = {m: round(statistics.fmean(v), 4) for m, v in totals.items() if v}
        results.append({
            "checkpoint": variant["checkpoint"],
            "step": variant["step"],
            "label": variant["label"],
            "metrics": summary,
            "per_task": {t: {m: round(statistics.fmean(v), 4) for m, v in d.items() if v}
                         for t, d in per_task.items()},
            "seconds_per_item": variant.get("seconds_per_item"),
        })

    results.sort(key=lambda r: r["step"])
    path = run.stage_dir("evaluate") / "metrics.json"
    path.write_text(json.dumps(results, indent=2))

    run.write_manifest(
        "evaluate",
        inputs={"graders": grader_names, "split": split},
        outputs={"metrics": str(path)},
        extra={"results": results},
    )
    return {"graders": grader_names, "results": [
        {"label": r["label"], **r["metrics"]} for r in results
    ]}
