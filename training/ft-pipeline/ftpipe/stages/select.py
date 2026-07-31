"""select: metrics -> the checkpoint to ship, with a written rationale.

Model-selection rationale is itself a graded deliverable, so this stage writes
a human-readable report next to the machine-readable choice.
"""

from __future__ import annotations

import json

from ftpipe import registry
from ftpipe.artifacts import Run


def run(cfg: dict, run: Run) -> dict:
    select_cfg = cfg.get("select", {}) or {}
    policy = registry.get("policy", select_cfg.get("policy", "guardrailed"))(select_cfg)

    results = run.read_manifest("evaluate")["results"]
    tuned = [r for r in results if r["checkpoint"] is not None]
    base = next((r for r in results if r["checkpoint"] is None), None)

    decision = policy.choose(tuned)
    primary = decision.get("primary", select_cfg.get("primary", "component_recall"))

    if base is not None and decision.get("chosen"):
        base_value = base["metrics"].get(primary)
        chosen_value = decision.get("primary_value")
        if base_value is not None and chosen_value is not None:
            decision["base_value"] = base_value
            decision["improvement"] = round(chosen_value - base_value, 4)
            decision["beat_base"] = chosen_value > base_value

    out_dir = run.stage_dir("select")
    (out_dir / "chosen.json").write_text(json.dumps(decision, indent=2))
    (out_dir / "report.md").write_text(_report(decision, results, primary))

    run.write_manifest("select", inputs={"policy": select_cfg}, outputs={
        "chosen": str(out_dir / "chosen.json"), "report": str(out_dir / "report.md")})
    return {"chosen": decision.get("chosen"), "reason": decision.get("reason"),
            "improvement_over_base": decision.get("improvement"),
            "beat_base": decision.get("beat_base")}


def _report(decision: dict, results: list[dict], primary: str) -> str:
    lines = ["# Checkpoint selection", ""]
    lines.append(f"**Primary metric:** `{primary}`  ")
    lines.append(f"**Chosen:** `{decision.get('chosen')}`  ")
    lines.append(f"**Rationale:** {decision.get('reason')}")
    if decision.get("improvement") is not None:
        verdict = "improved on" if decision.get("beat_base") else "DID NOT beat"
        lines.append(f"\n**Base vs fine-tuned:** {decision['base_value']} -> "
                     f"{decision['primary_value']} ({decision['improvement']:+.4f}) — {verdict} the base model.")

    metric_names = sorted({m for r in results for m in r["metrics"]})
    lines += ["", "## All variants", "", "| variant | " + " | ".join(metric_names) + " | s/item |",
              "|---|" + "---|" * (len(metric_names) + 1)]
    for row in results:
        cells = [f"{row['metrics'].get(m, ''):}" for m in metric_names]
        lines.append(f"| {row['label']} | " + " | ".join(str(c) for c in cells)
                     + f" | {row.get('seconds_per_item', '')} |")

    rejected = [c for c in decision.get("candidates", []) if c.get("violations")]
    if rejected:
        lines += ["", "## Rejected by guardrails", ""]
        for cand in rejected:
            lines.append(f"- `{cand['label']}`: " + "; ".join(cand["violations"]))
    return "\n".join(lines) + "\n"
