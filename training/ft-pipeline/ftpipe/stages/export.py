"""export: the chosen adapter + the evidence bundle.

Assembles what a judge asks for — config, hyperparameters, metrics, selection
rationale, and a model card — from artifacts already produced, rather than
reconstructing them at the deadline.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ftpipe.artifacts import Run, git_revision


def run(cfg: dict, run: Run) -> dict:
    out_dir = run.stage_dir("export")
    chosen = json.loads((run.root / "select" / "chosen.json").read_text())
    metrics = json.loads((run.root / "evaluate" / "metrics.json").read_text())
    train_manifest = run.read_manifest("train")

    bundle = out_dir / "bundle"
    bundle.mkdir(exist_ok=True)

    copied = None
    bytes_copied = 0
    if chosen.get("chosen"):
        src = Path(chosen["chosen"])
        if src.is_dir():
            dst = bundle / "adapter"
            if dst.exists():
                shutil.rmtree(dst)
            # Ship ONLY the adapter. A trainer checkpoint directory also holds
            # optimizer/scheduler/RNG state — resumption data that is useless
            # for serving and an order of magnitude larger than the weights.
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                "optimizer.pt", "optimizer.bin", "scheduler.pt", "rng_state*.pth",
                "trainer_state.json", "training_args.bin", "global_step*", "*.distcp",
            ))
            copied = str(dst)
            bytes_copied = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())

    (bundle / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (bundle / "selection.json").write_text(json.dumps(chosen, indent=2))
    (bundle / "train_config.json").write_text(json.dumps(train_manifest["inputs"], indent=2))
    shutil.copy(run.root / "select" / "report.md", bundle / "selection_report.md")
    if (run.root / "render" / "length_report.json").is_file():
        shutil.copy(run.root / "render" / "length_report.json", bundle / "length_report.json")

    (bundle / "MODEL_CARD.md").write_text(_model_card(cfg, chosen, metrics, train_manifest, run))

    run.write_manifest("export", inputs={"chosen": chosen.get("chosen")},
                       outputs={"bundle": str(bundle), "adapter": copied,
                                "adapter_mb": round(bytes_copied / 1e6, 1)})
    return {"bundle": str(bundle), "adapter": copied,
            "adapter_mb": round(bytes_copied / 1e6, 1)}


def _model_card(cfg, chosen, metrics, train_manifest, run) -> str:
    inputs = train_manifest["inputs"]
    base = next((m for m in metrics if m["checkpoint"] is None), {})
    lines = [
        "# Model card", "",
        f"- **Base model:** `{inputs.get('model_id')}`",
        f"- **Method:** LoRA (`{json.dumps(inputs.get('lora'))}`)",
        f"- **Optimiser:** `{json.dumps(inputs.get('optim'))}`",
        f"- **Sequence length:** {inputs.get('seq_len')} (measured, see length_report.json)",
        f"- **Backend:** `{inputs.get('backend')}`",
        f"- **Chosen checkpoint:** `{chosen.get('chosen')}` (step {chosen.get('step')})",
        f"- **Run id:** `{run.run_id}`  ·  **code revision:** `{git_revision()}`",
        "", "## Task", "",
        "Given a question plus already-verified facts, produce the final answer text.",
        "The model does not plan, retrieve, or call tools.",
        "", "## Results", "",
        f"Base metrics: `{json.dumps(base.get('metrics', {}))}`", "",
        f"Selected: `{json.dumps({'metric': chosen.get('primary'), 'value': chosen.get('primary_value')})}`",
        "", "See `selection_report.md` for the full per-checkpoint table and guardrail rejections.",
        "", "## Reproducing", "",
        "```bash", "ftpipe run --config <config.yaml>", "```",
        "", "Every stage manifest records input hashes, config, and code revision.",
    ]
    return "\n".join(lines) + "\n"
