"""train: rendered train split -> LoRA checkpoints.

All the LoRA detail lives in the backend. This stage only enforces the one
thing worth being strict about: seq_len must be a decision, not a default.
"""

from __future__ import annotations

from pathlib import Path

from ftpipe import config
from ftpipe.artifacts import Run
from ftpipe.backends import get_backend


def run(cfg: dict, run: Run) -> dict:
    train_cfg = dict(cfg.get("train", {}) or {})
    train_cfg.setdefault("seed", config.get(cfg, "seed", 0))

    if train_cfg.get("seq_len") is None:
        report = run.read_manifest("render").get("length_report", {})
        raise config.UndecidedError(
            "train.seq_len is null. Run `render` and set it from the measured p95 "
            f"rather than guessing.\nMeasured: "
            + ", ".join(f"{k}: p95={v['total']['p95']}" for k, v in report.items())
        )

    rendered = run.read_manifest("render")["outputs"]["train"]["path"]
    out_dir = run.stage_dir("train")

    backend = get_backend(train_cfg)
    checkpoints = backend.train(Path(rendered), out_dir)

    run.write_manifest(
        "train",
        inputs={"rendered_train": rendered, "backend": train_cfg.get("backend"),
                "model_id": train_cfg.get("model_id"), "lora": train_cfg.get("lora"),
                "optim": train_cfg.get("optim"), "seq_len": train_cfg.get("seq_len")},
        outputs={"checkpoints": checkpoints},
    )
    return {"backend": train_cfg.get("backend"), "n_checkpoints": len(checkpoints),
            "checkpoints": [c["step"] for c in checkpoints]}
