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

    # backend.train() writes this next to the checkpoints when the backend
    # supports it (peft does; noop doesn't run real steps so has nothing to
    # log). Recorded in the manifest so downstream tools (the dashboard) find
    # it via `read_manifest("train")` rather than guessing the path.
    log_history_path = out_dir / "log_history.json"
    outputs = {"checkpoints": checkpoints}
    if log_history_path.is_file():
        outputs["log_history"] = str(log_history_path)

    run.write_manifest(
        "train",
        inputs={"rendered_train": rendered, "backend": train_cfg.get("backend"),
                "model_id": train_cfg.get("model_id"), "lora": train_cfg.get("lora"),
                "optim": train_cfg.get("optim"), "seq_len": train_cfg.get("seq_len"),
                "loss": train_cfg.get("loss")},
        outputs=outputs,
    )
    return {"backend": train_cfg.get("backend"), "n_checkpoints": len(checkpoints),
            "checkpoints": [c["step"] for c in checkpoints]}
