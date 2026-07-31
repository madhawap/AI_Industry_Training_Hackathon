"""Hyperparameter sweep driver for the ft-pipeline.

Loops the full `ftpipe.cli run` over a small grid of hyperparameters, each
trial in its own `run_id` (so runs never collide -- `run_id: auto` in a base
config resolves to `name`, which is constant, so every trial must override
`run_id` explicitly or they'd all write into the same directory and clobber
each other). No changes to the pipeline itself: this only composes
`--set key=value` overrides, the same mechanism `ftpipe.cli run --set ...`
already exposes.

Each trial is a fresh subprocess (`python -m ftpipe.cli run ...`), not an
in-process call into `ftpipe.cli.main`, so the registry's module-import cache
and argparse state never leak between trials -- each trial is exactly the
command you'd type by hand, which is also what makes it independently
reproducible.

After every trial, reads `runs/<run_id>/select/chosen.json` (the pipeline's
own checkpoint-selection decision) and prints a ranked summary. A trial that
errors (including one an earlier --set produced an invalid config for) is
recorded as failed and excluded from the ranking rather than crashing the
whole sweep.

USAGE (this only prints the plan and estimated cost by default -- see --run):

    python3 sweep.py --config config/generated.yaml
    python3 sweep.py --config config/generated.yaml --grid my_grid.json
    python3 sweep.py --config config/generated.yaml --run          # actually launches training

Each trial trains a real LoRA adapter (or your configured backend). This is a
GPU-cost action -- --run is required to actually launch anything; without it,
sweep.py only prints the resolved trial list and exits, exactly the same
"tell me the commands, don't launch them" caution as everywhere else this
model of interaction has been used in this repo.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_GRID = [
    # (label suffix, overrides applied on top of --config)
    {"train.lora.rank": 16, "train.lora.alpha": 32},
    {"train.lora.rank": 32, "train.lora.alpha": 64},   # matches config/generated.yaml's own default
    {"train.lora.rank": 64, "train.lora.alpha": 128},
    {"train.optim.lr": 1.0e-4},
    {"train.optim.lr": 2.0e-5},
    {"train.lora.init_lora_weights": True},            # zero-init baseline, to check PiSSA actually helps
]


def _trial_run_id(base_name: str, index: int, overrides: dict) -> str:
    tag = "-".join(f"{k.rsplit('.', 1)[-1]}{v}".replace(".", "p") for k, v in overrides.items())
    return f"sweep-{base_name}-{index:02d}-{tag}"[:80]


def _run_trial(config_path: str, run_id: str, overrides: dict, conda_env: str, dry_run: bool) -> dict:
    cmd = ["conda", "run", "-n", conda_env, "python", "-m", "ftpipe.cli", "run",
           "--config", config_path, "--set", f"run_id={run_id}"]
    for key, value in overrides.items():
        cmd += ["--set", f"{key}={json.dumps(value) if isinstance(value, str) else value}"]

    print(f"\n{'[DRY RUN] ' if dry_run else ''}{' '.join(cmd)}")
    if dry_run:
        return {"run_id": run_id, "overrides": overrides, "status": "dry_run"}

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        return {"run_id": run_id, "overrides": overrides, "status": "failed",
                "returncode": proc.returncode}

    chosen_path = Path("runs") / run_id / "select" / "chosen.json"
    if not chosen_path.is_file():
        return {"run_id": run_id, "overrides": overrides, "status": "no_select_output"}
    decision = json.loads(chosen_path.read_text())
    return {"run_id": run_id, "overrides": overrides, "status": "ok",
            "primary": decision.get("primary"), "primary_value": decision.get("primary_value"),
            "improvement_over_base": decision.get("improvement"), "beat_base": decision.get("beat_base"),
            "chosen_checkpoint": decision.get("chosen")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="base config, e.g. config/generated.yaml")
    parser.add_argument("--grid", help="JSON file: a list of {dotted.key: value} override dicts. Defaults to a small built-in grid.")
    parser.add_argument("--conda-env", default="ft-pipeline")
    parser.add_argument("--run", action="store_true",
                         help="actually launch each trial. Without this flag, only prints the resolved "
                              "trial list and commands (a GPU-cost action needs an explicit opt-in).")
    parser.add_argument("--out", default="sweep_results.json")
    args = parser.parse_args()

    grid = json.loads(Path(args.grid).read_text()) if args.grid else DEFAULT_GRID
    base_name = Path(args.config).stem

    print(f"{len(grid)} trials planned against {args.config}:")
    for i, overrides in enumerate(grid):
        print(f"  {i:02d}. {overrides}")
    if not args.run:
        print("\n(dry run -- pass --run to actually launch training for each trial above)")

    results = []
    for i, overrides in enumerate(grid):
        run_id = _trial_run_id(base_name, i, overrides)
        results.append(_run_trial(args.config, run_id, overrides, args.conda_env, dry_run=not args.run))

    Path(args.out).write_text(json.dumps(results, indent=2))

    ranked = sorted(
        (r for r in results if r.get("status") == "ok" and r.get("primary_value") is not None),
        key=lambda r: r["primary_value"], reverse=True,
    )
    print(f"\n{'='*70}\nResults written to {args.out}\n{'='*70}")
    if not args.run:
        return
    if not ranked:
        print("No trial produced a usable select/chosen.json (all dry-run, failed, or still running).")
        return
    print(f"{'run_id':<45} {'primary_value':>14} {'vs base':>10}")
    for r in ranked:
        improvement = r.get("improvement_over_base")
        print(f"{r['run_id']:<45} {r['primary_value']:>14.4f} {improvement:>+10.4f}" if improvement is not None
              else f"{r['run_id']:<45} {r['primary_value']:>14.4f} {'n/a':>10}")
    best = ranked[0]
    print(f"\nBest: {best['run_id']} ({best['overrides']}) -> {best['primary']}={best['primary_value']:.4f}")


if __name__ == "__main__":
    main()
