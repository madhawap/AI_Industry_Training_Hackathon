"""Command line: run the whole pipeline or any single stage.

    python -m ftpipe.cli run     --config config/skeleton.yaml
    python -m ftpipe.cli stage   render --config config/skeleton.yaml
    python -m ftpipe.cli run     --config config/skeleton.yaml --set train.backend=peft
    python -m ftpipe.cli plugins

The `python ftpipe/cli.py ...` spelling works too — see the sys.path note below.
Either way, run it from the ft-pipeline directory: `--config` and the adapter's
`path` are resolved relative to the working directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# Support `python ftpipe/cli.py ...` alongside `python -m ftpipe.cli ...`.
# Invoked as a script, Python puts THIS file's directory (ftpipe/) on sys.path
# rather than its parent, so the `from ftpipe import ...` lines below would die
# with `ModuleNotFoundError: No module named 'ftpipe'`. Under -m the package is
# already importable and __package__ is set, so this is a no-op.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from ftpipe import config as cfg_mod
    from ftpipe import registry
    from ftpipe.artifacts import Run, config_hash
    from ftpipe.stages import ORDER, RUNNERS
except ModuleNotFoundError as exc:
    # Almost always the wrong interpreter — the system python3 has none of the
    # deps. Say so, instead of leaving a bare "No module named 'yaml'".
    print(
        f"error: cannot import {exc.name!r} — this looks like the wrong interpreter.\n"
        f"  running: {sys.executable}\n"
        f"  ft-pipeline needs its own env:\n"
        f"    conda run -n ft-pipeline python -m ftpipe.cli ...\n"
        f"  (see README 'Setup' if that env does not exist yet)",
        file=sys.stderr,
    )
    raise SystemExit(1) from None

BOLD, GREEN, YELLOW, RED, NC = "\033[1m", "\033[0;32m", "\033[1;33m", "\033[0;31m", "\033[0m"


def _run_id(cfg: dict) -> str:
    """The run directory must be STABLE across invocations of the same run.

    It is derived from `name` alone, never from the whole config: stages are
    meant to be runnable one at a time (`stage render`, then `stage train`),
    and hashing the full config meant a single `--set` between commands sent
    the second stage to a different directory, where the first stage's output
    did not exist. The exact config is still recorded — in config.snapshot.yaml
    and in every stage manifest — so provenance does not depend on the path.
    """
    explicit = cfg.get("run_id")
    if explicit and explicit != "auto":
        return str(explicit)
    name = cfg.get("name")
    return str(name) if name else f"run-{config_hash(cfg)}"


def _load(args) -> dict:
    cfg = cfg_mod.load(args.config, *(args.overlay or []))
    return cfg_mod.apply_overrides(cfg, args.set)


def _execute(cfg: dict, stages: list[str]) -> int:
    run = Run(_run_id(cfg), cfg)
    print(f"{BOLD}run:{NC} {run.run_id}   {BOLD}dir:{NC} {run.root}")
    for stage in stages:
        print(f"\n{BOLD}{GREEN}== {stage}{NC}")
        try:
            result = RUNNERS[stage](cfg, run)
        except cfg_mod.UndecidedError as exc:
            print(f"{YELLOW}  ⏸ undecided:{NC} {exc}")
            return 2
        except Exception as exc:  # noqa: BLE001
            print(f"{RED}  ✗ {stage} failed:{NC} {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
            return 1
        print(json.dumps(result, indent=2, default=str))
    print(f"\n{BOLD}{GREEN}✓ done{NC}  ({run.root})")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ftpipe")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("run", "stage"):
        p = sub.add_parser(name)
        if name == "stage":
            p.add_argument("stages", nargs="+", choices=ORDER)
        p.add_argument("--config", required=True)
        p.add_argument("--overlay", action="append")
        p.add_argument("--set", action="append", default=[])

    sub.add_parser("plugins")

    args = parser.parse_args(argv)

    if args.command == "plugins":
        for kind in ("adapter", "renderer", "grader", "policy", "backend"):
            print(f"{BOLD}{kind}{NC}: {', '.join(registry.available(kind))}")
        return 0

    cfg = _load(args)
    stages = ORDER if args.command == "run" else args.stages
    return _execute(cfg, stages)


if __name__ == "__main__":
    sys.exit(main())
