#!/usr/bin/env python3
"""
Merge a PEFT/LoRA adapter into its base model and write a standalone
HuggingFace model directory that vLLM can serve.

Not meant to be run directly on the host — there is no torch there. Use
./merge-adapter.sh, which runs this inside the NeMo container.

Why merging rather than --enable-lora: the checkpoint sets use_dora=true,
and vLLM's runtime LoRA path rejects DoRA adapters outright. Merging folds
the adapter into the base weights, after which it is just a normal model.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# The tokenizer lives with the base model, not the checkpoint — a Trainer
# checkpoint of an adapter saves neither tokenizer nor config. vLLM needs
# both next to the weights, so they are copied across after the merge.
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
]

# Sampled before and after the merge to prove the adapter actually landed.
# A LoRA that silently no-ops still saves a perfectly valid model directory,
# so "it ran without error" is not evidence that anything changed.
PROBE_WEIGHT = "model.layers.0.self_attn.q_proj.weight"


def log(msg):
    print(f"[merge] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model directory")
    ap.add_argument("--adapter", required=True, help="adapter/checkpoint directory")
    ap.add_argument("--out", required=True, help="output directory for merged model")
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        help="merge dtype; must match the base model's to avoid a silent "
        "precision downgrade (base config here says bfloat16)",
    )
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty --out")
    args = ap.parse_args()

    base, adapter, out = Path(args.base), Path(args.adapter), Path(args.out)
    dtype = getattr(torch, args.dtype)

    if not (base / "config.json").is_file():
        sys.exit(f"error: {base}/config.json not found — not a model directory")
    if not (adapter / "adapter_config.json").is_file():
        sys.exit(f"error: {adapter}/adapter_config.json not found — not a PEFT adapter")
    if out.exists() and any(out.iterdir()) and not args.force:
        sys.exit(f"error: {out} exists and is not empty (pass --force to overwrite)")

    cfg = json.loads((adapter / "adapter_config.json").read_text())
    log(f"adapter: {cfg.get('peft_type')} r={cfg.get('r')} alpha={cfg.get('lora_alpha')} "
        f"dora={cfg.get('use_dora')} rslora={cfg.get('use_rslora')}")
    log(f"adapter was trained against: {cfg.get('base_model_name_or_path')}")

    # PEFT warns about unknown keys if the adapter was written by a newer
    # version than the one installed. Harmless when the unknown keys are all
    # null (they are experimental variants that were not used here), but worth
    # surfacing rather than burying, so it is left to print normally.

    log(f"loading base model from {base} on CPU as {args.dtype} (~16GB RSS for 8B)")
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=dtype,
        device_map=None,          # CPU: leaves the GPU free for a running vLLM
        low_cpu_mem_usage=True,
    )

    before = model.state_dict()[PROBE_WEIGHT].clone()

    log(f"applying adapter from {adapter}")
    model = PeftModel.from_pretrained(model, adapter, torch_dtype=dtype)

    log("merging (merge_and_unload)")
    model = model.merge_and_unload()

    after = model.state_dict()[PROBE_WEIGHT]
    delta = (after.float() - before.float()).abs().max().item()
    if delta == 0.0:
        sys.exit(
            f"error: {PROBE_WEIGHT} is byte-identical after merging.\n"
            "       The adapter did not apply — check that its target_modules "
            "match this base architecture."
        )
    log(f"verified: {PROBE_WEIGHT} changed, max |delta| = {delta:.6f}")

    out.mkdir(parents=True, exist_ok=True)
    log(f"saving merged model to {out}")
    model.save_pretrained(out, safe_serialization=True)

    log("copying tokenizer from base model")
    tok = AutoTokenizer.from_pretrained(base)
    tok.save_pretrained(out)
    # save_pretrained does not carry every auxiliary file across versions;
    # backfill anything it missed so the merged dir is self-contained.
    for name in TOKENIZER_FILES:
        src, dst = base / name, out / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
            log(f"  copied {name}")

    missing = [f for f in ("config.json", "tokenizer_config.json") if not (out / f).is_file()]
    if missing:
        sys.exit(f"error: merged directory is missing {missing}")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    log(f"done — {out} ({size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
