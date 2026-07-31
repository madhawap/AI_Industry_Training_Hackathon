"""render: canonical splits -> model-ready messages, plus a LENGTH REPORT.

The length report exists so `train.seq_len` is measured rather than guessed —
the organisers' baseline warns that long sequences OOM, and the honest way to
pick a length is percentiles over the real rendered prompts.
"""

from __future__ import annotations

import json

from ftpipe import config, registry
from ftpipe.artifacts import Run, file_hash
from ftpipe.schema import read_jsonl


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def run(cfg: dict, run: Run) -> dict:
    renderer_cfg = cfg.get("renderer", {}) or {}
    name = config.require(cfg, "renderer.name", "which prompt renderer to use")
    renderer = registry.get("renderer", name)(renderer_cfg)

    curate = run.read_manifest("curate")["outputs"]
    out_dir = run.stage_dir("render")

    # Optional real tokenizer for an accurate length report; falls back to a
    # chars/4 estimate so this stage never hard-requires a model download.
    tokenizer = None
    model_id = config.get(cfg, "train.model_id", None)
    if model_id and cfg.get("render", {}).get("measure_with_tokenizer", True):
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as exc:  # noqa: BLE001 - degrade, don't block
            print(f"  ! tokenizer unavailable ({exc.__class__.__name__}); estimating lengths")

    outputs, report = {}, {}
    for split, info in curate.items():
        records = read_jsonl(info["path"])
        path = out_dir / f"{split}.jsonl"
        lengths, prompt_lengths = [], []
        with open(path, "w") as fh:
            for rec in records:
                rendered = renderer.render(rec, purpose="train")
                fh.write(json.dumps(rendered, ensure_ascii=False) + "\n")

                if tokenizer is not None:
                    full = tokenizer.apply_chat_template(rendered["messages"], tokenize=False)
                    prompt_only = tokenizer.apply_chat_template(
                        [m for m in rendered["messages"] if m["role"] != "assistant"],
                        tokenize=False, add_generation_prompt=True,
                    )
                    lengths.append(len(tokenizer(full, add_special_tokens=False)["input_ids"]))
                    prompt_lengths.append(len(tokenizer(prompt_only, add_special_tokens=False)["input_ids"]))
                else:
                    chars = sum(len(m["content"]) for m in rendered["messages"])
                    lengths.append(chars // 4)
                    prompt_lengths.append(chars // 5)

        outputs[split] = {"path": str(path), "n": len(records), "sha": file_hash(path)}
        report[split] = {
            "unit": "tokens" if tokenizer is not None else "estimated_tokens",
            "total": {"p50": _percentile(lengths, 50), "p95": _percentile(lengths, 95),
                      "max": max(lengths) if lengths else 0},
            "prompt": {"p50": _percentile(prompt_lengths, 50), "p95": _percentile(prompt_lengths, 95),
                       "max": max(prompt_lengths) if prompt_lengths else 0},
        }

    (out_dir / "length_report.json").write_text(json.dumps(report, indent=2))
    run.write_manifest(
        "render",
        inputs={"renderer": name, "renderer_cfg": renderer_cfg, "tokenizer": model_id},
        outputs=outputs,
        extra={"length_report": report},
    )

    suggested = max(r["total"]["p95"] for r in report.values())
    return {"length_report": report,
            "suggested_seq_len": _round_up(suggested),
            "note": "set train.seq_len from this rather than guessing"}


def _round_up(value: int) -> int:
    for candidate in (256, 512, 768, 1024, 1536, 2048, 4096):
        if value <= candidate:
            return candidate
    return value
