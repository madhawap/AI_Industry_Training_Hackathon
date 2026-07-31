"""Training + generation backends.

Two implementations behind one interface:
  noop  — writes checkpoint stubs, no GPU. Keeps the eight stages runnable
          anywhere (CI, laptop, a busy GPU box) so the pipeline can be
          exercised end-to-end before any real training is possible.
  peft  — real LoRA via transformers + peft.

The LoRA specifics live entirely here. Swapping to a different trainer (a NeMo
container, say) means writing a third backend and changing `train.backend` in
config — no other stage knows how training happens.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ftpipe.registry import register


# ---------------------------------------------------------------------------
# noop
# ---------------------------------------------------------------------------
@register("backend", "noop")
class NoopBackend:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def train(self, rendered_path: Path, out_dir: Path) -> list[dict]:
        steps = int(self.cfg.get("optim", {}).get("steps", 100))
        every = int(self.cfg.get("checkpoint_every", 20))
        checkpoints = []
        for step in range(every, steps + 1, every):
            ckpt = out_dir / f"checkpoint-{step}"
            ckpt.mkdir(parents=True, exist_ok=True)
            (ckpt / "NOOP").write_text(f"stub checkpoint at step {step}\n")
            checkpoints.append({"checkpoint": str(ckpt), "step": step})
        return checkpoints

    def generate(self, checkpoint: str | None, prompts: list[dict], **kw) -> list[str]:
        # Echo the last user message so the pipeline produces gradeable text.
        out = []
        for item in prompts:
            user = [m for m in item["messages"] if m["role"] == "user"]
            out.append(user[-1]["content"][:200] if user else "")
        return out


# ---------------------------------------------------------------------------
# peft (real LoRA)
# ---------------------------------------------------------------------------
@register("backend", "peft")
class PeftBackend:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_id = cfg.get("model_id")
        if not self.model_id:
            raise ValueError("train.model_id must be set for the peft backend")

    # -- helpers -------------------------------------------------------------
    def _tokenizer(self):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def _encode(self, tok, messages: list[dict], max_len: int):
        """Full sequence with the prompt masked out, so loss is computed on the
        assistant turn only — we are teaching output style, not re-teaching the
        input distribution."""
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_text = tok.apply_chat_template(messages, tokenize=False)

        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"][:max_len]
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}

    # -- training ------------------------------------------------------------
    def train(self, rendered_path: Path, out_dir: Path) -> list[dict]:
        import json as _json

        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

        from ftpipe import registry as _registry

        lora_cfg = self.cfg.get("lora", {})
        optim = self.cfg.get("optim", {})
        loss_cfg = self.cfg.get("loss", {}) or {}
        max_len = int(self.cfg["seq_len"])

        tok = self._tokenizer()
        rows = [json.loads(l) for l in Path(rendered_path).read_text().splitlines() if l.strip()]
        encoded = [self._encode(tok, r["messages"], max_len) for r in rows]
        dataset = Dataset.from_list(encoded)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=getattr(torch, self.cfg.get("dtype", "bfloat16")),
            device_map=self.cfg.get("device_map", "auto"),
        )
        model.config.use_cache = False
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora_cfg.get("rank", 32)),
                lora_alpha=int(lora_cfg.get("alpha", 64)),
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_cfg.get("target_modules", "all-linear"),
                # DoRA (Liu et al. 2024) decomposes the update into magnitude + direction,
                # consistently >= vanilla LoRA at the same rank/cost.
                use_dora=bool(lora_cfg.get("use_dora", False)),
                # rsLoRA (Kalajdzievski 2023) fixes LoRA's rank scaling so higher ranks
                # don't blow up the update magnitude.
                use_rslora=bool(lora_cfg.get("use_rslora", False)),
                # init_lora_weights: True is the reference implementation's zero-init
                # (B=0, so LoRA is a no-op before training). "pissa" (Meng et al. 2024)
                # initialises A/B from the base weight's principal singular
                # vectors/values instead, which the paper reports converges faster and
                # ends up at a better optimum than zero-init at the same rank/cost --
                # also accepts "eva", "olora", "gaussian", or True/False for the prior
                # defaults; see the peft LoraConfig docstring for the full set.
                init_lora_weights=lora_cfg.get("init_lora_weights", True),
            ),
        )
        model.print_trainable_parameters()

        # Loss is a named, swappable plugin (default "masked_ce" == the plain
        # masked cross-entropy transformers already computes internally when
        # unconfigured; see ftpipe/losses.py for the two opt-in research knobs).
        loss_fn = _registry.get("loss", loss_cfg.get("name", "masked_ce"))(loss_cfg)

        every = int(self.cfg.get("checkpoint_every", 20))
        # report_to=["tensorboard"] (set in config, not hardcoded) writes event
        # files incrementally *during* training -- unlike log_history.json
        # below, which is only written once trainer.train() returns -- so
        # `tensorboard --logdir <logging_dir>` gives a live-updating loss
        # curve while a run is in progress. Needs the `tensorboard` package;
        # empty list (the old behaviour) needs nothing extra.
        report_to = self.cfg.get("report_to", [])
        logging_dir = str(Path(out_dir) / "tensorboard") if report_to else None
        args = TrainingArguments(
            output_dir=str(out_dir),
            max_steps=int(optim.get("steps", 100)),
            per_device_train_batch_size=int(optim.get("micro_batch", 1)),
            gradient_accumulation_steps=int(optim.get("grad_accum", 8)),
            learning_rate=float(optim.get("lr", 5e-5)),
            warmup_steps=int(optim.get("warmup", 10)),
            lr_scheduler_type=optim.get("scheduler", "cosine"),
            logging_steps=max(1, every // 4),
            logging_dir=logging_dir,
            save_strategy="steps",
            save_steps=every,
            save_total_limit=None,
            bf16=True,
            report_to=report_to,
            seed=int(self.cfg.get("seed", 0)),
            # NEFTune (Jain et al. 2023): noises the embeddings during training only —
            # free instruction-tuning quality bump, no inference cost. null disables it.
            neftune_noise_alpha=optim.get("neftune_noise_alpha"),
        )
        LossTrainer = _make_loss_trainer()
        trainer = LossTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=_PadCollator(tok.pad_token_id),
            loss_fn=loss_fn,
            loss_tokenizer=tok,
        )
        trainer.train()

        # Training-progress curve: Trainer's own step-by-step log, otherwise
        # discarded once the process exits. Written next to the checkpoints so
        # a run's loss-over-steps is reproducible from disk, not just the
        # console this run happened to print to.
        (Path(out_dir) / "log_history.json").write_text(
            _json.dumps(trainer.state.log_history, indent=2, default=str)
        )

        checkpoints = []
        for path in sorted(Path(out_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
            checkpoints.append({"checkpoint": str(path), "step": int(path.name.split("-")[1])})
        return checkpoints

    # -- generation ----------------------------------------------------------
    def generate(self, checkpoint: str | None, prompts: list[dict],
                 max_new_tokens: int = 128, temperature: float = 0.0,
                 batch_size: int = 8) -> list[str]:
        import torch
        from transformers import AutoModelForCausalLM

        tok = self._tokenizer()
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=getattr(torch, self.cfg.get("dtype", "bfloat16")),
            device_map=self.cfg.get("device_map", "auto"),
        )
        if checkpoint:  # None => the untuned base model, i.e. the comparison baseline
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, checkpoint)
        model.eval()

        texts = [
            tok.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True)
            for item in prompts
        ]
        outputs: list[str] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=tok.pad_token_id,
                )
            for i in range(len(batch)):
                new_tokens = generated[i][enc["input_ids"].shape[1]:]
                outputs.append(tok.decode(new_tokens, skip_special_tokens=True))
        return outputs


def _make_loss_trainer():
    """`transformers.Trainer` subclass whose `compute_loss` routes through a
    named `ftpipe.losses` plugin instead of the model class's built-in loss.
    Built lazily (not a module-level class) so importing `ftpipe.backends`
    never requires transformers to be installed -- only actually training
    does, same as every other transformers/peft/torch import in this file."""
    from transformers import Trainer

    class _LossTrainer(Trainer):
        def __init__(self, *args, loss_fn=None, loss_tokenizer=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._loss_fn = loss_fn
            self._loss_tokenizer = loss_tokenizer

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = self._loss_fn(outputs.logits, labels, self._loss_tokenizer)
            return (loss, outputs) if return_outputs else loss

    return _LossTrainer


class _PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch

        width = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            pad = width - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_id] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def get_backend(cfg: dict):
    from ftpipe.registry import get

    name = cfg.get("backend") or "noop"
    return get("backend", name)(cfg)
