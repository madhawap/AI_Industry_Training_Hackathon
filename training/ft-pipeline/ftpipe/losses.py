"""Named, swappable training losses -- the "loss" plugin kind, same pattern
as adapters/renderers/graders/backends: `train.loss.name` in config selects
an implementation, so changing the loss is a config edit, not a code change.

`masked_ce.MaskedCrossEntropy` makes explicit something that was previously
implicit: `PeftBackend._encode` already masks the prompt out of `labels`
(-100), so `transformers`' default internal loss already computes plain
masked cross-entropy. Pulling it out here does two things:

  1. Makes it a named, independently testable unit instead of "whatever the
     base model class's forward() happens to do."
  2. Adds two research-backed knobs on top, both off by default so the
     un-configured behaviour is bit-for-bit the same masked CE as before:

     - `label_smoothing` -- standard SFT regulariser against overconfident
       next-token predictions (Szegedy et al. 2016; routine in modern SFT
       recipes, e.g. Llama/Nemotron post-training reports).
     - `fact_token_weight` -- upweight tokens that contain a digit. The
       challenge's component-based grading and this project's own
       `component_match` grader check numbers and dates specifically; a
       wrong digit costs the same points as a wrong sentence but is one
       token out of an average ~300-token answer, so plain mean CE gives it
       negligible gradient signal relative to the surrounding prose tokens.
"""

from __future__ import annotations

from ftpipe.registry import register


class MaskedCrossEntropy:
    """Next-token cross-entropy over positions where `labels != -100` only
    (the assistant turn -- see `PeftBackend._encode`).

    `fact_token_weight=1.0, label_smoothing=0.0` (the defaults) is exactly
    the mean masked CE `transformers` already computes internally; this class
    only changes behaviour when configured to.
    """

    name = "masked_ce"

    def __init__(self, cfg: dict):
        self.label_smoothing = float(cfg.get("label_smoothing", 0.0))
        self.fact_token_weight = float(cfg.get("fact_token_weight", 1.0))
        self._digit_token_ids: set[int] | None = None  # built lazily, cached per tokenizer

    def _digit_token_ids_for(self, tokenizer) -> set[int]:
        if self._digit_token_ids is None:
            vocab_size = len(tokenizer)
            tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
            self._digit_token_ids = {
                i for i, t in enumerate(tokens) if t is not None and any(c.isdigit() for c in t)
            }
        return self._digit_token_ids

    def __call__(self, logits, labels, tokenizer):
        import torch
        import torch.nn.functional as F

        # Standard causal shift: the logit at position i predicts the token at i+1.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
            label_smoothing=self.label_smoothing,
        ).view(shift_labels.shape)

        valid = (shift_labels != -100).float()
        weights = valid
        if self.fact_token_weight != 1.0:
            digit_ids = self._digit_token_ids_for(tokenizer)
            flat = shift_labels.clamp(min=0).reshape(-1).tolist()
            digit_mask = torch.tensor(
                [1.0 if t in digit_ids else 0.0 for t in flat],
                dtype=per_token.dtype, device=per_token.device,
            ).view(shift_labels.shape)
            weights = valid * torch.where(
                digit_mask.bool(), torch.full_like(valid, self.fact_token_weight), torch.ones_like(valid)
            )

        denom = weights.sum().clamp(min=1.0)
        return (per_token * weights).sum() / denom


@register("loss", "masked_ce")
def build(cfg: dict) -> MaskedCrossEntropy:
    return MaskedCrossEntropy(cfg)
