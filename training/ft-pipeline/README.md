# ft-pipeline

A staged, contract-agnostic fine-tuning pipeline. Built so that the parts we
*have* decided (LoRA, checkpoint selection, splitting, evidence) can be built
and tested **before** the parts we have not (tool-result schema, prompt format,
grading contract, serving system prompt).

Method is **LoRA** — the base model's weights are frozen and a small adapter
(tens of MB) is trained alongside them.

## Quick start

```bash
conda run -n ft-pipeline python -m ftpipe.cli plugins                       # what's pluggable
conda run -n ft-pipeline python -m ftpipe.cli run --config config/skeleton.yaml   # no GPU, no data
CUDA_VISIBLE_DEVICES=2 conda run -n ft-pipeline python -m ftpipe.cli run \
    --config config/nemotron.yaml                                          # real LoRA
```

Re-run one stage only (e.g. after changing a grader — no regeneration needed):

```bash
python -m ftpipe.cli stage evaluate select --config config/nemotron.yaml
```

Tests (no pytest needed) — includes the **contract-change drill**, which fails
if knowledge of the prompt format ever leaks outside `renderers/`:

```bash
conda run -n ft-pipeline python tests/test_pipeline.py    # 20/20
```

## The eight stages

| stage | in → out | decides |
|---|---|---|
| `ingest` | raw → `canonical.jsonl` | which source adapter |
| `curate` | canonical → `train/val/test` | dedupe, **group-aware** split |
| `render` | canonical → messages + **length report** | prompt format, seq-len budget |
| `train` | messages → LoRA checkpoints | backend, hyperparameters |
| `predict` | base **and** every checkpoint → predictions | decoding |
| `evaluate` | predictions × graders → metrics | what "good" means |
| `select` | metrics → chosen checkpoint + rationale | shipping policy |
| `export` | → evidence bundle | packaging |

Every stage is file-in / file-out, so any stage can be re-run alone. `predict`
(slow, GPU) is split from `evaluate` (fast, CPU) on purpose: grader definitions
change often, and re-grading stored predictions costs seconds.

## The canonical record

The one shape everything agrees on. `inputs` and `eval` are **opaque dicts** —
that is where all the undecided contract detail hides:

```python
{"id", "task", "target",          # committed structure
 "inputs": {...},                 # renderer is the only reader
 "eval":   {...},                 # grader is the only reader
 "meta":   {...}}                 # provenance + group_key
```

When the schema lands, you write one adapter and one renderer. Curate, train,
predict, select and export never change.

## Three deliberate design choices

**One prompt builder.** `renderers/chat.py::build_messages` builds both training
text and live requests (`serving_payload`). Train/serve prompt skew is the most
common silent cause of "trained fine, worse in production"; sharing the builder
makes it structurally impossible rather than a discipline problem.

**Group-aware splitting.** `curate` splits on `meta.group_key`, never on rows.
Generated data is full of near-duplicates; letting a template straddle
train/val produces a val score that is fiction.

**Guardrailed selection.** `select` maximises a primary metric *subject to*
guardrails, so a checkpoint that scores higher while inventing numbers
(`hallucinated_number_rate`) cannot win. It also prefers the earliest
checkpoint within a tie tolerance.

## Undecided values are `null`, not defaults

`config.require()` raises `UndecidedError` naming the exact key instead of
guessing. `train.seq_len` is deliberately strict — it must come from the
measured p95 in `render`'s length report, because guessing it is where the OOM
risk lives.

## Models

`train.model_id` is the only line that changes:

- `unsloth/Llama-3.2-3B-Instruct` — same Llama-3 architecture and chat template
  as the hackathon target, ~6 GB bf16, fits beside other jobs. Use for iteration.
- `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` — the hackathon model (ungated),
  ~16 GB bf16. Needs a mostly-free GPU.

## LoRA/DoRA/rsLoRA/NEFTune, and the eval fallback chain

`backends.py`'s `peft` backend and the graders mirror the techniques exercised
in `../finetune_nemotron_updated.ipynb`, kept as config rather than hardcoded:

- `train.lora.use_dora` / `use_rslora` — DoRA (Liu et al. 2024) and rsLoRA
  (Kalajdzievski 2023), both `true` in `config/nemotron.yaml`.
- `train.optim.neftune_noise_alpha` — NEFTune (Jain et al. 2023) embedding
  noise, training-only, `null` disables it.
- `evaluate.graders` gained **`reference_overlap`** (token-F1 always;
  ROUGE-L/BLEU too if `rouge-score`/`sacrebleu` are installed) — the fallback
  that still scores something once `eval.components` is missing, same idea as
  the notebook's `HAS_GRADING_COMPONENTS` branch, just always-on rather than
  conditional, since `component_match` already scores `{}` (excluded from the
  mean) when there are no components.
- **`llm_judge`** — groundedness/correctness/concision via Anthropic or
  OpenAI, one call per prediction. Not in any default `graders` list (costs
  network + an API key); opt in with:
  ```yaml
  evaluate:
    graders: [component_match, format_health, llm_judge]
    llm_judge: {provider: anthropic, model: claude-sonnet-5}
  ```
  and set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` for `provider: openai`).

## Layout

```
ftpipe/
  schema.py      canonical record + validation
  registry.py    name -> implementation
  config.py      YAML + null-means-undecided
  artifacts.py   run dirs, manifests, provenance
  backends.py    noop | peft (LoRA lives here)
  adapters/      raw -> canonical      [swap when schema lands]
  renderers/     canonical -> prompt   [swap when contract lands]
  graders/       component_match, format_health, exact_match
  policies/      checkpoint selection
  stages/        the eight steps
config/          skeleton.yaml, nemotron.yaml
runs/<run_id>/   artifacts + manifests (git-ignored)
```

## Relationship to other work

Separate from `cognitivo_prep/training/finetune_nemotron.ipynb`, which does the
same job monolithically. This is the staged version; it can be moved under
`TeamSubmission/training/` unchanged when that repo is assembled.
