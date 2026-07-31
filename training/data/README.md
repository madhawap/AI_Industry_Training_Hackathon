# Generated training data

`generated_questions.jsonl` is produced by
[`src/data/generate_training_data.py`](../../src/data/generate_training_data.py). It is not
hand-written and not model-authored: every question is answered by actually running
`src.tfql`'s `execute_plan` — the same executor and operation registry the production agent
calls — against `warehouse.duckdb`, the DuckDB warehouse built from the real `data set/`
(RBA cash-rate decisions, ASX prices, AFR articles). Nothing in this file is invented; a wrong
number would come from a bug in an operation already covered by `src/tests/`, not from a model
guessing.

Regenerate after rebuilding the warehouse (see [`src/data/README.md`](../../src/data/README.md)
Quickstart):

```bash
python3 -m src.data.generate_training_data --warehouse ./warehouse.duckdb --out training/data/generated_questions.jsonl
```

## Categories

| category | what it is | how the answer is produced |
|---|---|---|
| `answerable` | Single- and cross-dataset questions across RBA, ASX and AFR. | The plan succeeds; the answer states exactly the fields the operation(s) returned. |
| `unanswerable` | Coverage gaps, unknown tickers, out-of-range dates (e.g. RBA decisions past 2021 when ASX/AFR end in 2021; a ticker not in the 18-ticker set; a date before a series starts). | The real operation is called and really fails (`DATE_OUTSIDE_COVERAGE`, `UNKNOWN_TICKER`, `NO_MATCHING_RECORDS` — see `tool_trace`); the answer states the refusal using the store's actual coverage bounds, never a guessed cutoff. |
| `extrapolation` | Prediction/forecast framing: future rates, future prices, "will X happen", "when will Y next occur". | The answer grounds itself in the last real observation the data contains (last known rate, last known close, a real historical pattern), then explicitly declines to invent a forecast — per the challenge brief's "state the limitation... instead of inventing a figure" rule. |

126 records as of the last generation run, across 29 template families (see `template_family`
below), roughly 55/46/25 easy/medium/hard and 111/15 single-/cross-dataset. Counts will change as
templates are added to the generator — rerun and re-read the script's stderr summary rather than
trusting a stale count here.

## Record shape

Each line matches the challenge brief's Required Response contract plus the
`public_questions.jsonl`-style grading metadata:

```jsonc
{
  "schema_version": "gen-1.0",
  "generation_method": "tfql_execute_plan_over_real_warehouse",
  "id": "GEN-ASX-drawdown-BHP.AX",
  "template_family": "GEN-ASX-drawdown",   // group key -- see below
  "category": "answerable",                // answerable | unanswerable | extrapolation
  "verification_status": "auto_generated", // not "verified": no human has read this one
  "difficulty": "hard",
  "datasets": ["ASX"],
  "dataset_scope": "single",               // single | cross
  "prompt": "What was BHP.AX's largest peak-to-trough drawdown, with the peak and trough dates?",
  "answer": "BHP.AX's largest peak-to-trough drawdown was -52.54%, ...",
  "reference_answer": "... (same as answer, kept for public_questions.jsonl-style consumers)",
  "steps": 1,
  "tool_trace": [{"tool": "execute_plan", "args": {"operations": [...]}, "result": {...}}],
  "derivation_methodology": "asx.max_drawdown(ticker=BHP.AX, basis=close).",
  "required_facts": ["BHP.AX's largest peak-to-trough drawdown was -52.54%, ..."],
  "grading": {"method": "component_based", "max_score": 10, "components": [...], "tolerance_note": "..."}
}
```

`tool_trace` is a real `PlanResult` dump (`bundle.model_dump(mode="json", exclude_none=True)`),
including real `status: "error"` entries and TFQL error codes for the `unanswerable` category —
not a synthesized placeholder.

`template_family` is the first three hyphen-separated tokens of `id` (e.g. both
`GEN-ASX-return-BHP.AX-2018` and `GEN-ASX-return-CBA.AX-2021` become `GEN-ASX-return`). It exists
because generated data is full of near-duplicate template shapes across sampled tickers/dates/
years; splitting a fine-tuning train/val/test set on rows rather than template families lets a
near-duplicate straddle the split and produces a validation score that doesn't mean anything.

## Feeding this into `training/ft-pipeline/`

[`training/ft-pipeline/config/generated.yaml`](../ft-pipeline/config/generated.yaml) points the
pipeline's `jsonl_generic` adapter at this file (`target_field: answer`,
`input_fields: [prompt, tool_trace, derivation_methodology]`, `eval_field: required_facts`,
`group_field: template_family`), and its `renderer.user_template` puts `prompt` + `tool_trace` +
`derivation_methodology` in the user turn and `answer` in the assistant turn — i.e. the same
"question + accumulated verified tool results -> answer" shape the challenge brief specifies for
Nemotron's synthesis step. Verified against this file's real output:

```bash
cd training/ft-pipeline
python3 -m ftpipe.cli stage ingest curate render --config config/generated.yaml
```

`ingest`/`curate` ran cleanly (126 rows, 29 groups, 107/11/8 train/val/test) and `render` produced
a length report (`suggested_seq_len: 1024`) without needing a GPU or model download. `train` /
`predict` need the heavier deps in `training/ft-pipeline/requirements.txt`, not exercised here.
