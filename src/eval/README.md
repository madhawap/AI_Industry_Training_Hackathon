# Finance hackathon evals

Offline evaluation harness for the finance hackathon agent, built against the
**real** datasets in `AI_Industry_Training_Hackathon/data set/` (RBA cash-rate
decisions, the 18-company ASX price basket, and the ~220k-article AFR
corpus). Every number in `dataset.py` was computed from those files with the
same deterministic logic the agent's own tools use -- nothing here is a
placeholder or an invented figure.

It's a sibling of `evals/` (the Chinook music-store eval suite), reusing its
patterns (dataset -> agent -> LangSmith-scored experiment) but simpler: no
reusable-evaluator registration, no online guardrails, no per-customer
context. Dataset upload, evaluators, and the run script live at the top
level of this folder rather than under a package, since Python module names
can't contain the hyphen in `evals-hackathon`.

## Architecture

This follows the two-model split required by the challenge brief
(`AI_Industry_Training_Hackathon/README.md` and `Participant_Package/Challenge_Brief.md`):

```
question -> [Qwen "agent-brain": plans + calls tools] -> [tool runtime: real data] -> [Nemotron: synthesizes answer]
```

- **`finance_data.py`** -- stdlib-only loaders/caches for the three real
  datasets (CSV for RBA, per-company JSONL for ASX, ~85 monthly JSONL files
  for AFR). AFR is the expensive one (~780MB, ~10-15s to parse); it's
  cached in-process after the first call.
- **`finance_tools.py`** -- 17 deterministic tools over that data: 5 RBA
  (extreme rate, longest gap, largest hike/cut cycle, rate-as-of-date,
  changes-in-period), 7 ASX (price extremes, single-day move, total
  return, basket rank, top performer, volume, drawdown), 5 AFR (corpus
  stats, busiest day, whole-word pattern count, longest headline,
  articles-on-date). AFR pattern matching follows the brief's non-negotiable
  method: case-insensitive, `\bwhole word\b`, matched once per record across
  `HEADLINE + SUBHEAD + INTRO + TEXT` combined.
- **`finance_agent.py`** -- a 2-node LangGraph graph:
  - `planner`: Qwen (`create_agent`, same machinery as the project root's
    `agent.py` / `simple-agent.py`) bound to all 17 tools, running its own
    plan/call-tools/observe loop.
  - `synthesize`: extracts the verified tool-call trace from the planner's
    messages (discarding Qwen's own prose) and asks the fine-tuned Nemotron
    model to write the final answer from that evidence alone. Controlled by
    `DOMAIN_PREDICT_MODE` (see below).
- **`dataset.py`** -- 9 `FinanceExample` rows (3 RBA, 3 ASX, 3 AFR),
  `difficulty`/`question`/`answer`/`grading_components` in the same shape
  as the `mock_questions.json` sample you shared, plus real `expected_tools`
  / `expected_tool_args`.
- **`evaluators_offline.py`** -- `tool_trajectory` (right tool(s), right
  args, in order) and `grading_components_present` (fraction of
  `grading_components` substrings found in the final answer -- partial
  credit, since most answers state several facts at once). Fully generic;
  needed no changes when the tools/dataset went from placeholders to real.
- **`run_eval_offline.py`** -- uploads/refreshes the dataset in LangSmith
  and runs `aevaluate` with the two evaluators above.
- **`generate_100_questions.py`** -- generated the 101-question set at
  `AI_Industry_Training_Hackathon/data set/evals/questions.json`, every
  answer/component computed the same way as `dataset.py`'s 9 examples.
  Re-run it to regenerate that file if the underlying data changes.
- **`api.py`** -- a minimal `/health` + `/query` HTTP wrapper around
  `finance_agent.graph`, for testing over HTTP exactly like the real
  submission contract (see `llm_judge_grader.py --endpoint`).
- **`llm_judge_grader.py`** + **`GRADER_README.md`** -- self-grades answers
  against the organizers' own hidden-question rubric (component-by-component
  LLM-as-judge, partial credit, the 60s/300s timing penalty) before you
  submit. Full writeup in `GRADER_README.md`.

## Seeing it run -- tracing and visualization

- **LangSmith tracing**: `.env` already has `LANGSMITH_TRACING=true`, so
  every model call and tool call inside both the planner and synthesis
  stages is traced automatically -- no extra wiring. `run_eval_offline.py`
  prints the experiment; open it in LangSmith to see per-example scores
  *and* the full trace (which tools Qwen called, with what args, what they
  returned, and what Nemotron/Qwen wrote as the final answer).
- **LangGraph Studio**: `finance-agent` is registered in the project root's
  `langgraph.json`, alongside the existing `agent` and `chinook-agent`
  graphs. Run `langgraph dev` from the repo root and open Studio to step
  through the `planner -> synthesize` graph interactively, independent of
  the LangSmith experiment run.

## Environment variables

All optional -- everything falls back to this repo's existing `MODEL_NAME` /
`OPENAI_BASE_URL` / `OPENAI_API_KEY`, so the graph runs today against
whatever gateway `.env` already points at (currently `gpt-5-mini`), standing
in for Qwen until the real `agent-brain` alias is reachable.

| Variable | Purpose |
|---|---|
| `AGENT_BRAIN_MODEL_NAME` / `AGENT_BRAIN_BASE_URL` / `AGENT_BRAIN_API_KEY` | Qwen planner. Point these at the supplied `agent-brain` LiteLLM alias when it's reachable. |
| `FINETUNED_MODEL_NAME` / `FINETUNED_BASE_URL` / `FINETUNED_API_KEY` | Fine-tuned Nemotron, served by vLLM. Required only when `DOMAIN_PREDICT_MODE=llm`. |
| `DOMAIN_PREDICT_MODE` | `"mock"` (default) -- skip Nemotron, use Qwen's own reply as the answer, so the pipeline runs before the adapter is served. `"llm"` -- call the real fine-tuned model. Matches the cluster bootstrap default described in the brief; **flip to `llm` before the real evaluation run**, per the brief's checklist. |
| `FINANCE_DATASET_NAME` | Override the LangSmith dataset name (default `finance-hackathon-eval`). |

## Running it

Requires the model gateway credentials in `.env` (already present at the
project root) and the project's dependencies installed (`.venv/` at the
repo root already has everything: `pip install -r requirements.txt`).
Verified working end-to-end in this session (real agent runs, real
LangSmith-free grading, real HTTP endpoint) against the model gateway
already configured in `.env`. From the project root:

```bash
.venv/bin/python evals-hackathon/run_eval_offline.py          # LangSmith experiment (needs LANGSMITH_API_KEY)
.venv/bin/python evals-hackathon/llm_judge_grader.py           # self-grade against the org rubric, no LangSmith needed
```

Optional flags for `run_eval_offline.py`: `--dataset-name`,
`--experiment-prefix`, `--max-concurrency`. For `llm_judge_grader.py`, see
`GRADER_README.md`.

The first run touching AFR data will take longer than later ones: the
loader parses all ~85 files once (~10-15s) and caches the result
in-process for the rest of that run/service lifetime.

## What's real vs. what's still a gap

**Real and verified:**
- All three loaders parse the actual supplied files (no mocking, no
  sampling) -- `finance_data.py` has no fallback/placeholder path.
- All 17 tools compute their answers deterministically from that data;
  every figure in `dataset.py` and the generated 101-question set was
  produced by calling the tool functions directly against the real files.
- AFR pattern counting follows the brief's exact non-negotiable method
  (whole-word, case-insensitive, combined fields, once per record).
- The full 101-question set scored **99.0% (301/304 components), 95/101
  perfect** via `llm_judge_grader.py` against the in-process graph; a random
  10-question sample scored **100%** via both `--endpoint` (against
  `api.py`) and `--answers-file` (pre-recorded) modes.
- `api.py`'s dataset pre-warming (see `GRADER_README.md` ->
  "A concurrency bug this caught") fixed a real 300s+ stall under 3-way
  concurrent AFR requests -- worth carrying into any submission service
  that lazy-loads its data the same way.

**Known gaps -- next steps toward the actual submission:**
1. **Nemotron isn't wired up yet.** `DOMAIN_PREDICT_MODE` defaults to
   `mock`, so the "final answer" is currently Qwen's own tool-grounded
   reply. Set `FINETUNED_*` and `DOMAIN_PREDICT_MODE=llm` once the
   fine-tuned adapter is serving, then re-run `llm_judge_grader.py
   --endpoint` against the real service to see how synthesis quality holds up.
2. **`api.py` is a testing double, not the submission service.** It proves
   the `/health` + `/query` contract works against this repo's
   `finance_agent.py`, but the actual submission service
   (`cognitivo_prep/src/main.py`) is a separate, more complete
   implementation (real Qwen + LiteLLM + tfql) -- point
   `llm_judge_grader.py --endpoint` at that once its backend is reachable.
3. **The one remaining content miss** (see `GRADER_README.md`'s worked
   examples): the largest-cuts-cycle question drops the before/after rate
   values in synthesis. Worth a prompt tweak in `finance_agent.py`'s
   `SYNTHESIS_SYSTEM_PROMPT` if it recurs.
