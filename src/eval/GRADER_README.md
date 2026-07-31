# `llm_judge_grader.py` — self-grading against the organizers' own rubric

This is private, offline tooling: **we run the organizers' hidden-question
grading methodology on ourselves**, before submitting, so we know our real
score ahead of time. It is not part of the submitted agent and never touches
the submission's actual pipeline — see [Judge model independence](#judge-model-independence-important)
below.

It mirrors `AI_Industry_Training_Hackathon/Participant_Package/handout/03_scoring_and_examples.md`
exactly:

- Each question has one or more **components** — a specific fact worth some
  number of points (`expected_fact` + `points`).
- A judge model is given the question, the candidate answer, and **one**
  expected fact at a time, and replies YES or NO.
- Equivalent formatting is accepted (`"1,234"` == `"1234"`, `"Jan 2024"` ==
  `"2024-01"`, minor rephrasing that preserves meaning) — but a correct
  number in the wrong context, hedging language, or a refusal is **not**
  accepted.
- `hidden_question_score = sum(earned_points) / sum(max_points) * 100%`,
  with a 20% penalty on a question's earned points if the answer took longer
  than 60s, and 0 points past 300s (the organizers' exact timing rules).

## How scoring actually works

For every question:

1. Get an answer (see [three ways to supply answers](#three-ways-to-supply-the-answers-being-graded)).
2. For every grading component (`expected_fact` + `points`):
   - **Deterministic fast-path (safe NO only).** If the fact is a number or
     date and it does not appear *anywhere* in the answer, in any
     equivalent format, that's an unambiguous miss — scored `NO` with no
     LLM call. This never fast-paths to `YES`: the brief's own example
     ("there are 41 records in total, 20 of which are holds" being rejected
     as evidence for "20 increases") shows a present number can still be in
     the wrong context, so presence alone never proves a component. Only
     genuine *absence* is safe to decide without the judge.
   - **Everything else goes to the LLM judge** — present numbers/dates
     (context must be checked) and every non-numeric fact.
3. `earned = sum(points for components marked YES)`, then the 60s/300s
   timing penalty is applied.
4. Totals across all graded questions produce the overall percentage.

Run `--llm-only` to disable the deterministic fast-path and judge every
component with the LLM, matching the organizers' pure-LLM methodology
exactly (useful for a final sanity check before submitting; the hybrid mode
is faster and used by default).

## Three ways to supply the answers being graded

| Mode | Flag | What it does |
|---|---|---|
| In-process (default) | *(none)* | Calls `finance_agent.graph` directly in this process. |
| Live endpoint | `--endpoint URL` | POSTs `{"question": ...}` to `URL/query` and grades whatever comes back. Works against `api.py` (this repo's demo service) or the real submission service (`cognitivo_prep/src/main.py`) once its model backend is reachable — this is how you test a **deployed** answer, including one synthesized by a real fine-tuned Nemotron, not just the `DOMAIN_PREDICT_MODE=mock` fallback. |
| Recorded answers | `--answers-file PATH` | Grades pre-recorded `{id\|question, answer, steps, tool_trace}` rows with **no** agent or endpoint call at all — e.g. answers you already collected, or a `curl .../query` response pasted into a file. Instant, since there's nothing to run. |

`--endpoint` and `--answers-file` are mutually exclusive; omit both to use
the in-process graph.

### Worked example: endpoint mode

Start the demo service (wraps `finance_agent.graph` in the same `/health` +
`/query` contract the real submission uses):

```bash
.venv/bin/uvicorn api:app --app-dir evals-hackathon --port 8001
```

It pre-warms all three datasets (RBA/ASX/AFR) at startup before `/health`
returns 200 — see [A concurrency bug this caught](#a-concurrency-bug-this-caught).

```bash
curl -s http://127.0.0.1:8001/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the lowest cash-rate target in the RBA dataset and when did it first take effect?"}'
# {"answer":"...0.1%...2020-11-04...16 decision records...","steps":1,
#  "tool_trace":[{"tool":"rba_extreme_rate","args":{"mode":"lowest"},"result":"..."}]}
```

Grade a random sample of 10 questions against it:

```bash
.venv/bin/python evals-hackathon/llm_judge_grader.py \
  --questions "AI_Industry_Training_Hackathon/data set/evals/questions.json" \
  --endpoint http://127.0.0.1:8001 \
  --sample 10 --seed 42
```

Result from this exact run:

```
component score   32.0/32  (100.0%)
perfect answers   10/10
over 60s          0 (each loses 20% of its earned points)
wall clock        52.8s at concurrency 3
```

### Worked example: recorded-answers mode

Take the same 10 questions' already-collected answers (any JSON list shaped
like `answer_template.json`, keyed by `id` or `question`) and grade them
with zero agent calls:

```bash
.venv/bin/python evals-hackathon/llm_judge_grader.py \
  --questions "AI_Industry_Training_Hackathon/data set/evals/questions.json" \
  --answers-file evals-hackathon/sample_recorded_answers.json \
  --sample 10 --seed 42
```

```
component score   32.0/32  (100.0%)
perfect answers   10/10
wall clock        4.2s at concurrency 3
```

Same score as the live run (as it must be — it's the same recorded
answers), but ~13x faster since nothing is actually invoked: this mode is
for re-scoring answers you already have, or grading someone else's pasted
`/query` output without needing their service running.

`sample_recorded_answers.json` shows the exact row shape expected:
`{"id", "question", "answer", "steps", "tool_trace"}` per row (`seconds` is
optional, defaults to `0` — no slow-response penalty applies to an answer
you didn't just time yourself).

## Judge model independence (important)

The organizers' real hidden-question judge is **Qwen3.6-35B-A3B-FP8**,
reached through the private `agent-brain` service — not the team's
submitted agent, and not a stand-in for the required fine-tuned Nemotron
synthesis step (`Challenge_Brief.md`, `03_scoring_and_examples.md`). Since
this whole script is us rehearsing that grading on ourselves, `_build_judge`
mirrors that choice, with an explicit fallback chain:

1. `JUDGE_MODEL_NAME` / `JUDGE_MODEL_BASE_URL` / `JUDGE_MODEL_API_KEY` — set
   these for a dedicated judge endpoint.
2. Falls back to `AGENT_BRAIN_*` — reuses the *connection details* for the
   same Qwen deployment `finance_agent.py`'s planner talks to, since that's
   where the real judge model lives, but builds its own `ChatOpenAI`
   instance and makes independent API calls.
3. Falls back further to this repo's generic `MODEL_NAME` gateway so the
   grader still runs before any Qwen endpoint is reachable (what every
   example in this doc actually used).

At every step, the judge is a **separate object, separate prompt, separate
conversation** from `finance_agent.planner_model` — it never shares state
with the planner and never calls or substitutes for `finance_agent.synthesize`
(Nemotron). Swapping `DOMAIN_PREDICT_MODE=mock` → `llm` in `finance_agent.py`
does not change anything about how this grader judges answers, and nothing
in this file can affect what the submitted agent actually does.

## A concurrency bug this caught

Building `api.py`'s endpoint-mode demo surfaced a real reliability issue:
the first few `/query` requests that happen to need AFR data raced to
cold-load the ~780MB corpus simultaneously, and under 3-way concurrency
(the exact load the brief requires the submission to handle) this produced
300s+ stalls even though each AFR tool call is fast once the corpus is
warm. The fix — pre-warming every dataset during FastAPI's `lifespan`
startup hook, gating `/health` on it, exactly the pattern
`cognitivo_prep/src/main.py` already uses for its own warehouse load — is
worth carrying into any real submission service that lazy-loads its data.

## CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--questions PATH` | this repo's `dataset.py` examples | A `public_questions.jsonl` / `mock_questions.json`-shaped file. |
| `--endpoint URL` | — | Grade a live `/query` service instead of the in-process graph. |
| `--answers-file PATH` | — | Grade pre-recorded answers, no agent/endpoint call. |
| `--judge-model NAME` | `JUDGE_MODEL_NAME`/`.env` chain | Override the judge model. |
| `--llm-only` | off | Disable the deterministic fast-NO path; judge every component with the LLM. |
| `--concurrency N` | `3` | Simultaneous questions (matches the harness's documented default). |
| `--sample N` | — | Grade a random sample of N questions instead of all of them. |
| `--seed S` | random | Seed for `--sample`, for a reproducible sample. |
| `--limit N` | — | Only grade the first N questions (applied after `--sample`). |
| `--threshold PCT` | `0` | Exit non-zero if the overall score is below this percentage. |

## Output format

Console report, one line per question:

```
ok [easy  ]   5.1s 3/3  What is the lowest cash-rate target in the RBA dataset, when did
   [hard  ]   5.5s 4/6  Which loosening cycle ...
       missed (deterministic): '4.75' -- fact not found in the answer in any equivalent numeric/date form
       answer: ...
```

`ok` = every component matched. Each miss shows which grading mode caught it
(`deterministic` or `llm`) and the judge's one-sentence reason (or the
deterministic fast-path's fixed explanation). Summary:

```
component score   301.0/304  (99.0%)
perfect answers   95/101
over 60s          5 (each loses 20% of its earned points)
wall clock        432.1s at concurrency 3
```
