# TFQL — Typed Financial Query Language

Qwen selects typed operations. This package validates and executes them. The
fine-tuned Nemotron writes the answer from the resulting evidence bundle.

```
question
  → Qwen planner            emits one execute_plan envelope
  → TFQL validator          whole plan checked before anything runs
  → TFQL executor           deterministic ops, in dependency order
  → evidence bundle         computed values + method + coverage + warnings
  → fine-tuned Nemotron     writes the final answer from the bundle alone
```

## Why it exists

Three forces, each of which would produce something like this on its own.

**The latency budget makes tool-call *count* the dominant variable.** 60 seconds
for full points, three questions running concurrently. Nearly all of that is
model forward passes — the data work here is microseconds. Batching independent
operations into one envelope is the only lever that attacks the real cost.

**The judge checks exact values and language models produce plausible ones.**
Two of the challenge brief's three zero-score examples are arithmetic or
composition failures. The partial-credit example lost half its marks because a
date was one day off.

**Fine-tuning needs a stable input contract.** The evidence bundle *is* the
training input format, so its shape is pinned by `SCHEMA_VERSION` and changes
deliberately.

## Layout

| module | role |
|---|---|
| `errors.py` | structured error codes; nothing ever substitutes a plausible value for a failure |
| `precision.py` | rates as integer basis points; percent conversion at the output boundary only |
| `dates.py` | the predecessor/successor lookup — `previous \| next \| nearest \| exact` |
| `coverage.py` | per-dataset intervals, clamping, and cross-dataset overlap |
| `evidence.py` | the bundle handed to the synthesiser |
| `invariants.py` | financial identity checks that turn a wrong number into an error |
| `registry.py` | the closed operation vocabulary, and the planner catalogue generated from it |
| `store.py` | read-only warehouse access and the startup precompute |
| `models.py` | the plan envelope and result contract |
| `executor.py` | plan validation, dependency ordering, partial-failure semantics |
| `operations/` | the four operation boxes |

## Design rules

1. **One op answers one question shape**, not one primitive. `rba.longest_hold`,
   not "min of a column".
2. **Return every component that shape is graded on.** `rba.rate_extreme`
   returns the rate *and* its first effective date *and* the record count,
   because the worked example lost 50% on the date.
3. **Every number ships in the unit the answer will state.** `return_pct` sits
   beside `return_decimal` so the synthesiser never multiplies. Any conversion
   left undone is one the model will attempt.
4. **Business rules live inside the op.** AFR field scope and ASX trading-day
   alignment are not arguments the planner can get wrong.
5. **Coverage and method are always reported**, so cross-dataset gaps are stated
   rather than silently spanned.
6. **A failed operation never fails the plan.** Partial credit is explicit in
   the rubric; three correct components out of four still score.

## Non-negotiable rules this enforces

From the Setup Instructions, verified by `tests/test_afr_matching.py`:

- AFR pattern counts search **HEADLINE + SUBHEAD + INTRO + TEXT** combined, via
  the single `AFR_ALL_TEXT` expression, case-insensitively, **once per record**.
- Whole-word searches are `\b`-anchored. On the mock corpus, `RBA` matches
  4 articles whole-word but 5 as a substring — the fifth is *Transu**rba**n*.
  `bank` gives 13 versus 21.
- **Counting never touches the FTS index.** That index stems (`bank` → `banking`)
  and omits `intro`. Retrieval uses it; counting does not.

## Operations

**RBA** `rate_extreme` `rate_at_date` `change_summary` `longest_hold`
`rate_cycle` `period_comparison`

**ASX** `return` `price_extreme` `biggest_move` `max_drawdown` `rank_returns`
`volume_rank` `event_window` `equal_weight_basket` `summary_stat`

**AFR** `pattern_count` `retrieve_articles` `date_count`

**Cross** `rate_event_market_return` `news_rate_context`

Everything else that spans datasets uses `depends_on` rather than a new
operation — `${peak.data.date}` references resolve at execution time.

## Verification status

Against the challenge brief's worked examples:

| example | expected | result |
|---|---|---|
| lowest rate | 0.1 / 2020-11-04 / 16 records | matches |
| longest hold | 1036 days / 2016-08-03 → 2019-06-05 / 1.5 → 1.25 | matches |
| tightening cycle | 13 hikes / +4.25pp / 0.1 → 4.35 | matches |
| highest rate | 4.75 / 2010-11-02 / 11 records | rate and count match; see below |

### End-to-end, through the real model

Run 2026-07-31 against local vLLM `qwen3.6-27b-fp8-modok`, full pipeline
(Qwen plans → TFQL executes → Qwen synthesises), scored by counting the
components the challenge brief lists for each worked example:

| case | latency | ops | components |
|---|---|---|---|
| easy — lowest rate | 6.3 s | 1 | 3/3 |
| medium — longest hold | 6.1 s | 1 | 5/5 |
| hard — tightening cycle | 7.5 s | 1 | 4/4 |
| cross — BHP peak + rate that day | 10.1 s | 2 | 3/3 |

**15/15 components**, every question inside the 60-second budget. Three
concurrent questions completed in 8.1 s wall clock.

The planner batched into a single `execute_plan` call in every case, and
resolved a `${peak.data.date}` dependency unaided on the cross-dataset one.

### Disable reasoning tokens on the planner

`enable_thinking` must be **false** for both model calls:

```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

Measured on the same two-operation plan:

| mode | planner latency | completion tokens | plan |
|---|---|---|---|
| thinking on (vLLM default) | 83–92 s | 1726 | correct |
| thinking off | 5.4 s | 110 | identical |

Fifteen times faster for the same output. The reasoning trace is wasted work
here — the operation catalogue already constrains the output space, so the
planner only selects operations and never needs to reason about arithmetic.
Left on, a single question breaches the 60-second cliff on planning alone.

## Known limitations

1. **The mock RBA data is a day off from the real series.** The brief's
   partial-credit example states the judge expects `2010-11-02` for the first
   4.75 record; the mock CSV contains `3 Nov 2010,+0.25,4.75`. The operation
   correctly reports what it is given. Re-verify against the real dataset.

2. **`AFR_ALL_TEXT` is evaluated per query, not materialised.** The shared
   warehouse is read-only, so the four-field concatenation is built in SQL each
   time. Free at 110 articles; on the real ~200k corpus it should be
   materialised at ingest, or replaced with a startup-built inverted index.

3. **Mock coverage is narrow.** 4 tickers instead of 18, and AFR spans only
   2015-01-05 → 2015-03-31. The three-dataset overlap is about twelve weeks, so
   cross-dataset behaviour is under-exercised.

4. **`intro` is empty in all 110 mock AFR records**, so the four-field rule
   cannot be validated end-to-end here — only that the scope is correct.

5. **Operations run sequentially.** Deliberate: data work is microseconds
   against model calls measured in seconds, so parallelising costs determinism
   and buys nothing measurable.

6. **No timeout enforcement per operation yet.** `OPERATION_TIMEOUT` is defined
   but unused; nothing currently runs long enough to need it.

## Running

```bash
conda env create -f environment.yml
export PYTHONNOUSERSITE=1        # user site-packages otherwise shadow the env
conda run -n cognitivo python -m pytest src/tests -q
```

The warehouse location defaults to the shared mock data and is overridable:

```bash
export TFQL_WAREHOUSE=/path/to/warehouse.duckdb
```
