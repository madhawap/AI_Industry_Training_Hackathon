# How to Run

This repository is a Cognitivo hackathon financial Q&A agent. It exposes a FastAPI service that:

1. Uses **Qwen (`agent-brain`)** via LiteLLM to plan and emit TFQL tool calls
2. Executes those calls against a **DuckDB warehouse** (RBA / ASX / AFR)
3. Uses **fine-tuned Nemotron (`domain-ft`)** to synthesize the final `answer`

How the pieces fit together (LangGraph workflow, tool registry, data access) is documented in [`ARCHITECTURE.md`](ARCHITECTURE.md).

Default listen address from `config.yaml`: `0.0.0.0:8011`.

---

## Prerequisites

- Python 3.11+ recommended
- Raw datasets under `data set/` (already in this repo):
  - `data set/AFR/`
  - `data set/ASX/`
  - `data set/RBA Rates/`
- Model stack reachable through LiteLLM (you must start the proxy yourself — it is **not** auto-started):
  - LiteLLM proxy on the brain/agent node (default `http://localhost:4000`)
  - `agent-brain` alias → Qwen vLLM (usually `:8000` on this node)
  - `domain-ft` alias → fine-tuned Nemotron vLLM (usually `:8001` on the fine-tuning node)

On the Atom cluster, source the organizer env first if present:

```bash
source ~/team.env
```

Cluster layout (see `Participant_Package/handout/02_execution_guide.md`):

```text
brain/agent node                     fine-tuning/model node
─────────────────────                ────────────────────────
• LiteLLM proxy   :4000              • Fine-tuned vLLM  :8001
  ├─ agent-brain → Qwen :8000
  └─ domain-ft   → model node :8001
• Agent server    :8011 (this repo)
```

---

## 1. Install dependencies

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install 'litellm[proxy]'   # required to run the local LiteLLM proxy (not in requirements.txt)
```

For unit tests, also install pytest (not listed in `requirements.txt`):

```bash
pip install pytest pytest-asyncio
```

---

## 2. Build the DuckDB warehouse

The agent does **not** read raw JSONL/CSV at request time. Build `warehouse.duckdb` once (or after data changes).

```bash
cd src/data
python3 normalize_dates.py "../../data set" ./normalized
python3 setup_duckdb.py ./normalized ./parquet ./warehouse.duckdb
python3 test_queries.py ./warehouse.duckdb
cd ../..
```

Expected outputs (gitignored):

- `src/data/normalized/`
- `src/data/parquet/`
- `src/data/warehouse.duckdb`

Point the agent at that warehouse:

```bash
export TFQL_WAREHOUSE="$(pwd)/src/data/warehouse.duckdb"
```

If `TFQL_WAREHOUSE` is unset, the code looks for the cluster default path:

`/home/datasets/cognitivo_hackathon/mock_data/warehouse.duckdb`

---

## 3. Configure environment

Copy the example env file and edit as needed:

```bash
cp .env.example .env
```

Key variables (also documented in `.env.example` and `config.yaml`):

| Variable | Purpose | Typical value |
|---|---|---|
| `LITELLM_BASE_URL` | OpenAI-compatible LiteLLM base | `http://localhost:4000` |
| `LITELLM_KEY` | LiteLLM credential | organizer-provided (or empty / `dummy` for local) |
| `BRAIN_MODEL` | Planning / tool-calling model alias | `agent-brain` |
| `DOMAIN_FT_MODEL` | Final synthesis model alias | `domain-ft` |
| `DOMAIN_PREDICT_MODE` | API shape for domain-ft: `chat` / `llm` / `completion` | `chat` |
| `MAX_AGENT_STEPS` | Max brain↔tool loop iterations | `5` |
| `TFQL_WAREHOUSE` | Path to `warehouse.duckdb` | see step 2 |
| `APP_CONFIG` | Optional override for `config.yaml` | absolute path |

`DOMAIN_PREDICT_MODE=llm` is accepted as an alias for `chat`. Always call the real `domain-ft` endpoint for scoring — do not leave synthesis on a mock path.

Optional: keep settings in `config.yaml` (env placeholders like `${LITELLM_BASE_URL:-http://localhost:4000}` are expanded at load time). Env vars win for the LiteLLM knobs above.

---

## 4. Start LiteLLM (required)

The agent talks **only** to LiteLLM (`LITELLM_BASE_URL`). If nothing is listening on `:4000`, `/query` fails and `smoke_test/check_litellm.py` reports `Connection error.`

### 4a. Ensure upstream model backends are up

LiteLLM only routes; the underlying vLLM servers must already be running.

**Qwen brain (this node, port 8000)** — on the Atom cluster the organizer container is typically named `vllm-brain`:

```bash
docker ps -a --filter name=vllm-brain
# If Status is Exited:
docker start vllm-brain

# Wait until the OpenAI-compatible API answers (model load can take a few minutes):
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/v1/models
# expect 200
```

**Fine-tuned Nemotron (fine-tuning node, port 8001)** — start/serve your adapter on the other node (see handout `02_execution_guide.md`). LiteLLM’s `domain-ft` route must point at that host.

### 4b. Start the LiteLLM proxy

Organizer config on this cluster lives at `~/litellm/config.yaml` (routes `agent-brain` → Qwen, `domain-ft` → fine-tuned node). From the repo venv:

```bash
source .venv/bin/activate
litellm --config ~/litellm/config.yaml --port 4000 --host 0.0.0.0
```

Leave that process running (dedicated terminal or background job). Confirm the routes match your cluster IPs:

```yaml
model_list:
  - model_name: agent-brain
    litellm_params:
      model: openai/Qwen/...
      api_base: http://<brain-node>:8000/v1
  - model_name: domain-ft
    litellm_params:
      model: openai/nemotron-...
      api_base: http://<fine-tuning-node>:8001/v1
```

If you do not have `~/litellm/config.yaml`, create one from the example above (see also `Participant_Package/handout/02_execution_guide.md`) and pass that path to `--config`.

### 4c. Verify LiteLLM

```bash
# from repo root, venv active, .env loaded by the script
python smoke_test/check_litellm.py list      # GET /v1/models
python smoke_test/check_litellm.py           # ping agent-brain + domain-ft
```

More detail: `smoke_test/README.md`.

---

## 5. Start the agent

From the repository root (with the venv active, `TFQL_WAREHOUSE` set, and LiteLLM up from step 4):

```bash
python -m src.main
```

Equivalent:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8011
```

Startup warms the TFQL store (loads RBA/ASX into memory, opens AFR via DuckDB). `/health` returns **503** until that finishes, then **200**.

> **Port note:** this agent defaults to `:8011` so it does not collide with Nemotron vLLM’s `:8001` (on the fine-tuning node, or if you serve it locally). Do not point `LITELLM_BASE_URL` at the agent; keep it on LiteLLM `:4000`.

### Health check

```bash
curl -s http://localhost:8011/health
# {"status":"ok"}
```

### Ask a question

```bash
curl -s http://localhost:8011/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What was the RBA cash rate target on 2020-03-19?"}'
```

Expected shape:

```json
{
  "answer": "...",
  "steps": 2,
  "tool_trace": [
    {"tool": "execute_plan", "args": {}, "result": "..."}
  ]
}
```

The evaluation harness grades only `answer`. The service must tolerate **at least three concurrent** `POST /query` requests.

---

## 6. Run tests (no LLM required for most)

TFQL unit tests need a built warehouse:

```bash
export TFQL_WAREHOUSE="$(pwd)/src/data/warehouse.duckdb"
python -m pytest src/tests -q
```

---

## 7. Local offline evaluation (optional)

With the agent running (`python -m src.main`) and LiteLLM + both model aliases live, POST every public question to `/query` and write `src/eval/results.json`:

```bash
python -m src.eval.run_questions
# or explicitly:
python -m src.eval.run_questions Participant_Package/public_questions.jsonl \
  --endpoint http://localhost:8011 --concurrency 3
```

Each results record contains `question`, full `/query` response (`answer` / `steps` / `tool_trace`), plus `required_facts`, `scoring_notes`, and `reference_answer` from the JSONL.

Useful flags: `--limit N`, `--concurrency 3`, `--output path`, `--timeout 300`.

---

## Architecture (runtime)

```text
POST /query
  → Qwen (agent-brain) plans / emits tool calls
  → TFQL runtime validates + executes against DuckDB
  → loop until done or MAX_AGENT_STEPS
  → Nemotron (domain-ft) synthesizes final answer
  → JSON { answer, steps, tool_trace }
```

More detail:

- **App architecture, LangGraph workflow, tool registry, how to add tools: [`ARCHITECTURE.md`](ARCHITECTURE.md)** (with diagrams)
- Agent contract / scoring: `README.md`
- Dataset + ingest pipeline: `src/data/README.md`
- TFQL operations: `src/tfql/README.md`
- Cluster serving layout: `Participant_Package/handout/02_execution_guide.md`
- LiteLLM smoke tests: `smoke_test/README.md`

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Startup fails: `warehouse not found` | Missing DB or wrong path | Build warehouse (step 2) and set `TFQL_WAREHOUSE` |
| `/health` stays 503 | Store still warming or crashed mid-load | Check logs; ensure DuckDB file is valid |
| `smoke_test/... list` → `Connection error` | LiteLLM not running on `:4000` | Start proxy (step 4); confirm `LITELLM_BASE_URL` |
| `502 Upstream / workflow error` | LiteLLM or model alias down | Verify step 4; `check_litellm.py`; `agent-brain` / `domain-ft` |
| LiteLLM up but model calls fail | Upstream vLLM stopped / wrong `api_base` | `docker start vllm-brain`; fix IPs in `~/litellm/config.yaml` |
| Wrong AFR counts | Incomplete field search or missing `\b` | AFR ops search HEADLINE+SUBHEAD+INTRO+TEXT; use word boundaries |
| Port mismatch with `submission.json` | Template uses `:5000`, this app defaults to `:8011` | Align `config.yaml` / uvicorn port with `submission.json` → `agent.endpoint` |
| `ContextWindowExceededError` (4096) | `max_tokens_*` + prompt exceed vLLM `max_model_len` | Keep `max_tokens_brain` / `max_tokens_synthesis` well under 4096 (defaults: 256 / 512); restart the agent after changing `config.yaml` |

---

## Quick checklist

1. `pip install -r requirements.txt` and `pip install 'litellm[proxy]'`
2. Build `src/data/warehouse.duckdb`
3. `export TFQL_WAREHOUSE=...`
4. Configure `.env` / LiteLLM aliases
5. Ensure Qwen (`vllm-brain` / `:8000`) and Nemotron (`domain-ft` backend) are up
6. `litellm --config ~/litellm/config.yaml --port 4000 --host 0.0.0.0`
7. `python smoke_test/check_litellm.py` → PASS
8. `python -m src.main`
9. Confirm `GET /health` → 200 and `POST /query` returns `answer`
10. Update `submission.json` with the public endpoint and commit SHA before scoring
