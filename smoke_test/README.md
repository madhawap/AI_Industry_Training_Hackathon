# LiteLLM smoke tests

Quick checks that the org LiteLLM proxy and model aliases are reachable.

| Alias | Role | Shortcut |
|---|---|---|
| `agent-brain` | Qwen planner | `qwen` / `brain` |
| `domain-ft` | fine-tuned Nemotron | `nemotron` / `domain` |

Reads `LITELLM_*` / model aliases from the repo-root `.env`.

```bash
# from repo root, with venv activated
python smoke_test/check_litellm.py              # both
python smoke_test/check_litellm.py qwen         # Qwen only
python smoke_test/check_litellm.py nemotron     # Nemotron only
python smoke_test/check_litellm.py list         # list /v1/models
```
