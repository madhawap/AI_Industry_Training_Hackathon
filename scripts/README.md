# Bringup Scripts

These scripts need to be copied to `~/local-llm-setup` to be brought up.

They expect a `models/` directory alongside them (`~/local-llm-setup/models`),
holding the base model, adapter checkpoints, and merged outputs. Everything
served or merged must live under that directory — it is the only path
bind-mounted into the containers, and bind mounts do not follow symlinks out
of the tree.

## Quickstart

```bash
cp scripts/{run-vllm.sh,merge-adapter.sh,merge_adapter.py} ~/local-llm-setup/
cd ~/local-llm-setup

# 1. Fold the LoRA/DoRA checkpoint into the base model (CPU-only, ~10 min).
ADAPTER=models/checkpoint-100 ./merge-adapter.sh

# 2. Serve the merged model on port 8001.
MODEL=nemotron-8b-finance-merged ./run-vllm.sh

# 3. Wait for "Application startup complete", then verify.
docker logs -f vllm-ft
curl -s localhost:8001/v1/models | python3 -m json.tool
```

Step 1 needs ~25GB of free RAM. If a vLLM server is already up it will
usually still fit, but `docker stop vllm-ft` first if the preflight check
complains.

## Scripts

| Script | Purpose |
| --- | --- |
| `run-vllm.sh` | Launches a vLLM OpenAI-compatible server in Docker (`vllm-ft`, host port 8001), bind-mounting `models/` at `/models`. Validates the model path first and gives a real error for the common failure modes: symlinks that escape the mount, PEFT adapters passed where a full model is expected, missing `config.json`. Replaces any existing container of the same name. |
| `merge-adapter.sh` | Wrapper that merges a PEFT adapter into its base model and writes a standalone model directory under `models/` that `run-vllm.sh` can serve. Runs `merge_adapter.py` inside `nvcr.io/nvidia/nemo:25.09` because the host has no torch and the vLLM image ships no peft. Resolves symlinks, preflights RAM, and runs CPU-only so a live vLLM server keeps the GPU. |
| `merge_adapter.py` | The merge itself: load base on CPU, apply the adapter, `merge_and_unload`, save with the base model's tokenizer copied alongside. Verifies a probe weight actually changed, so a silently no-op adapter fails loudly instead of producing a valid-looking model. Not runnable on the host directly — invoke it through `merge-adapter.sh`. |

Merging is required rather than vLLM's `--enable-lora` because the checkpoint
sets `use_dora=true`, which vLLM's runtime LoRA path rejects outright.

## Configuration

Both shell scripts are configured entirely by environment variable, so
swapping models is a one-liner rather than an edit.

`run-vllm.sh`:

| Variable | Default | Notes |
| --- | --- | --- |
| `MODEL` | `Llama-3.1-Nemotron-Nano-8B-v1` | Subdirectory of `models/`, or a HuggingFace repo ID |
| `SERVED_NAME` | `nemotron-8b-finance` | What clients put in the OpenAI `model` field |
| `NAME` / `PORT` | `vllm-ft` / `8001` | Change both to run a second server alongside |
| `GPU_MEM_UTIL` | `0.45` | Sized for an 8B sharing the GPU; raise toward `0.90` if vLLM has it to itself |
| `MAX_MODEL_LEN` | `4096` | Costs KV-cache memory, so it trades against `GPU_MEM_UTIL` |
| `IMAGE` | `vllm/vllm-openai:latest` | Locally present tag; pinning to `:v0.21.0` triggers a ~25GB pull |
| `HF_TOKEN` | *(empty)* | Needed only for gated HuggingFace repos |

`merge-adapter.sh`:

| Variable | Default | Notes |
| --- | --- | --- |
| `BASE` | `models/Llama-3.1-Nemotron-Nano-8B-v1` | Must be the model the adapter was trained against |
| `ADAPTER` | `models/checkpoint-100` | Any Trainer checkpoint containing `adapter_config.json` |
| `OUT_NAME` | `nemotron-8b-finance-merged` | Output directory name under `models/` |
| `DTYPE` | `bfloat16` | Must match the base model's `config.json` |
| `FORCE` | *(unset)* | Set to `1` to overwrite a non-empty output directory |

Both also honour `MODELS_DIR`, which defaults to `models/` next to the script.
