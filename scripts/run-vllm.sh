#!/usr/bin/env bash
#
# Launch a vLLM OpenAI-compatible server in Docker.
#
# Reconstructed from `docker inspect vllm-ft` on 2026-07-31, which was the only
# surviving copy of the original hand-typed `docker run`.
#
# Usage:
#   ./run-vllm.sh                          # serve the defaults below
#   MODEL=My-New-Model ./run-vllm.sh       # serve a different dir under models/
#   NAME=vllm-test PORT=8002 ./run-vllm.sh # run a second server alongside
#
# Every setting is an env-var override, so swapping models is a one-liner
# rather than an edit. See "Tuning notes" at the bottom.

set -euo pipefail

# --- Configuration -----------------------------------------------------------

# Host directory bind-mounted to /models in the container. Any model you want
# to serve must live underneath it.
MODELS_DIR="${MODELS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models}"

# Model to serve. Either a subdirectory of MODELS_DIR, or a HuggingFace repo ID
# (in which case set HF_TOKEN below for gated repos).
MODEL="${MODEL:-Llama-3.1-Nemotron-Nano-8B-v1}"

# Name clients pass in the OpenAI API "model" field.
SERVED_NAME="${SERVED_NAME:-nemotron-8b-finance}"

NAME="${NAME:-vllm-ft}"
PORT="${PORT:-8001}"           # host port; container always listens on 8000

# `latest` is what the running container was built from and is the only tag
# present locally (image f023269abe06, labelled vllm/vllm-openai:v0.21.0).
# Pinning to :v0.21.0 is more reproducible but will trigger a ~25GB pull.
IMAGE="${IMAGE:-vllm/vllm-openai:latest}"

# Fraction of GPU memory vLLM may claim. 0.45 was chosen to leave room for
# another GPU tenant; raise it toward 0.90 if vLLM has the GPU to itself.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"

# Max context length. Costs KV-cache memory, so it trades against GPU_MEM_UTIL.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Set to a token for gated HuggingFace repos. Ignored when empty.
HF_TOKEN="${HF_TOKEN:-}"

# --- Resolve the model path --------------------------------------------------

if [[ -d "${MODELS_DIR}/${MODEL}" ]]; then
    model_arg="/models/${MODEL}"

    # A symlink is followed by the shell but NOT by the bind mount. If it
    # resolves outside MODELS_DIR the container sees a dangling link, vLLM
    # falls back to treating the path as a HuggingFace repo ID, and the error
    # ("repo id must be in the form ...") points nowhere near the real cause.
    real="$(readlink -f "${MODELS_DIR}/${MODEL}")"
    real_root="$(readlink -f "${MODELS_DIR}")"
    if [[ "${real}" != "${real_root}/"* ]]; then
        echo "Error: ${MODELS_DIR}/${MODEL}" >&2
        echo "       resolves to ${real}" >&2
        echo "       which is outside the bind-mounted directory, so the" >&2
        echo "       container cannot see it." >&2
        echo "Fix: copy the model under ${MODELS_DIR}, or add a second" >&2
        echo "     -v mount for its real location." >&2
        exit 1
    fi

    # A PEFT adapter is not a servable model — it has no base weights,
    # config.json or tokenizer. It needs --enable-lora/--lora-modules on top
    # of a base model, or to be merged into the base first.
    if [[ -f "${MODELS_DIR}/${MODEL}/adapter_config.json" ]]; then
        echo "Error: ${MODEL} is a PEFT adapter, not a full model." >&2
        echo "       Merge it into its base model and serve the merged" >&2
        echo "       directory, or serve the base with --enable-lora." >&2
        if grep -q '"use_dora": true' "${MODELS_DIR}/${MODEL}/adapter_config.json" 2>/dev/null; then
            echo "Note: this adapter sets use_dora=true, which vLLM rejects" >&2
            echo "      outright — merging is the only option." >&2
        fi
        exit 1
    fi

    if [[ ! -f "${MODELS_DIR}/${MODEL}/config.json" ]]; then
        echo "Error: ${MODELS_DIR}/${MODEL}/config.json not found;" >&2
        echo "       this does not look like a HuggingFace model directory." >&2
        exit 1
    fi

elif [[ "${MODEL}" =~ ^[^/]+/[^/]+$ ]]; then
    # Exactly one slash and no local directory — a HuggingFace repo ID.
    model_arg="${MODEL}"
    echo "Note: '${MODEL}' is not a directory under ${MODELS_DIR};"
    echo "      treating it as a HuggingFace repo ID."
else
    echo "Error: no such model directory: ${MODELS_DIR}/${MODEL}" >&2
    echo "Available:" >&2
    ls -1 "${MODELS_DIR}" >&2
    exit 1
fi

# --- Replace any existing container by this name -----------------------------

if docker container inspect "${NAME}" >/dev/null 2>&1; then
    echo "Removing existing container '${NAME}'..."
    # Restart policy is unless-stopped, so stop before rm.
    docker stop "${NAME}" >/dev/null
    docker rm "${NAME}" >/dev/null
fi

# --- Launch ------------------------------------------------------------------

docker_args=(
    -d
    --name "${NAME}"
    --gpus all
    --restart unless-stopped
    -p "${PORT}:8000"
    -v "${MODELS_DIR}:/models"
    # HF cache lives beside the models so re-pulls survive container removal.
    -v "${MODELS_DIR}/.hf-cache:/root/.cache/huggingface"
)

if [[ -n "${HF_TOKEN}" ]]; then
    docker_args+=(-e "HF_TOKEN=${HF_TOKEN}")
fi

echo "Starting '${NAME}': ${model_arg} as '${SERVED_NAME}' on port ${PORT}"

docker run "${docker_args[@]}" "${IMAGE}" \
    --model "${model_arg}" \
    --served-model-name "${SERVED_NAME}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}"

cat <<EOF

Started. Model loading takes a few minutes — watch for "Application startup complete":

    docker logs -f ${NAME}

Then verify:

    curl -s localhost:${PORT}/v1/models | python3 -m json.tool

Tuning notes when switching to a different model:
  - GPU_MEM_UTIL=0.45 is sized for an 8B sharing the GPU. A larger model will
    OOM at this value; raise it if vLLM is the only tenant.
  - SERVED_NAME is what API clients send. Keep it stable or update the callers.
  - A newer architecture may need a newer vLLM than the pinned image.
EOF
