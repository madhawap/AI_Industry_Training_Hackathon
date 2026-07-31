#!/usr/bin/env bash
#
# Merge a PEFT/LoRA adapter into its base model, producing a directory that
# ./run-vllm.sh can serve directly.
#
# Usage:
#   ./merge-adapter.sh                                  # defaults below
#   ADAPTER=/path/to/checkpoint-120 OUT_NAME=v2 ./merge-adapter.sh
#
# Then:
#   MODEL=nemotron-8b-finance-merged ./run-vllm.sh
#
# Runs inside nvcr.io/nvidia/nemo:25.09 because the host has no torch, and
# because the vllm/vllm-openai image ships transformers 5.x without peft.
# The merge is CPU-only, so a vLLM server can keep serving on the GPU while
# this runs.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS_DIR="${MODELS_DIR:-${HERE}/models}"

# Base model. Must be the model the adapter was trained against — its
# adapter_config.json records "nvidia/Llama-3.1-Nemotron-Nano-8B-v1", which
# is the local copy below.
BASE="${BASE:-${MODELS_DIR}/Llama-3.1-Nemotron-Nano-8B-v1}"

# Adapter checkpoint. Resolved to a real path because models/checkpoint-60 is
# a symlink out of the tree, and bind mounts do not follow symlinks.
ADAPTER="${ADAPTER:-${MODELS_DIR}/checkpoint-100}"

# Merged output lands under MODELS_DIR so run-vllm.sh's bind mount can see it.
OUT_NAME="${OUT_NAME:-nemotron-8b-finance-merged}"
OUT="${MODELS_DIR}/${OUT_NAME}"

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo:25.09}"
DTYPE="${DTYPE:-bfloat16}"   # matches the base model's config.json
FORCE="${FORCE:-}"           # set to 1 to overwrite a non-empty OUT

# --- Resolve and validate ----------------------------------------------------

base_real="$(readlink -f "${BASE}")"
adapter_real="$(readlink -f "${ADAPTER}")"

if [[ ! -f "${base_real}/config.json" ]]; then
    echo "Error: no config.json under ${BASE} (-> ${base_real})" >&2
    exit 1
fi
if [[ ! -f "${adapter_real}/adapter_config.json" ]]; then
    echo "Error: no adapter_config.json under ${ADAPTER} (-> ${adapter_real})" >&2
    exit 1
fi

# 8B in bf16 needs ~16GB resident for the base, plus the adapter and the
# state dict held during save. Fail early rather than after a 10-minute load.
avail_gb="$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)"
if (( avail_gb < 25 )); then
    echo "Error: ${avail_gb}GB available RAM; the merge needs ~25GB." >&2
    echo "       Stop the vLLM container (docker stop vllm-ft) and retry." >&2
    exit 1
fi

mkdir -p "${OUT}"

echo "Base:    ${base_real}"
echo "Adapter: ${adapter_real}"
echo "Output:  ${OUT}"
echo

# --- Merge -------------------------------------------------------------------

py_args=(--base /base --adapter /adapter --out /out --dtype "${DTYPE}")
[[ -n "${FORCE}" ]] && py_args+=(--force)

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e HF_HOME=/tmp/hf \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -v "${base_real}:/base:ro" \
    -v "${adapter_real}:/adapter:ro" \
    -v "${OUT}:/out" \
    -v "${HERE}/merge_adapter.py:/merge_adapter.py:ro" \
    --entrypoint python3 \
    "${IMAGE}" /merge_adapter.py "${py_args[@]}"

cat <<EOF

Merged. Serve it with:

    MODEL=${OUT_NAME} ./run-vllm.sh

SERVED_NAME stays 'nemotron-8b-finance' by default, so existing API clients
need no change.
EOF
