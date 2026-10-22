#!/usr/bin/env bash
# Download the CausalLens weights from the Hugging Face Hub (litcoderr/CausalLens):
#   checkpoints/CausalLens/        LoRA adapter + causal Q-Former + processor  (~0.2 GB)
#   checkpoints/CausalLens/base/   Qwen3-VL-8B base model                      (~17.5 GB)
#
#   ./download.sh                  # everything
#   SKIP_BASE=1 ./download.sh      # adapters only; point BASE_MODEL at your own copy of the base
set -euo pipefail
cd "$(dirname "$0")"

HF_REPO=${HF_REPO:-litcoderr/CausalLens}
DEST=${DEST:-checkpoints/CausalLens}
HF=.venv/bin/hf

if [[ ! -x $HF ]]; then
  echo "$HF not found. Run ./setup_env.sh first." >&2
  exit 1
fi

exclude=()
if [[ ${SKIP_BASE:-0} == 1 ]]; then
  exclude=(--exclude "base/*")
fi

"$HF" download "$HF_REPO" --local-dir "$DEST" "${exclude[@]}"
echo "Weights ready in $DEST"
