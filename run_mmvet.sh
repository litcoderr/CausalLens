#!/usr/bin/env bash
# Run MM-Vet inference with CausalLens (one process per GPU).
#
#   GPUS=0,1,2,3 ./run_mmvet.sh                  # CausalLens: LoRA + causal Q-Former
#   GPUS=0,1,2,3 MODE=lora ./run_mmvet.sh        # LoRA only (ablation)
#   GPUS=0,1,2,3 MODE=base ./run_mmvet.sh        # base model only (baseline)
#   GPUS=0 ./run_mmvet.sh --max_samples 8        # quick smoke test
#
# Environment variables (defaults in brackets):
#   GPUS        comma-separated GPU ids                  [$CUDA_VISIBLE_DEVICES, else 0]
#   MODE        causallens | lora | base                 [causallens]
#   BASE_MODEL  Qwen3-VL base model directory            [checkpoints/CausalLens/base]
#   CKPT        CausalLens checkpoint directory          [checkpoints/CausalLens]
#   MMVET_DIR   directory with mm-vet.json and images/   [data/mm-vet]
#   OUTPUT      predictions JSON                         [results/mmvet_$MODE.json]
#   PYTHON      Python interpreter                       [.venv/bin/python]
# Any extra arguments are passed to eval_mmvet.py (see: .venv/bin/python eval_mmvet.py --help).
set -euo pipefail
cd "$(dirname "$0")"

GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
MODE=${MODE:-causallens}
BASE_MODEL=${BASE_MODEL:-checkpoints/CausalLens/base}
CKPT=${CKPT:-checkpoints/CausalLens}
MMVET_DIR=${MMVET_DIR:-data/mm-vet}
OUTPUT=${OUTPUT:-results/mmvet_${MODE}.json}
PYTHON=${PYTHON:-.venv/bin/python}

die() { echo "error: $*" >&2; exit 1; }

case $MODE in
  causallens) mode_args=(--checkpoint "$CKPT") ;;
  lora)       mode_args=(--checkpoint "$CKPT" --no_causal_qformer) ;;
  base)       mode_args=() ;;
  *)          die "unknown MODE=$MODE (expected causallens, lora or base)" ;;
esac

[[ -x $PYTHON ]] || die "$PYTHON not found. Run ./setup_env.sh first."
[[ -f $BASE_MODEL/config.json ]] || die "base model not found at $BASE_MODEL. Run ./download.sh or set BASE_MODEL."
[[ $MODE == base || -f $CKPT/causal_qformer_adapter.pt ]] || die "CausalLens checkpoint not found at $CKPT. Run ./download.sh or set CKPT."
[[ -f $MMVET_DIR/mm-vet.json ]] || die "MM-Vet not found at $MMVET_DIR (expects mm-vet.json and images/). See README.md."

num_gpus=$(tr ',' '\n' <<< "$GPUS" | grep -c .)
port=${PORT:-$("$PYTHON" -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')}
mkdir -p logs
log=logs/mmvet_${MODE}_$(date +%Y%m%d_%H%M%S).log

echo "MODE=$MODE GPUS=$GPUS -> $OUTPUT (log: $log)"
CUDA_VISIBLE_DEVICES=$GPUS "$PYTHON" -m accelerate.commands.launch \
  --num_processes "$num_gpus" --num_machines 1 --mixed_precision no --dynamo_backend no \
  --main_process_port "$port" \
  eval_mmvet.py \
  --base_model "$BASE_MODEL" \
  "${mode_args[@]}" \
  --mmvet_dir "$MMVET_DIR" \
  --output "$OUTPUT" \
  "$@" 2>&1 | tee "$log"
