#!/usr/bin/env bash
# Create ./.venv (Python 3.11) with the pinned dependencies from requirements.txt.
#
#   ./setup_env.sh
#
# Needs uv (https://docs.astral.sh/uv/). uv downloads Python 3.11 itself if missing.
set -euo pipefail
cd "$(dirname "$0")"

UV=$(command -v uv || true)
for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
  [[ -z $UV && -x $candidate ]] && UV=$candidate
done
if [[ -z $UV ]]; then
  echo "uv not found. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

"$UV" venv --python 3.11 --allow-existing .venv
"$UV" pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements.txt

.venv/bin/python - <<'EOF'
import torch, transformers, peft, accelerate
print(f"torch {torch.__version__} (CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()})")
print(f"transformers {transformers.__version__}, peft {peft.__version__}, accelerate {accelerate.__version__}")
EOF
echo "Environment ready: $(pwd)/.venv"
