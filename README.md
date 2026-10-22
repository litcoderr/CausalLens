# CausalLens

CausalLens evaluates **Qwen3-VL-8B** with two additions on the
[MM-Vet](https://github.com/yuweihao/MM-Vet) benchmark:

- **Causal residual Q-Former** on the visual tokens. Learnable queries are first shifted by a
  backdoor-adjustment style intervention over a fixed confounder dictionary (16384 x 256 entries with
  a prior). They then read the image through cross-attention, and their result is projected back
  onto every visual token as a gated residual (`out = x + alpha * clip(delta, |x|)`).
- **LoRA adapter** (r=16) on the language model's attention and MLP projections.

```
image -> Qwen3-VL vision tower -> causal residual Q-Former -> Qwen3-VL LLM (+LoRA) -> answer
```

Weights: [huggingface.co/litcoderr/CausalLens](https://huggingface.co/litcoderr/CausalLens)

## Repository layout

```
CausalLens/
├── setup_env.sh          # create .venv with pinned dependencies (uv)
├── download.sh           # download weights from the Hugging Face Hub
├── run_mmvet.sh          # multi-GPU MM-Vet inference
├── eval_mmvet.py         # inference script (launched by run_mmvet.sh)
├── requirements.txt
└── causallens/
    ├── qformer.py        # causal residual Q-Former + hook into Qwen3-VL
    └── model.py          # load base model + LoRA + Q-Former + processor
```

## Quick start

```bash
git clone https://github.com/litcoderr/CausalLens.git && cd CausalLens
./setup_env.sh                                   # 1. environment
./download.sh                                    # 2. weights (~18 GB)
# 3. put MM-Vet in data/mm-vet (see below)
GPUS=0,1,2,3 ./run_mmvet.sh                      # 4. inference -> results/mmvet_causallens.json
```

## 1. Environment

```bash
./setup_env.sh
```

This script creates `.venv/` with Python 3.11 and the pinned packages from `requirements.txt`
(torch 2.7.1 for CUDA 11.8, transformers 4.57.5, peft 0.18.1, accelerate 1.12.0).
It needs [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`) and an
NVIDIA driver >= 520.
Without uv: `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## 2. Weights

```bash
./download.sh                  # adapters + base model
SKIP_BASE=1 ./download.sh      # adapters only, if you already have the base model
```

The weights come from [`litcoderr/CausalLens`](https://huggingface.co/litcoderr/CausalLens):

| Local path | Content |
| --- | --- |
| `checkpoints/CausalLens/` | LoRA adapter, causal Q-Former (`causal_qformer_adapter.pt`), processor files |
| `checkpoints/CausalLens/base/` | Qwen3-VL-8B base checkpoint the adapters were trained on (17.5 GB) |

If the base model is stored somewhere else, pass `BASE_MODEL=/path/to/base` to `run_mmvet.sh`.

## 3. MM-Vet data

`run_mmvet.sh` expects `data/mm-vet/mm-vet.json` and `data/mm-vet/images/`. To download MM-Vet from
the [official repository](https://github.com/yuweihao/MM-Vet):

```bash
mkdir -p data
curl -L -o data/mm-vet.zip https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip
unzip -q data/mm-vet.zip -d data && rm data/mm-vet.zip        # -> data/mm-vet/
```

If MM-Vet is stored somewhere else, pass `MMVET_DIR=/path/to/mm-vet` to `run_mmvet.sh`.

## 4. Run MM-Vet inference

```bash
GPUS=0,1,2,3 ./run_mmvet.sh                    # CausalLens (LoRA + causal Q-Former)
GPUS=0,1,2,3 MODE=lora ./run_mmvet.sh          # ablation: LoRA only
GPUS=0,1,2,3 MODE=base ./run_mmvet.sh          # baseline: base model only
GPUS=0 ./run_mmvet.sh --max_samples 8          # smoke test on the first 8 questions
```

The script starts one process per GPU. The 218 questions are split across these processes, and each
process holds the full model in bf16 (about 18 GB of GPU memory). On 8x RTX 3090 (24 GB), a full run
takes about 4 minutes including model loading, with a peak of about 19 GB per GPU.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPUS` | `$CUDA_VISIBLE_DEVICES`, else `0` | comma-separated GPU ids |
| `MODE` | `causallens` | `causallens`, `lora` or `base` |
| `BASE_MODEL` | `checkpoints/CausalLens/base` | Qwen3-VL base model directory |
| `CKPT` | `checkpoints/CausalLens` | CausalLens checkpoint directory |
| `MMVET_DIR` | `data/mm-vet` | MM-Vet directory |
| `OUTPUT` | `results/mmvet_$MODE.json` | predictions file |
| `PYTHON` | `.venv/bin/python` | interpreter, e.g. from another environment |

Arguments after `run_mmvet.sh` go to `eval_mmvet.py`. These include `--max_samples N`,
`--max_new_tokens` (2048), `--min_pixels` / `--max_pixels` (1097600 / 3211264) and `--flash_attn`
(requires `flash-attn`). Decoding is greedy.

Outputs:

- `results/mmvet_<MODE>.json` holds the answers as `{"v1_0": "<answer>", ...}`. This is the
  format that the MM-Vet evaluator expects.
- `logs/mmvet_<MODE>_<timestamp>.log` holds the full console log.

A question that fails is recorded as `"Error: ..."`. The run continues, and the number of
errors is printed at the end.

## 5. Scoring

MM-Vet grades open-ended answers with GPT-4. You can upload `results/mmvet_causallens.json` to the
[MM-Vet online evaluator](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator), or run the official
grader with your OpenAI key:

```bash
git clone https://github.com/yuweihao/MM-Vet.git third_party/MM-Vet
uv pip install --python .venv/bin/python openai pandas
.venv/bin/python third_party/MM-Vet/mm-vet_evaluator.py \
    --mmvet_path data/mm-vet \
    --result_file results/mmvet_causallens.json \
    --result_path results/grading \
    --openai_api_key "$OPENAI_API_KEY"
```

## Python API

```python
from causallens import load_model_and_processor

model, processor = load_model_and_processor(
    base_model="checkpoints/CausalLens/base",
    checkpoint="checkpoints/CausalLens",   # None -> base model only
    use_causal_qformer=True,               # False -> LoRA only
)
```

`model` is a regular `Qwen3VLForConditionalGeneration`, so `processor` and `model.generate`
work as usual. See `eval_mmvet.py` for the full inference loop.
