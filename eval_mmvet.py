#!/usr/bin/env python3
"""Generate answers for the MM-Vet benchmark with CausalLens (Qwen3-VL).

Launch through ``run_mmvet.sh`` (multi-GPU via accelerate). The output is a JSON
object ``{"v1_0": "<answer>", ...}``, the format expected by the official MM-Vet
GPT-4 evaluator.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import gather_object
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from causallens.model import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, load_model_and_processor


def load_mmvet(mmvet_dir: str) -> list[dict[str, str]]:
    """Read ``mm-vet.json`` (``{"v1_0": {"imagename": ..., "question": ...}, ...}``)."""
    mmvet_dir = Path(mmvet_dir)
    with open(mmvet_dir / "mm-vet.json", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for qid, item in data.items():
        image_path = mmvet_dir / "images" / item["imagename"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        samples.append({"qid": qid, "question": item["question"], "image": str(image_path)})
    return samples


@torch.no_grad()
def answer(model, processor, question: str, image_path: str, device, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question.strip()},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True, help="Qwen3-VL base model directory.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CausalLens checkpoint directory (LoRA + causal Q-Former). Omit to run the base model.",
    )
    parser.add_argument(
        "--no_causal_qformer",
        action="store_true",
        help="Load only the LoRA adapter from --checkpoint (ablation).",
    )
    parser.add_argument("--mmvet_dir", required=True, help="Directory with mm-vet.json and images/.")
    parser.add_argument("--output", required=True, help="Where to write the predictions JSON.")
    parser.add_argument("--max_samples", type=int, default=None, help="Only run the first N questions.")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--flash_attn", action="store_true", help="Use FlashAttention-2 (needs flash-attn).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # A few MM-Vet answers take minutes to generate; keep idle ranks from timing out at the gather.
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))])

    if args.checkpoint is None:
        mode = "base model"
    elif args.no_causal_qformer:
        mode = "LoRA only"
    else:
        mode = "LoRA + causal Q-Former"
    accelerator.print(f"Base model : {args.base_model}")
    accelerator.print(f"Checkpoint : {args.checkpoint} ({mode})")

    model, processor = load_model_and_processor(
        args.base_model,
        checkpoint=args.checkpoint,
        use_causal_qformer=not args.no_causal_qformer,
        flash_attn=args.flash_attn,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model.to(accelerator.device)

    samples = load_mmvet(args.mmvet_dir)[: args.max_samples]
    shard = samples[accelerator.process_index :: accelerator.num_processes]
    accelerator.print(f"MM-Vet     : {len(samples)} questions on {accelerator.num_processes} process(es)")

    results = []
    for sample in tqdm(shard, desc="MM-Vet", disable=not accelerator.is_main_process):
        try:
            prediction = answer(
                model,
                processor,
                sample["question"],
                sample["image"],
                accelerator.device,
                args.max_new_tokens,
            )
        except Exception as e:  # keep going; the evaluator scores a failed answer as 0
            print(f"[rank {accelerator.process_index}] ERROR on {sample['qid']}: {e}", flush=True)
            prediction = f"Error: {e}"
        results.append((sample["qid"], prediction))

    predictions = dict(gather_object(results))

    if accelerator.is_main_process:
        ordered = {s["qid"]: predictions[s["qid"]] for s in samples}
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(ordered, f, indent=2, ensure_ascii=False)

        num_errors = sum(p.startswith("Error:") for p in ordered.values())
        print(f"Saved {len(ordered)} predictions ({num_errors} errors) to {output}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
