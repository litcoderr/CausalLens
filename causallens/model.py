from __future__ import annotations

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from causallens.qformer import attach_causal_qformer, load_causal_qformer

# Image resolution bounds used for every reported MM-Vet number.
DEFAULT_MIN_PIXELS = 1097600
DEFAULT_MAX_PIXELS = 3211264


def load_model_and_processor(
    base_model: str,
    checkpoint: str | None = None,
    use_causal_qformer: bool = True,
    flash_attn: bool = False,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
):
    """Load Qwen3-VL ``base_model``, optionally with a CausalLens ``checkpoint`` on top.

    The checkpoint directory holds the LoRA adapter (``adapter_*``), the causal
    Q-Former (``causal_qformer_adapter.pt``) and the processor files. Without a
    checkpoint the plain base model is returned.
    """
    model_kwargs = {"dtype": torch.bfloat16}
    if flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, **model_kwargs)

    if checkpoint is not None:
        model.load_adapter(checkpoint)
        if use_causal_qformer:
            adapter = load_causal_qformer(checkpoint).to(torch.bfloat16)
            attach_causal_qformer(model, adapter)

    processor = AutoProcessor.from_pretrained(
        checkpoint or base_model,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return model.eval(), processor
