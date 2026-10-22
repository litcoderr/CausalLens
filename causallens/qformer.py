"""Causal residual Q-Former adapter for Qwen3-VL visual tokens.

The adapter sits between the Qwen3-VL vision tower and the language model and
adds a small, gated residual to every visual token ``x``:

    q   = LayerNorm(learnable queries)
    z   = (softmax(q D^T / sqrt(d)) * prior) D      # intervention over confounder dictionary D
    m   = CrossAttn(q + s * z, tokens(x))           # Q-Former: queries read the image
    r   = CrossAttn(tokens(x), m)                   # restore one update per visual token
    out = x + alpha * clip(W_out r, |x|)            # alpha = max_residual_ratio * sigmoid(gate)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ADAPTER_WEIGHTS_NAME = "causal_qformer_adapter.pt"


class CausalResidualQFormerAdapter(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        dictionary_shape: tuple[int, int],
        qformer_dim: int = 256,
        num_queries: int = 32,
        num_heads: int = 8,
        intervention_scale: float = 1.0,
        max_residual_ratio: float = 0.01,
        train_confounder_prior: bool = False,
    ) -> None:
        super().__init__()

        if qformer_dim % num_heads != 0:
            raise ValueError(
                f"qformer_dim={qformer_dim} must be divisible by num_heads={num_heads}."
            )

        dictionary_size, dictionary_dim = dictionary_shape
        self.qformer_dim = qformer_dim
        self.intervention_scale = intervention_scale
        self.max_residual_ratio = max_residual_ratio

        # Weights and buffers are placeholders; they are filled by load_state_dict().
        self.register_buffer(
            "confounder_dictionary", torch.zeros(dictionary_size, dictionary_dim)
        )
        if dictionary_dim == qformer_dim:
            self.dictionary_proj = nn.Identity()
        else:
            self.dictionary_proj = nn.Linear(dictionary_dim, qformer_dim, bias=False)

        self.input_proj = nn.Linear(visual_dim, qformer_dim, bias=False)
        self.learnable_queries = nn.Parameter(torch.zeros(1, num_queries, qformer_dim))
        self.query_norm = nn.LayerNorm(qformer_dim)
        self.token_norm = nn.LayerNorm(qformer_dim)
        self.qformer_cross_attention = nn.MultiheadAttention(
            qformer_dim, num_heads, batch_first=True
        )
        self.token_restore_attention = nn.MultiheadAttention(
            qformer_dim, num_heads, batch_first=True
        )
        self.output_proj = nn.Linear(qformer_dim, visual_dim, bias=False)
        self.gate_logit = nn.Parameter(torch.zeros(()))

        if train_confounder_prior:
            self.prior_logits = nn.Parameter(torch.zeros(dictionary_size))
            self.register_buffer("fixed_prior", torch.empty(0), persistent=False)
        else:
            self.prior_logits = None
            self.register_buffer("fixed_prior", torch.zeros(dictionary_size))

    def _prior(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.prior_logits is not None:
            prior = F.softmax(self.prior_logits.float(), dim=0)
        else:
            prior = self.fixed_prior.float()
        return prior.to(device=device, dtype=dtype)

    def _dictionary_context(self, queries: torch.Tensor) -> torch.Tensor:
        dictionary = self.dictionary_proj(self.confounder_dictionary)
        dictionary = dictionary.to(device=queries.device, dtype=queries.dtype)
        prior = self._prior(dtype=queries.dtype, device=queries.device)

        logits = torch.matmul(queries, dictionary.transpose(0, 1))
        logits = logits / math.sqrt(self.qformer_dim)
        attention = F.softmax(logits, dim=-1)

        weighted_attention = attention * prior.view(1, 1, -1)
        return torch.matmul(weighted_attention, dictionary)

    @staticmethod
    def _bound_delta(delta: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Clip each token's update so that ||delta|| <= ||base||."""
        delta_norm = delta.float().norm(dim=-1, keepdim=True)
        base_norm = base.detach().float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        clip_scale = torch.minimum(
            torch.ones_like(delta_norm),
            base_norm / delta_norm.clamp_min(1e-6),
        )
        return delta * clip_scale.to(delta.dtype)

    def effective_gate(self) -> torch.Tensor:
        return self.max_residual_ratio * torch.sigmoid(self.gate_logit)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        # Qwen3-VL yields one [N_tokens, D] tensor per image; accept [B, N, D] too.
        squeeze_batch = visual_tokens.ndim == 2
        if squeeze_batch:
            visual_tokens = visual_tokens.unsqueeze(0)
        elif visual_tokens.ndim != 3:
            raise ValueError(
                "Expected visual tokens with shape [N,D] or [B,N,D], got "
                f"{tuple(visual_tokens.shape)}"
            )

        base = visual_tokens
        tokens_norm = self.token_norm(self.input_proj(base))

        queries = self.query_norm(self.learnable_queries.expand(base.shape[0], -1, -1))
        deconfounded_queries = (
            queries + self.intervention_scale * self._dictionary_context(queries)
        )

        morph_queries, _ = self.qformer_cross_attention(
            query=deconfounded_queries,
            key=tokens_norm,
            value=tokens_norm,
            need_weights=False,
        )
        restored_tokens, _ = self.token_restore_attention(
            query=tokens_norm,
            key=morph_queries,
            value=morph_queries,
            need_weights=False,
        )

        delta = self._bound_delta(self.output_proj(restored_tokens), base)
        alpha = self.effective_gate().to(device=base.device, dtype=base.dtype)
        output = base + alpha * delta

        return output.squeeze(0) if squeeze_batch else output


def load_causal_qformer(checkpoint_dir: str | Path) -> CausalResidualQFormerAdapter:
    """Build the adapter from ``<checkpoint_dir>/causal_qformer_adapter.pt``."""
    payload: dict[str, Any] = torch.load(
        Path(checkpoint_dir) / ADAPTER_WEIGHTS_NAME, map_location="cpu", weights_only=True
    )
    config = payload["config"]
    adapter = CausalResidualQFormerAdapter(
        visual_dim=config["visual_dim"],
        dictionary_shape=tuple(config["dictionary_shape"]),
        qformer_dim=config["qformer_dim"],
        num_queries=config["num_queries"],
        num_heads=config["num_heads"],
        intervention_scale=config["intervention_scale"],
        max_residual_ratio=config["max_residual_ratio"],
        train_confounder_prior=config.get("train_confounder_prior", False),
    )
    adapter.load_state_dict(payload["state_dict"])
    return adapter.eval()


def attach_causal_qformer(model: nn.Module, adapter: CausalResidualQFormerAdapter) -> None:
    """Route every image embedding of a Qwen3VLForConditionalGeneration through ``adapter``."""
    core = model.model  # Qwen3VLModel
    core.causal_qformer = adapter  # register as submodule so .to(device) moves it
    get_image_features = core.get_image_features

    def get_image_features_with_causal_qformer(pixel_values, image_grid_thw=None):
        image_embeds, deepstack_image_embeds = get_image_features(pixel_values, image_grid_thw)
        return tuple(adapter(x) for x in image_embeds), deepstack_image_embeds

    core.get_image_features = get_image_features_with_causal_qformer
