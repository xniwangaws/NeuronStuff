"""NeuronS3DiffAttention — trace-friendly self / cross attention with DeModLoRA.

Replaces diffusers' Attention class for the S3Diff UNet use case. Minimal forward:
no attention mask, no upcasting options, no processor dispatch — just q/k/v/out
linears (each LoRA-modulated) plus scaled dot product attention.

For cross-attention, encoder_hidden_states is supplied; for self-attention,
we fall back to hidden_states.

Correctness reference: diffusers.models.attention_processor.Attention with
AttnProcessor2_0 (scaled_dot_product_attention path).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from de_mod_lora import DeModLoRALinear


class NeuronS3DiffAttention(nn.Module):
    """Attention with four DeModLoRALinear projections.

    Args:
      query_dim: input feature dim for Q (== hidden size)
      cross_attention_dim: input feature dim for K/V (== hidden size for self-attn,
                           == CLIP hidden size for cross-attn). Default = query_dim.
      heads: number of attention heads
      dim_head: per-head dim (so inner_dim = heads * dim_head)
      lora_rank: LoRA rank (32 for UNet, 16 for VAE)
      scaling: LoRA scaling (alpha/rank)
      out_bias: whether to_out[0] has bias (diffusers default True)
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        lora_rank: int = 32,
        scaling: float = 1.0,
        out_bias: bool = True,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim or query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = 1.0 / math.sqrt(dim_head)

        # Diffusers Attention: to_q/k/v have bias=False by default (unless
        # bias=True at construction). S3Diff's state_dict shows no to_q/k/v
        # bias (only .weight), so bias=False.
        self.to_q = DeModLoRALinear(query_dim, self.inner_dim, lora_rank,
                                    bias=False, scaling=scaling)
        self.to_k = DeModLoRALinear(self.cross_attention_dim, self.inner_dim, lora_rank,
                                    bias=False, scaling=scaling)
        self.to_v = DeModLoRALinear(self.cross_attention_dim, self.inner_dim, lora_rank,
                                    bias=False, scaling=scaling)
        # to_out in diffusers is a ModuleList [Linear, Dropout]; we only need the Linear
        # at index 0. We host it under `to_out_0` and expose `to_out` as a trivial
        # ModuleList for state_dict compatibility.
        self.to_out_0 = DeModLoRALinear(self.inner_dim, query_dim, lora_rank,
                                        bias=out_bias, scaling=scaling)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """(B, seq, inner_dim) -> (B, heads, seq, dim_head)."""
        B, seq, _ = t.shape
        return t.view(B, seq, self.heads, self.dim_head).transpose(1, 2).contiguous()

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        """(B, heads, seq, dim_head) -> (B, seq, inner_dim)."""
        B, _, seq, _ = t.shape
        return t.transpose(1, 2).contiguous().view(B, seq, self.inner_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        de_mod_q: torch.Tensor,
        de_mod_k: torch.Tensor,
        de_mod_v: torch.Tensor,
        de_mod_out: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        hidden_states: (B, seq_q, query_dim)
        encoder_hidden_states: (B, seq_kv, cross_attention_dim) or None (for self-attn)
        de_mod_*: (B, lora_rank, lora_rank) each
        """
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        q = self.to_q(hidden_states, de_mod_q)              # (B, seq_q, inner)
        k = self.to_k(encoder_hidden_states, de_mod_k)      # (B, seq_kv, inner)
        v = self.to_v(encoder_hidden_states, de_mod_v)      # (B, seq_kv, inner)

        q = self._split_heads(q)  # (B, H, seq_q, d)
        k = self._split_heads(k)  # (B, H, seq_kv, d)
        v = self._split_heads(v)  # (B, H, seq_kv, d)

        # scaled dot product attention (matches diffusers AttnProcessor2_0)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        out = self._merge_heads(out)                        # (B, seq_q, inner)
        out = self.to_out_0(out, de_mod_out)                # (B, seq_q, query_dim)
        return out
