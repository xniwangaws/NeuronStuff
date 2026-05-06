"""NeuronS3DiffBasicTransformerBlock — a diffusers BasicTransformerBlock replacement.

Layout (match diffusers BasicTransformerBlock with norm_type="layer_norm"):
  x -> norm1 -> attn1(self, hidden) + x
    -> norm2 -> attn2(cross, encoder_hidden) + x  (if encoder_hidden_states given)
    -> norm3 -> ff(x)                    + x
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from de_mod_lora import DeModLoRALinear
from s3diff_attention import NeuronS3DiffAttention


class NeuronS3DiffGEGLU(nn.Module):
    """GEGLU activation — 2 linears (proj: in->2*hidden) split + GELU gate + linear(hidden->out).

    diffusers' FeedForward uses GEGLU when act_fn="geglu":
      ff.net[0] = GEGLU(in, hidden*2) — ff.net.0.proj is DeModLoRA(in, hidden*2)
      ff.net[1] = Dropout (skip)
      ff.net[2] = DeModLoRA(hidden, out)
    """

    def __init__(self, dim_in: int, dim_out: int, lora_rank: int, scaling: float = 1.0):
        super().__init__()
        # proj: dim_in -> dim_out * 2 (split into gate + value)
        self.proj = DeModLoRALinear(dim_in, dim_out * 2, lora_rank, bias=True, scaling=scaling)

    def forward(self, x: torch.Tensor, de_mod_proj: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(x, de_mod_proj)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class NeuronS3DiffFeedForward(nn.Module):
    """Matches diffusers FeedForward with activation="geglu", mult=4."""

    def __init__(self, dim: int, mult: int = 4, lora_rank: int = 32, scaling: float = 1.0):
        super().__init__()
        inner_dim = dim * mult
        # Under diffusers naming: ff.net = ModuleList([GEGLU(dim, inner), Dropout, Linear(inner, dim), Dropout])
        # The LoRA sites are: ff.net.0.proj  and  ff.net.2
        self.net_0 = NeuronS3DiffGEGLU(dim, inner_dim, lora_rank=lora_rank, scaling=scaling)
        self.net_2 = DeModLoRALinear(inner_dim, dim, lora_rank=lora_rank, bias=True, scaling=scaling)

    def forward(self, x: torch.Tensor,
                de_mod_ff_0_proj: torch.Tensor,
                de_mod_ff_2: torch.Tensor) -> torch.Tensor:
        h = self.net_0(x, de_mod_ff_0_proj)
        h = self.net_2(h, de_mod_ff_2)
        return h


class NeuronS3DiffBasicTransformerBlock(nn.Module):
    """Matches diffusers BasicTransformerBlock with:
      - norm_type="layer_norm"
      - num_embeds_ada_norm=None (no AdaLN)
      - cross_attention_dim != None (both attn1 and attn2 active)

    Forward graph:
      h1 = attn1(norm1(x))                  + x
      h2 = attn2(norm2(h1), encoder_hidden) + h1
      out= ff   (norm3(h2))                 + h2
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        lora_rank: int = 32,
        lora_scaling: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.cross_attention_dim = cross_attention_dim
        self.lora_rank = lora_rank

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True)
        self.attn1 = NeuronS3DiffAttention(
            query_dim=dim,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            lora_rank=lora_rank,
            scaling=lora_scaling,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True)
        self.attn2 = NeuronS3DiffAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            lora_rank=lora_rank,
            scaling=lora_scaling,
        )
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=True)
        self.ff = NeuronS3DiffFeedForward(dim, mult=4, lora_rank=lora_rank, scaling=lora_scaling)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        # de_mod for attn1 (self): q, k, v, to_out.0
        de_mod_attn1_q: torch.Tensor,
        de_mod_attn1_k: torch.Tensor,
        de_mod_attn1_v: torch.Tensor,
        de_mod_attn1_o: torch.Tensor,
        # de_mod for attn2 (cross)
        de_mod_attn2_q: torch.Tensor,
        de_mod_attn2_k: torch.Tensor,
        de_mod_attn2_v: torch.Tensor,
        de_mod_attn2_o: torch.Tensor,
        # de_mod for ff
        de_mod_ff_0: torch.Tensor,
        de_mod_ff_2: torch.Tensor,
    ) -> torch.Tensor:
        h = self.attn1(self.norm1(hidden_states), None,
                       de_mod_attn1_q, de_mod_attn1_k, de_mod_attn1_v, de_mod_attn1_o)
        hidden_states = h + hidden_states

        h = self.attn2(self.norm2(hidden_states), encoder_hidden_states,
                       de_mod_attn2_q, de_mod_attn2_k, de_mod_attn2_v, de_mod_attn2_o)
        hidden_states = h + hidden_states

        h = self.ff(self.norm3(hidden_states), de_mod_ff_0, de_mod_ff_2)
        hidden_states = h + hidden_states
        return hidden_states
