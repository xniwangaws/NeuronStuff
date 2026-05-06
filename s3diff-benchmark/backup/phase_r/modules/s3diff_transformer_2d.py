"""NeuronS3DiffTransformer2DModel — wraps BasicTransformerBlock with proj_in/proj_out.

Replaces diffusers.Transformer2DModel with num_layers=1 (SD-Turbo config).

Forward:
  x_4d: (B, C, H, W)
  -> GroupNorm
  -> permute (B, H*W, C) + proj_in
  -> BasicTransformerBlock
  -> proj_out
  -> reshape (B, C, H, W) + residual

proj_in and proj_out are DeModLoRALinear in S3Diff config (1x1 conv equivalent).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from de_mod_lora import DeModLoRALinear
from s3diff_transformer_block import NeuronS3DiffBasicTransformerBlock


class NeuronS3DiffTransformer2DModel(nn.Module):
    """diffusers Transformer2DModel replacement for S3Diff's SD-Turbo UNet.

    Assumes:
      - num_layers = 1 (one BasicTransformerBlock)
      - use_linear_projection = True  (proj_in/out are Linear, not Conv2d 1x1)
      - attention_type = "default"
      - norm_type = "layer_norm" inside BasicTransformerBlock
      - norm (outer) is GroupNorm(32, in_channels)
    """

    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        lora_rank: int = 32,
        lora_scaling: float = 0.25,
        num_groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=eps,
                                 affine=True)
        self.proj_in = DeModLoRALinear(in_channels, self.inner_dim, lora_rank,
                                       bias=True, scaling=lora_scaling)
        self.transformer_blocks = nn.ModuleList([
            NeuronS3DiffBasicTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
                lora_rank=lora_rank,
                lora_scaling=lora_scaling,
            )
        ])
        self.proj_out = DeModLoRALinear(self.inner_dim, in_channels, lora_rank,
                                        bias=True, scaling=lora_scaling)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        # de_mod for proj_in / proj_out
        de_mod_proj_in: torch.Tensor,
        de_mod_proj_out: torch.Tensor,
        # de_mod for BasicTransformerBlock (10 tensors)
        de_mod_attn1_q: torch.Tensor,
        de_mod_attn1_k: torch.Tensor,
        de_mod_attn1_v: torch.Tensor,
        de_mod_attn1_o: torch.Tensor,
        de_mod_attn2_q: torch.Tensor,
        de_mod_attn2_k: torch.Tensor,
        de_mod_attn2_v: torch.Tensor,
        de_mod_attn2_o: torch.Tensor,
        de_mod_ff_0: torch.Tensor,
        de_mod_ff_2: torch.Tensor,
    ) -> torch.Tensor:
        """hidden_states: (B, C, H, W); encoder_hidden_states: (B, 77, cross_dim)."""
        B, C, H, W = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        # (B, C, H, W) -> (B, H*W, C)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(B, H * W, C)
        hidden_states = self.proj_in(hidden_states, de_mod_proj_in)

        hidden_states = self.transformer_blocks[0](
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            de_mod_attn1_q=de_mod_attn1_q,
            de_mod_attn1_k=de_mod_attn1_k,
            de_mod_attn1_v=de_mod_attn1_v,
            de_mod_attn1_o=de_mod_attn1_o,
            de_mod_attn2_q=de_mod_attn2_q,
            de_mod_attn2_k=de_mod_attn2_k,
            de_mod_attn2_v=de_mod_attn2_v,
            de_mod_attn2_o=de_mod_attn2_o,
            de_mod_ff_0=de_mod_ff_0,
            de_mod_ff_2=de_mod_ff_2,
        )
        hidden_states = self.proj_out(hidden_states, de_mod_proj_out)

        # (B, H*W, C) -> (B, C, H, W)
        hidden_states = hidden_states.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return hidden_states + residual
