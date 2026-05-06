"""NeuronS3DiffResnetBlock2D — diffusers ResnetBlock2D replacement with DeModLoRAConv2d.

Diffusers ResnetBlock2D forward (time embedding variant):
    h = GroupNorm(x) -> SiLU -> Conv2d(conv1)  -> + time_emb_proj(t).unsqueeze(-1,-1)
    h = GroupNorm(h) -> SiLU -> Dropout -> Conv2d(conv2)
    return h + conv_shortcut(x) (if channels mismatch) else h + x

In S3Diff the conv1/conv2 (and conv_shortcut if present) are DeModLoRAConv2d.
The time_emb_proj is plain nn.Linear (no LoRA per S3Diff target_modules_unet).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from de_mod_lora import DeModLoRAConv2d


class NeuronS3DiffResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        lora_rank: int = 32,
        lora_scaling: float = 0.25,
        groups: int = 32,
        eps: float = 1e-5,
        use_shortcut: Optional[bool] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lora_rank = lora_rank

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = DeModLoRAConv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, lora_rank=lora_rank,
            bias=True, scaling=lora_scaling,
        )

        self.time_emb_proj = nn.Linear(temb_channels, out_channels, bias=True)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = DeModLoRAConv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, lora_rank=lora_rank,
            bias=True, scaling=lora_scaling,
        )

        # Shortcut: diffusers creates conv_shortcut when in_channels != out_channels
        self.use_shortcut = (use_shortcut if use_shortcut is not None
                             else in_channels != out_channels)
        if self.use_shortcut:
            self.conv_shortcut = DeModLoRAConv2d(
                in_channels, out_channels,
                kernel_size=1, padding=0, lora_rank=lora_rank,
                bias=True, scaling=lora_scaling,
            )
        else:
            self.conv_shortcut = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        de_mod_conv1: torch.Tensor,
        de_mod_conv2: torch.Tensor,
        de_mod_conv_shortcut: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        h = self.norm1(hidden_states)
        h = F.silu(h)
        h = self.conv1(h, de_mod_conv1)

        # Apply time embedding: diffusers uses silu then linear
        t = F.silu(temb)
        t = self.time_emb_proj(t).unsqueeze(-1).unsqueeze(-1)  # (B, out_c, 1, 1)
        h = h + t

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h, de_mod_conv2)

        if self.use_shortcut:
            assert de_mod_conv_shortcut is not None, "conv_shortcut requires de_mod"
            residual = self.conv_shortcut(residual, de_mod_conv_shortcut)

        return h + residual
