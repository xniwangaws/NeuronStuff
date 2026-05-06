"""DeModLoRALinear / DeModLoRAConv2d — S3Diff LoRA-with-de_mod, trace-friendly.

Replaces the peft-wrapped LoraLayer + `my_lora_fwd` runtime attribute injection
with a proper nn.Module that takes `de_mod` as an explicit forward argument.

Semantics preserved from `~/workspace/s3diff_eager/repo/src/model.py:22-60`:

  result = base(x) + lora_B(einsum('...lk,...kr->...lr', lora_A(dropout(x)), de_mod)) * scaling

Algebra folded (Phase E C-1a, CPU-verified max|diff|=1.8e-7 vs un-folded):

  a  = lora_A(x)                                    # (..., r)
  BD = einsum('or,bkr->bok', lora_B.weight, de_mod) # (B, out, r)
  out_adapter = einsum('bok,...k->...o', BD, a)    # (..., out)
  result = base(x) + out_adapter * scaling

Dropout is dropped (training-only). Dora is not supported (S3Diff doesn't use it).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeModLoRALinear(nn.Module):
    """Linear with a static LoRA adapter modulated by a per-image de_mod matrix.

    Parameters:
      in_features, out_features, lora_rank
      bias: whether base linear has bias (matches base_layer config)
      scaling: LoRA alpha / rank, baked into weights during load
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int,
        bias: bool = True,
        scaling: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.scaling = scaling
        self.base = nn.Linear(in_features, out_features, bias=bias)
        # lora_A: in -> r, lora_B: r -> out, both no bias
        self.lora_A = nn.Linear(in_features, lora_rank, bias=False)
        self.lora_B = nn.Linear(lora_rank, out_features, bias=False)

    def forward(self, x: torch.Tensor, de_mod: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features), de_mod: (B, r, r) -> (..., out_features)."""
        base_out = self.base(x)
        # LoRA A: (..., in) -> (..., r)
        a = self.lora_A(x)
        # BD: fold lora_B.weight @ de_mod  -> (B, out, r)
        # lora_B.weight: (out, r), de_mod: (B, r, r) -> BD: (B, out, r)
        bd = torch.einsum("or,bkr->bok", self.lora_B.weight, de_mod)
        # out_adapter: (B, out, r) + (..., r) -> (..., out)
        # Broadcast B dim if needed; S3Diff always has B=1 at inference
        # a: (..., r), bd: (B, out, r) -> (B, ..., out)
        # For B=1 collapse: use einsum explicit
        out_adapter = torch.einsum("bok,...k->...o", bd, a) if bd.shape[0] == 1 else \
                      torch.einsum("bok,b...k->b...o", bd, a if a.dim() > 1 and a.shape[0] == bd.shape[0] else a.unsqueeze(0).expand(bd.shape[0], *a.shape))
        return base_out + out_adapter * self.scaling


class DeModLoRAConv2d(nn.Module):
    """Conv2d with a static LoRA adapter modulated by a per-image de_mod matrix.

    Parameters:
      in_channels, out_channels, kernel_size, stride, padding, groups
      lora_rank, scaling
      bias: whether base conv has bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        lora_rank: int,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        groups: int = 1,
        bias: bool = True,
        scaling: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lora_rank = lora_rank
        self.scaling = scaling
        self.base = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=groups, bias=bias,
        )
        # peft's LoraLayer for Conv2d creates lora_A as Conv2d(in, r, kernel) and lora_B as Conv2d(r, out, 1)
        self.lora_A = nn.Conv2d(
            in_channels, lora_rank,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=groups, bias=False,
        )
        self.lora_B = nn.Conv2d(lora_rank, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, de_mod: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W), de_mod: (B, r, r) -> (B, out_channels, H_out, W_out)."""
        base_out = self.base(x)
        # lora_A conv: (B, in, H, W) -> (B, r, H', W')
        a = self.lora_A(x)
        # lora_B.weight: (out, r, 1, 1). For einsum we need (out, r).
        lb = self.lora_B.weight.squeeze(-1).squeeze(-1)  # (out, r)
        # BD: (B, out, r) = einsum('or,bkr->bok', lb, de_mod)
        bd = torch.einsum("or,bkr->bok", lb, de_mod)
        # apply BD along channel dim of a: (B, out, H', W') = einsum('bok,bkhw->bohw', bd, a)
        out_adapter = torch.einsum("bok,bkhw->bohw", bd, a)
        return base_out + out_adapter * self.scaling
