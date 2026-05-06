"""DeModLoRA{Linear,Conv2d} variants that read de_mod from a module attribute.

These are drop-in replacements for peft's LoraLayer (linear/conv2d) that preserve
the dynamic de_mod semantics of S3Diff's my_lora_fwd, but use the folded einsum
from Phase E C-1a for efficiency.

Unlike the forward-arg variant in de_mod_lora.py, these modules read
`self.de_mod` at forward time. The outer wrapper (e.g. S3Diff main forward)
is responsible for setting it before each UNet call. In torch_neuronx.trace,
`self.de_mod` can be a view into a shape-stable shared buffer (Path A pattern)
so trace captures the buffer read rather than baking a constant.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DeModLoRALinearAttr(nn.Module):
    """Linear + LoRA modulated by self.de_mod (set externally)."""

    def __init__(self, in_features, out_features, lora_rank, bias=True, scaling=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.scaling = scaling
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A = nn.Linear(in_features, lora_rank, bias=False)
        self.lora_B = nn.Linear(lora_rank, out_features, bias=False)
        # de_mod is a tensor attribute set externally before forward. Placeholder:
        self.de_mod = None  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        de_mod = self.de_mod  # (B, r, r)
        base_out = self.base(x)
        a = self.lora_A(x)                                    # (..., r)
        bd = torch.einsum("or,bkr->bok", self.lora_B.weight, de_mod)  # (B, out, r)
        # B=1 fast path
        if bd.shape[0] == 1:
            out_adapter = torch.einsum("ok,...k->...o", bd[0], a)
        else:
            out_adapter = torch.einsum("bok,b...k->b...o", bd, a if a.dim() > 1 and a.shape[0] == bd.shape[0] else a.unsqueeze(0).expand(bd.shape[0], *a.shape))
        return base_out + out_adapter * self.scaling


class DeModLoRAConv2dAttr(nn.Module):
    """Conv2d + LoRA modulated by self.de_mod (set externally)."""

    def __init__(
        self, in_channels, out_channels, kernel_size, lora_rank,
        stride=1, padding=0, groups=1, bias=True, scaling=1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lora_rank = lora_rank
        self.scaling = scaling
        self.base = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups, bias=bias)
        # peft LoraLayer for Conv2d: lora_A preserves spatial (in -> r, same kernel),
        # lora_B is 1x1 (r -> out)
        self.lora_A = nn.Conv2d(in_channels, lora_rank, kernel_size=kernel_size,
                                stride=stride, padding=padding, groups=groups, bias=False)
        self.lora_B = nn.Conv2d(lora_rank, out_channels, kernel_size=1, bias=False)
        self.de_mod = None  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        de_mod = self.de_mod  # (B, r, r)
        base_out = self.base(x)
        a = self.lora_A(x)   # (B, r, H', W')
        lb = self.lora_B.weight.squeeze(-1).squeeze(-1)  # (out, r)
        bd = torch.einsum("or,bkr->bok", lb, de_mod)    # (B, out, r)
        out_adapter = torch.einsum("bok,bkhw->bohw", bd, a)
        return base_out + out_adapter * self.scaling


def convert_peft_linear_to_demod(peft_module, adapter_name="default"):
    """Return a new DeModLoRALinearAttr with weights copied from a peft LoraLayer-wrapped Linear."""
    base = peft_module.base_layer
    lora_A = peft_module.lora_A[adapter_name]
    lora_B = peft_module.lora_B[adapter_name]
    scaling = float(peft_module.scaling[adapter_name])
    new = DeModLoRALinearAttr(
        in_features=base.in_features,
        out_features=base.out_features,
        lora_rank=lora_A.out_features,  # rank
        bias=base.bias is not None,
        scaling=scaling,
    )
    with torch.no_grad():
        new.base.weight.copy_(base.weight)
        if base.bias is not None:
            new.base.bias.copy_(base.bias)
        new.lora_A.weight.copy_(lora_A.weight)
        new.lora_B.weight.copy_(lora_B.weight)
    return new


def convert_peft_conv2d_to_demod(peft_module, adapter_name="default"):
    """Return a new DeModLoRAConv2dAttr with weights copied from a peft LoraLayer-wrapped Conv2d."""
    base = peft_module.base_layer
    lora_A = peft_module.lora_A[adapter_name]
    lora_B = peft_module.lora_B[adapter_name]
    scaling = float(peft_module.scaling[adapter_name])
    new = DeModLoRAConv2dAttr(
        in_channels=base.in_channels,
        out_channels=base.out_channels,
        kernel_size=base.kernel_size,
        lora_rank=lora_A.out_channels,
        stride=base.stride,
        padding=base.padding,
        groups=base.groups,
        bias=base.bias is not None,
        scaling=scaling,
    )
    with torch.no_grad():
        new.base.weight.copy_(base.weight)
        if base.bias is not None:
            new.base.bias.copy_(base.bias)
        new.lora_A.weight.copy_(lora_A.weight)
        new.lora_B.weight.copy_(lora_B.weight)
    return new


def replace_lora_modules_in_unet(unet, adapter_name="default"):
    """Walk UNet, replace every peft LoraLayer (Linear or Conv2d type) with our DeModLoRA* variant.

    Returns a mapping {module_path: new_module} for debug.
    """
    from peft.tuners.lora.layer import Linear as PeftLinear, Conv2d as PeftConv2d

    replaced = {}

    # Walk once to find targets
    targets = []
    for name, m in unet.named_modules():
        if isinstance(m, PeftLinear):
            targets.append((name, "linear", m))
        elif isinstance(m, PeftConv2d):
            targets.append((name, "conv2d", m))

    # Replace in-place
    for name, kind, m in targets:
        if kind == "linear":
            new = convert_peft_linear_to_demod(m, adapter_name)
        else:
            new = convert_peft_conv2d_to_demod(m, adapter_name)

        # Install into parent
        parent_name, _, attr = name.rpartition(".")
        parent = unet.get_submodule(parent_name) if parent_name else unet
        # If attr is numeric index (ff.net.2), use setattr on ModuleList requires int
        if attr.isdigit():
            parent[int(attr)] = new
        else:
            setattr(parent, attr, new)

        replaced[name] = new

    return replaced
