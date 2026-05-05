"""Adapter — make NeuronS3DiffUNet drop-in callable from S3Diff's forward.

S3Diff's forward in s3diff_tile.py calls:
  pos_model_pred = self.unet(lq_latent, self.timesteps,
                              encoder_hidden_states=pos_caption_enc).sample

It also assigns `module.de_mod = ...` on every module in `self.unet_lora_layers`
list BEFORE the forward. So our adapter must:

1. Expose `.sample` attribute on its return type
2. Catch the `module.de_mod` assignments and build the de_mod_map that our
   NeuronS3DiffUNet.forward expects
3. Match signature `(sample, timestep, encoder_hidden_states=..., return_dict=...)`

We do this by keeping a *dummy* diffusers UNet as the outward-facing shell
(so `named_modules()` still works for de_mod assignment), and replacing the
`.forward` with one that redirects to our NeuronS3DiffUNet.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from collections import namedtuple


UNet2DConditionOutput = namedtuple("UNet2DConditionOutput", ["sample"])


class _AdapterModule(nn.Module):
    """Helper nn.Module that exposes a `.de_mod` attribute settable from outside."""
    def __init__(self):
        super().__init__()
        self.de_mod = None  # set externally by S3Diff's forward


def wrap_custom_unet(custom_unet, ref_unet):
    """Wrap `custom_unet` to look like `ref_unet` for S3Diff's forward loop.

    Steps:
      1. Keep `ref_unet` intact (we rely on its submodule tree for de_mod
         name collection).
      2. Monkey-patch ref_unet's `forward` to:
           - read `de_mod` from each peft module
           - build de_mod_map
           - call custom_unet.forward(...)
           - wrap output in UNet2DConditionOutput

    Returns `ref_unet` (the wrapper), whose forward now redirects to custom_unet.
    """
    from peft.tuners.lora.layer import Linear as PeftLinear, Conv2d as PeftConv2d

    # Collect (path -> peft_module) mapping once at wrap time
    lora_modules = {}
    for name, m in ref_unet.named_modules():
        if isinstance(m, (PeftLinear, PeftConv2d)):
            lora_modules[name] = m

    def new_forward(sample, timestep, encoder_hidden_states=None, **kwargs):
        # Harvest de_mod attrs that S3Diff set before calling us
        de_mod_map = {}
        for name, m in lora_modules.items():
            de_mod = getattr(m, "de_mod", None)
            if de_mod is not None:
                de_mod_map[name] = de_mod

        # Also move custom_unet to the same dtype/device as ref_unet on first call
        # (S3Diff calls .to() on ref_unet; we need to mirror for our custom_unet)
        first_param = next(iter(ref_unet.parameters()))
        target_device = first_param.device
        target_dtype = first_param.dtype
        if next(iter(custom_unet.parameters())).device != target_device or \
           next(iter(custom_unet.parameters())).dtype != target_dtype:
            custom_unet.to(device=target_device, dtype=target_dtype)

        out = custom_unet(
            sample=sample, timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            de_mod_map=de_mod_map,
        )

        if kwargs.get("return_dict", True):
            return UNet2DConditionOutput(sample=out)
        return (out,)

    ref_unet.forward = new_forward
    return ref_unet
