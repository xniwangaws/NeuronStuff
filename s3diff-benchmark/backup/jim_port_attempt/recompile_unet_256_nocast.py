"""Recompile UNet at LR=256 with --model-type=unet-inference instead of --auto-cast matmult,
in hopes of eliminating the NaN output."""
import os, math, time, sys, torch, torch_neuronx
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from peft import LoraConfig
from huggingface_hub import hf_hub_download, snapshot_download
from typing import Any

LR = 256
PAD = 1024
LATENT = 128
COMPILE_DIR = os.path.expanduser("~/s3diff_jim/compile_256")


def my_lora_fwd(self, x, *args, **kwargs):
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)
            if not self.use_dora[active_adapter]:
                _tmp = lora_A(dropout(x))
                if isinstance(lora_A, torch.nn.Conv2d):
                    _tmp = torch.einsum('...khw,...kr->...rhw', _tmp, self.de_mod)
                elif isinstance(lora_A, torch.nn.Linear):
                    _tmp = torch.einsum('...lk,...kr->...lr', _tmp, self.de_mod)
                else:
                    raise NotImplementedError
                result = result + lora_B(_tmp) * scaling
            else:
                x = dropout(x)
                result = result + self._apply_dora(x, lora_A, lora_B, scaling, active_adapter)
        result = result.to(torch_result_dtype)
    return result


class UNetWrapper(nn.Module):
    def __init__(self, unet, unet_lora_layers, lora_rank_unet):
        super().__init__()
        self.unet = unet
        self.unet_lora_layers = unet_lora_layers
        self.lora_rank_unet = lora_rank_unet
        self.layer_block_map = {}
        for layer_name in unet_lora_layers:
            split_name = layer_name.split(".")
            if split_name[0] == 'down_blocks':
                self.layer_block_map[layer_name] = int(split_name[1])
            elif split_name[0] == 'mid_block':
                self.layer_block_map[layer_name] = 4
            elif split_name[0] == 'up_blocks':
                self.layer_block_map[layer_name] = int(split_name[1]) + 5
            else:
                self.layer_block_map[layer_name] = 9

    def forward(self, latent, timestep, encoder_hidden_states, de_mod_all):
        for layer_name, module in self.unet.named_modules():
            if layer_name in self.unet_lora_layers:
                block_idx = self.layer_block_map[layer_name]
                module.de_mod = de_mod_all[:, block_idx]
        return self.unet(latent, timestep, encoder_hidden_states=encoder_hidden_states).sample


pretrained_path = hf_hub_download(repo_id="zhangap/S3Diff", filename="s3diff.pkl")
sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")

unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
sd = torch.load(pretrained_path, map_location="cpu", weights_only=False)
lora_rank_unet = sd["rank_unet"]

unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                               target_modules=sd["unet_lora_target_modules"])
unet.add_adapter(unet_lora_config)
_sd_unet = unet.state_dict()
for k in sd["state_dict_unet"]:
    _sd_unet[k] = sd["state_dict_unet"][k]
unet.load_state_dict(_sd_unet)

unet_lora_layers = []
for name, module in unet.named_modules():
    if 'base_layer' in name:
        unet_lora_layers.append(name[:-len(".base_layer")])
for name, module in unet.named_modules():
    if name in unet_lora_layers:
        module.forward = my_lora_fwd.__get__(module, module.__class__)
unet.eval()

print(f"UNet LoRA layers: {len(unet_lora_layers)}")

flag_sets = [
    ("unet_inference_O1", ['--model-type=unet-inference', '-O1']),
    # backup: no cast, O1
    # ("no_cast_O1", ['-O1']),
]

for tag, flags in flag_sets:
    print(f"\n--- Trying UNet recompile flags={flags} ---")
    wrapper = UNetWrapper(unet, unet_lora_layers, lora_rank_unet)
    example_latent = torch.randn(1, 4, LATENT, LATENT)
    example_timestep = torch.tensor([999], dtype=torch.long)
    example_enc = torch.randn(1, 77, 1024)
    example_unet_de_mod = torch.randn(1, 10, lora_rank_unet, lora_rank_unet)
    try:
        t0 = time.time()
        m = torch_neuronx.trace(wrapper,
            (example_latent, example_timestep, example_enc, example_unet_de_mod),
            compiler_args=flags)
        dt = time.time() - t0
        print(f"  SUCCESS in {dt:.1f}s")
        outp = os.path.join(COMPILE_DIR, f"unet_{tag}.pt")
        torch.jit.save(m, outp)
        print(f"  Saved: {outp}")
        # Also rename current unet.pt to _matmult and put this one in place
        import shutil
        old = os.path.join(COMPILE_DIR, "unet.pt")
        bak = os.path.join(COMPILE_DIR, "unet_matmult.pt")
        if os.path.exists(old) and not os.path.exists(bak):
            shutil.move(old, bak)
            print(f"  Backed up old unet.pt -> unet_matmult.pt")
        shutil.copy(outp, old)
        print(f"  Active unet.pt now = {tag}")
        break
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")
