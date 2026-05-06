"""Retry VAE encoder compile for LR=256 with --model-type=unet-inference (Jim's VAE-dec flag)."""
import os, sys, math, time, torch
sys.path.insert(0, os.path.expanduser("~/s3diff_jim"))
os.environ["LR"] = "256"
os.environ["MODE"] = "resume_vae_enc"

# Import everything from run_jim_s3diff up through model build, then just re-try VAE enc.
# Simplest approach: copy the exact loader here.
import math, json, numpy as np, glob
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig
from huggingface_hub import hf_hub_download, snapshot_download
from typing import Any
import torch_neuronx

LR_H = LR_W = 256
SF = 4
HR_H, HR_W = LR_H * SF, LR_W * SF
PAD_H = (math.ceil(HR_H / 64)) * 64
PAD_W = (math.ceil(HR_W / 64)) * 64
LATENT_H, LATENT_W = PAD_H // 8, PAD_W // 8
COMPILE_DIR = os.path.expanduser("~/s3diff_jim/compile_256")

print(f"Retry VAE encoder at LR={LR_H} -> pad={PAD_H}")

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


class VAEEncoderWrapper(nn.Module):
    def __init__(self, vae, vae_lora_layers, lora_rank_vae):
        super().__init__()
        self.vae = vae
        self.vae_lora_layers = vae_lora_layers
        self.lora_rank_vae = lora_rank_vae
        self.layer_block_map = {}
        for layer_name in vae_lora_layers:
            split_name = layer_name.split(".")
            if split_name[1] == 'down_blocks':
                self.layer_block_map[layer_name] = int(split_name[2])
            elif split_name[1] == 'mid_block':
                self.layer_block_map[layer_name] = 4
            else:
                self.layer_block_map[layer_name] = 5

    def forward(self, pixel_values, de_mod_all):
        for layer_name, module in self.vae.named_modules():
            if layer_name in self.vae_lora_layers:
                block_idx = self.layer_block_map[layer_name]
                module.de_mod = de_mod_all[:, block_idx]
        latent = self.vae.encode(pixel_values).latent_dist.sample() * self.vae.config.scaling_factor
        return latent


pretrained_path = hf_hub_download(repo_id="zhangap/S3Diff", filename="s3diff.pkl")
sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")

vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
sd = torch.load(pretrained_path, map_location="cpu", weights_only=False)
lora_rank_vae = sd["rank_vae"]
vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                              target_modules=sd["vae_lora_target_modules"])
vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
_sd_vae = vae.state_dict()
for k in sd["state_dict_vae"]:
    _sd_vae[k] = sd["state_dict_vae"][k]
vae.load_state_dict(_sd_vae)

vae_lora_layers = []
for name, module in vae.named_modules():
    if 'base_layer' in name:
        vae_lora_layers.append(name[:-len(".base_layer")])
for name, module in vae.named_modules():
    if name in vae_lora_layers:
        module.forward = my_lora_fwd.__get__(module, module.__class__)

vae.eval()

print(f"VAE LoRA layers: {len(vae_lora_layers)}")

flag_sets = [
    ("unet_inference_O1", ['--model-type=unet-inference', '-O1']),
    ("unet_inference_O2", ['--model-type=unet-inference', '-O2']),
    ("auto_cast_all_O1", ['--auto-cast', 'all', '-O1']),
]

for tag, flags in flag_sets:
    print(f"\n--- Trying VAE encoder with flags={flags} ---")
    try:
        vae_enc_wrapper = VAEEncoderWrapper(vae, vae_lora_layers, lora_rank_vae)
        ep = torch.randn(1, 3, PAD_H, PAD_W)
        ed = torch.randn(1, 6, lora_rank_vae, lora_rank_vae)
        t0 = time.time()
        m = torch_neuronx.trace(vae_enc_wrapper, (ep, ed), compiler_args=flags)
        dt = time.time() - t0
        print(f"  SUCCESS in {dt:.1f}s with flags={flags}")
        out = os.path.join(COMPILE_DIR, f"vae_encoder_{tag}.pt")
        torch.jit.save(m, out)
        # Also save as the canonical name for the downstream script
        torch.jit.save(m, os.path.join(COMPILE_DIR, "vae_encoder.pt"))
        print(f"  Saved to {out}")
        with open(os.path.join(COMPILE_DIR, "vae_enc_winning_flags.json"), "w") as f:
            json.dump({"flags": flags, "tag": tag, "compile_time_s": dt}, f, indent=2)
        break
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:400]}")
