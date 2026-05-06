# [md cell 0]: # S3Diff One-Step Super-Resolution on AWS Neuron |  | This notebook demonstrates how to compile and run **S3Diff** -- a one-step 4x super-resolution model -- on trn2.3xlarge using `torch_neuronx.trace()`.

# [md cell 1]: ## Instance Setup |  | This notebook is designed to run on a **trn2.3xlarge** with the **Deep Learning AMI Neuron (Ubuntu 24.04)**. |  | To launch your instance, use the AWS Console or CLI. For a walkthrough 

# [md cell 2]: ## 1. Environment Setup

# --- cell 3 ---
# Verify Neuron devices
!neuron-ls

# --- cell 4 ---
# Install dependencies not in the DLAMI
!pip install -q diffusers peft einops
!pip install -q git+https://github.com/ArcticHare105/S3Diff.git

# --- cell 5 ---
import os
import sys
import math
import re
import time
import json
import numpy as np
import torch
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

print(f"torch_neuronx: {torch_neuronx.__version__}")
print(f"torch: {torch.__version__}")

# [md cell 6]: ## 2. S3Diff Architecture |  | S3Diff's pipeline has these components: |  | | Component | Size | Traceable? | Notes | | |-----------|------|-----------|-------| | | DEResNet (degradation estimator) | ~2.4M params

# [md cell 7]: ## 3. Define Helper Classes

# --- cell 8 ---
# Custom LoRA forward with degradation modulation
def my_lora_fwd(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """LoRA forward that injects degradation modulation between lora_A and lora_B."""
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

# --- cell 9 ---
# DEResNet - degradation estimator (from basicsr)
class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class DEResNet(nn.Module):
    def __init__(self, num_in_ch=3, num_degradation=2,
                 num_feats=[64, 64, 64, 128], num_blocks=[2, 2, 2, 2],
                 downscales=[1, 1, 2, 1]):
        super().__init__()
        num_stage = len(num_feats)
        self.conv_first = nn.ModuleList()
        for _ in range(num_degradation):
            self.conv_first.append(nn.Conv2d(num_in_ch, num_feats[0], 3, 1, 1))
        self.body = nn.ModuleList()
        for _ in range(num_degradation):
            body = []
            for stage in range(num_stage):
                for _ in range(num_blocks[stage]):
                    body.append(ResidualBlockNoBN(num_feats[stage]))
                if downscales[stage] == 1:
                    if stage < num_stage - 1 and num_feats[stage] != num_feats[stage + 1]:
                        body.append(nn.Conv2d(num_feats[stage], num_feats[stage + 1], 3, 1, 1))
                elif downscales[stage] == 2:
                    body.append(nn.Conv2d(num_feats[stage], num_feats[min(stage + 1, num_stage - 1)], 3, 2, 1))
            self.body.append(nn.Sequential(*body))
        self.num_degradation = num_degradation
        self.fc_degree = nn.ModuleList()
        for _ in range(num_degradation):
            self.fc_degree.append(nn.Sequential(
                nn.Linear(num_feats[-1], 512), nn.ReLU(inplace=True),
                nn.Linear(512, 1), nn.Sigmoid()))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        degrees = []
        for i in range(self.num_degradation):
            x_out = self.conv_first[i](x)
            feat = self.body[i](x_out)
            feat = self.avg_pool(feat).squeeze(-1).squeeze(-1)
            degrees.append(self.fc_degree[i](feat).squeeze(-1))
        return torch.stack(degrees, dim=1)

# --- cell 10 ---
# Wrapper modules for Neuron tracing
# These accept de_mod tensors as explicit inputs instead of relying on attribute mutation

class VAEEncoderWrapper(nn.Module):
    """Wraps VAE encoder to accept de_mod_all [B, 6, rank, rank] as explicit input."""
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


class UNetWrapper(nn.Module):
    """Wraps UNet to accept de_mod_all [B, 10, rank, rank] as explicit input."""
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


class VAEDecoderWrapper(nn.Module):
    """Simple wrapper for VAE decoder (no LoRA)."""
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    def forward(self, latent):
        return self.vae.decode(latent / self.vae.config.scaling_factor).sample


class TextEncoderWrapper(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder
    def forward(self, input_ids):
        return self.text_encoder(input_ids)[0]

# [md cell 11]: ## 4. Download and Build Model

# --- cell 12 ---
# Download S3Diff weights and SD-Turbo base model
pretrained_path = hf_hub_download(repo_id="zhangap/S3Diff", filename="s3diff.pkl")
sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
print(f"S3Diff weights: {pretrained_path}")
print(f"SD-Turbo: {sd_path}")

# --- cell 13 ---
# Clone S3Diff repo for DEResNet weights
!git clone --depth 1 https://github.com/ArcticHare105/S3Diff.git /tmp/s3diff_repo 2>/dev/null || echo 'Already cloned'
de_net_path = '/tmp/s3diff_repo/assets/mm-realsr/de_net.pth'
print(f"DEResNet weights: {de_net_path}")

# --- cell 14 ---
# Build model components
print("Loading SD-Turbo + S3Diff LoRA weights...")

tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").eval()
vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")

sd = torch.load(pretrained_path, map_location="cpu")
lora_rank_unet = sd["rank_unet"]
lora_rank_vae = sd["rank_vae"]
print(f"LoRA ranks: unet={lora_rank_unet}, vae={lora_rank_vae}")

# Add LoRA adapters and load trained weights
vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                              target_modules=sd["vae_lora_target_modules"])
vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
_sd_vae = vae.state_dict()
for k in sd["state_dict_vae"]:
    _sd_vae[k] = sd["state_dict_vae"][k]
vae.load_state_dict(_sd_vae)

unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                               target_modules=sd["unet_lora_target_modules"])
unet.add_adapter(unet_lora_config)
_sd_unet = unet.state_dict()
for k in sd["state_dict_unet"]:
    _sd_unet[k] = sd["state_dict_unet"][k]
unet.load_state_dict(_sd_unet)

# Monkey-patch LoRA forward methods with degradation modulation
vae_lora_layers = []
for name, module in vae.named_modules():
    if 'base_layer' in name:
        vae_lora_layers.append(name[:-len(".base_layer")])
for name, module in vae.named_modules():
    if name in vae_lora_layers:
        module.forward = my_lora_fwd.__get__(module, module.__class__)

unet_lora_layers = []
for name, module in unet.named_modules():
    if 'base_layer' in name:
        unet_lora_layers.append(name[:-len(".base_layer")])
for name, module in unet.named_modules():
    if name in unet_lora_layers:
        module.forward = my_lora_fwd.__get__(module, module.__class__)

vae.eval()
unet.eval()

# Modulation MLPs (tiny, run on CPU)
num_embeddings = 64
block_embedding_dim = 64
W = nn.Parameter(sd["w"], requires_grad=False)

vae_de_mlp = nn.Sequential(nn.Linear(num_embeddings * 4, 256), nn.ReLU(True))
unet_de_mlp = nn.Sequential(nn.Linear(num_embeddings * 4, 256), nn.ReLU(True))
vae_block_mlp = nn.Sequential(nn.Linear(block_embedding_dim, 64), nn.ReLU(True))
unet_block_mlp = nn.Sequential(nn.Linear(block_embedding_dim, 64), nn.ReLU(True))
vae_fuse_mlp = nn.Linear(256 + 64, lora_rank_vae ** 2)
unet_fuse_mlp = nn.Linear(256 + 64, lora_rank_unet ** 2)
vae_block_embeddings = nn.Embedding(6, block_embedding_dim)
unet_block_embeddings = nn.Embedding(10, block_embedding_dim)

for name, module in [("vae_de_mlp", vae_de_mlp), ("unet_de_mlp", unet_de_mlp),
                      ("vae_block_mlp", vae_block_mlp), ("unet_block_mlp", unet_block_mlp),
                      ("vae_fuse_mlp", vae_fuse_mlp), ("unet_fuse_mlp", unet_fuse_mlp)]:
    _ssd = module.state_dict()
    for k in sd[f"state_dict_{name}"]:
        _ssd[k] = sd[f"state_dict_{name}"][k]
    module.load_state_dict(_ssd)
vae_block_embeddings.load_state_dict(sd["state_embeddings"]["state_dict_vae_block"])
unet_block_embeddings.load_state_dict(sd["state_embeddings"]["state_dict_unet_block"])

for m in [vae_de_mlp, unet_de_mlp, vae_block_mlp, unet_block_mlp, vae_fuse_mlp, unet_fuse_mlp]:
    m.eval()

# DEResNet
de_net = DEResNet(num_in_ch=3, num_degradation=2)
de_net.load_state_dict(torch.load(de_net_path, map_location="cpu"))
de_net.eval()

# Scheduler
sched = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
sched.set_timesteps(1, device="cpu")

print(f"VAE LoRA layers: {len(vae_lora_layers)}")
print(f"UNet LoRA layers: {len(unet_lora_layers)}")
print("Model built successfully!")

# [md cell 15]: ## 5. Compile for Neuron |  | We compile 5 components: | 1. **DEResNet** -- simple CNN (8s) | 2. **Text Encoder** -- CLIP (106s) | 3. **VAE Encoder** -- with de_mod input (179s) | 4. **UNet** -- with de_mod input

# --- cell 16 ---
# Fixed resolution for tracing
LR_H, LR_W = 128, 128
SF = 4
HR_H, HR_W = LR_H * SF, LR_W * SF
PAD_H = (math.ceil(HR_H / 64)) * 64  # 512
PAD_W = (math.ceil(HR_W / 64)) * 64  # 512
LATENT_H, LATENT_W = PAD_H // 8, PAD_W // 8  # 64x64

COMPILE_DIR = os.path.expanduser("~/s3diff_compile")
os.makedirs(COMPILE_DIR, exist_ok=True)
print(f"Resolution: {LR_H}x{LR_W} -> {HR_H}x{HR_W}, latent={LATENT_H}x{LATENT_W}")

# --- cell 17 ---
# 1. DEResNet
de_net_neff = os.path.join(COMPILE_DIR, "de_net.pt")
if os.path.exists(de_net_neff):
    print("[DEResNet] Loading cached...")
    de_net_neuron = torch.jit.load(de_net_neff)
else:
    print("[DEResNet] Compiling...")
    t0 = time.time()
    de_net_neuron = torch_neuronx.trace(de_net, torch.randn(1, 3, LR_H, LR_W),
        compiler_args=['--auto-cast', 'matmult', '-O1'])
    print(f"  Compiled in {time.time()-t0:.1f}s")
    torch.jit.save(de_net_neuron, de_net_neff)

# --- cell 18 ---
# 2. Text Encoder
text_enc_neff = os.path.join(COMPILE_DIR, "text_encoder.pt")
if os.path.exists(text_enc_neff):
    print("[Text Encoder] Loading cached...")
    text_enc_neuron = torch.jit.load(text_enc_neff)
else:
    print("[Text Encoder] Compiling...")
    t0 = time.time()
    text_enc_neuron = torch_neuronx.trace(TextEncoderWrapper(text_encoder),
        torch.zeros(1, 77, dtype=torch.long),
        compiler_args=['--auto-cast', 'matmult', '-O1'])
    print(f"  Compiled in {time.time()-t0:.1f}s")
    torch.jit.save(text_enc_neuron, text_enc_neff)

# --- cell 19 ---
# 3. VAE Encoder (with dynamic LoRA de_mod input)
vae_enc_neff = os.path.join(COMPILE_DIR, "vae_encoder.pt")
if os.path.exists(vae_enc_neff):
    print("[VAE Encoder] Loading cached...")
    vae_enc_neuron = torch.jit.load(vae_enc_neff)
else:
    print("[VAE Encoder] Compiling (with de_mod input)...")
    vae_enc_wrapper = VAEEncoderWrapper(vae, vae_lora_layers, lora_rank_vae)
    example_pixels = torch.randn(1, 3, PAD_H, PAD_W)
    example_de_mod = torch.randn(1, 6, lora_rank_vae, lora_rank_vae)
    t0 = time.time()
    vae_enc_neuron = torch_neuronx.trace(vae_enc_wrapper,
        (example_pixels, example_de_mod),
        compiler_args=['--auto-cast', 'matmult', '-O1'])
    print(f"  Compiled in {time.time()-t0:.1f}s")
    torch.jit.save(vae_enc_neuron, vae_enc_neff)

# --- cell 20 ---
# 4. UNet (with dynamic LoRA de_mod input)
unet_neff = os.path.join(COMPILE_DIR, "unet.pt")
if os.path.exists(unet_neff):
    print("[UNet] Loading cached...")
    unet_neuron = torch.jit.load(unet_neff)
else:
    print("[UNet] Compiling (with de_mod input, ~12 min)...")
    unet_wrapper = UNetWrapper(unet, unet_lora_layers, lora_rank_unet)
    example_latent = torch.randn(1, 4, LATENT_H, LATENT_W)
    example_timestep = torch.tensor([999], dtype=torch.long)
    example_enc = torch.randn(1, 77, 1024)  # SD-Turbo CLIP hidden dim
    example_unet_de_mod = torch.randn(1, 10, lora_rank_unet, lora_rank_unet)
    t0 = time.time()
    unet_neuron = torch_neuronx.trace(unet_wrapper,
        (example_latent, example_timestep, example_enc, example_unet_de_mod),
        compiler_args=['--auto-cast', 'matmult', '-O1'])
    print(f"  Compiled in {time.time()-t0:.1f}s")
    torch.jit.save(unet_neuron, unet_neff)

# --- cell 21 ---
# 5. VAE Decoder (no LoRA)
# NOTE: --auto-cast=matmult causes compiler error 70 on VAE decoder with SDK 2.29
# Use --model-type=unet-inference instead
vae_dec_neff = os.path.join(COMPILE_DIR, "vae_decoder.pt")
if os.path.exists(vae_dec_neff):
    print("[VAE Decoder] Loading cached...")
    vae_dec_neuron = torch.jit.load(vae_dec_neff)
else:
    print("[VAE Decoder] Compiling...")
    t0 = time.time()
    vae_dec_neuron = torch_neuronx.trace(VAEDecoderWrapper(vae),
        torch.randn(1, 4, LATENT_H, LATENT_W),
        compiler_args=['--model-type=unet-inference', '-O1'])
    print(f"  Compiled in {time.time()-t0:.1f}s")
    torch.jit.save(vae_dec_neuron, vae_dec_neff)

print("\nAll 5 components compiled!")

# [md cell 22]: ## 6. Inference Helper Functions

# --- cell 23 ---
def compute_modulation(deg_score):
    """Compute degradation modulation matrices on CPU (tiny MLPs, <1ms)."""
    deg_proj = deg_score[..., None] * W[None, None, :] * 2 * np.pi
    deg_proj = torch.cat([torch.sin(deg_proj), torch.cos(deg_proj)], dim=-1)
    deg_proj = torch.cat([deg_proj[:, 0], deg_proj[:, 1]], dim=-1)

    vae_de_c_embed = vae_de_mlp(deg_proj)
    unet_de_c_embed = unet_de_mlp(deg_proj)

    vae_block_c_embeds = vae_block_mlp(vae_block_embeddings.weight)
    unet_block_c_embeds = unet_block_mlp(unet_block_embeddings.weight)

    B = deg_score.shape[0]
    vae_embeds = vae_fuse_mlp(torch.cat([
        vae_de_c_embed.unsqueeze(1).repeat(1, vae_block_c_embeds.shape[0], 1),
        vae_block_c_embeds.unsqueeze(0).repeat(B, 1, 1)
    ], -1))
    unet_embeds = unet_fuse_mlp(torch.cat([
        unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.shape[0], 1),
        unet_block_c_embeds.unsqueeze(0).repeat(B, 1, 1)
    ], -1))

    return (vae_embeds.reshape(B, 6, lora_rank_vae, lora_rank_vae),
            unet_embeds.reshape(B, 10, lora_rank_unet, lora_rank_unet))


@torch.no_grad()
def run_s3diff_neuron(lr_image,
                      pos_prompt="high quality, highly detailed, clean",
                      neg_prompt="blurry, dotted, noise, raster lines, unclear, lowres, over-smoothed"):
    """Run S3Diff super-resolution on Neuron."""
    to_tensor = transforms.ToTensor()
    im_lr = to_tensor(lr_image).unsqueeze(0)
    
    # Preprocess: resize 4x + normalize + pad
    ori_h, ori_w = im_lr.shape[2:]
    im_lr_resize = F.interpolate(im_lr, size=(ori_h * SF, ori_w * SF), mode='bilinear', align_corners=False)
    im_lr_resize_norm = (im_lr_resize * 2 - 1.0).clamp(-1, 1)
    resize_h, resize_w = im_lr_resize_norm.shape[2:]
    im_lr_resize_norm = F.pad(im_lr_resize_norm, pad=(0, PAD_W - resize_w, 0, PAD_H - resize_h), mode='reflect')

    # 1. DEResNet -> degradation scores
    deg_score = de_net_neuron(im_lr)

    # 2. Compute modulation on CPU
    vae_de_mod_all, unet_de_mod_all = compute_modulation(deg_score)

    # 3. Text encoding
    pos_tokens = tokenizer(pos_prompt, max_length=tokenizer.model_max_length,
                           padding="max_length", truncation=True, return_tensors="pt").input_ids
    neg_tokens = tokenizer(neg_prompt, max_length=tokenizer.model_max_length,
                           padding="max_length", truncation=True, return_tensors="pt").input_ids
    pos_enc = text_enc_neuron(pos_tokens)
    neg_enc = text_enc_neuron(neg_tokens)

    # 4. VAE Encode (with de_mod)
    latent = vae_enc_neuron(im_lr_resize_norm, vae_de_mod_all)

    # 5. UNet x2 for CFG (with de_mod)
    timestep = torch.tensor([999], dtype=torch.long)
    pos_pred = unet_neuron(latent, timestep, pos_enc, unet_de_mod_all)
    neg_pred = unet_neuron(latent, timestep, neg_enc, unet_de_mod_all)
    model_pred = neg_pred + 1.07 * (pos_pred - neg_pred)

    # 6. Scheduler step (CPU)
    x_denoised = sched.step(model_pred.cpu(), torch.tensor([999]), latent.cpu(), return_dict=True).prev_sample

    # 7. VAE Decode
    output = vae_dec_neuron(x_denoised).clamp(-1, 1)
    output = output[:, :, :resize_h, :resize_w]
    return transforms.ToPILImage()((output[0] * 0.5 + 0.5).cpu().clamp(0, 1))

# [md cell 24]: ## 7. Run Inference

# --- cell 25 ---
# Create a synthetic test image
test_img = Image.new('RGB', (LR_H, LR_W), color=(128, 100, 80))
pixels = test_img.load()
for y in range(LR_H):
    for x in range(LR_W):
        r = int(128 + 80 * np.sin(x / 10.0))
        g = int(100 + 60 * np.cos(y / 8.0))
        b = int(80 + 40 * np.sin((x + y) / 12.0))
        pixels[x, y] = (min(255, r), min(255, g), min(255, b))

print(f"Input: {test_img.size}")
test_img

# --- cell 26 ---
# Run inference
t0 = time.time()
output = run_s3diff_neuron(test_img)
print(f"First inference: {time.time()-t0:.2f}s (includes model loading)")
print(f"Output: {output.size}")
output

# [md cell 27]: ## 8. Benchmark

# --- cell 28 ---
# Pre-compute fixed text embeddings (prompts are constant)
pos_prompt = "high quality, highly detailed, clean"
neg_prompt = "blurry, dotted, noise, raster lines, unclear, lowres, over-smoothed"
pos_tokens = tokenizer(pos_prompt, max_length=tokenizer.model_max_length,
                       padding="max_length", truncation=True, return_tensors="pt").input_ids
neg_tokens = tokenizer(neg_prompt, max_length=tokenizer.model_max_length,
                       padding="max_length", truncation=True, return_tensors="pt").input_ids
pos_enc_cached = text_enc_neuron(pos_tokens)
neg_enc_cached = text_enc_neuron(neg_tokens)

# Prepare input
to_tensor = transforms.ToTensor()
im_lr = to_tensor(test_img).unsqueeze(0)
im_lr_resize = F.interpolate(im_lr, size=(LR_H * SF, LR_W * SF), mode='bilinear', align_corners=False)
im_lr_resize_norm = (im_lr_resize * 2 - 1.0).clamp(-1, 1)
resize_h, resize_w = im_lr_resize_norm.shape[2:]
im_lr_resize_norm = F.pad(im_lr_resize_norm, pad=(0, PAD_W - resize_w, 0, PAD_H - resize_h), mode='reflect')
timestep = torch.tensor([999], dtype=torch.long)

N_WARMUP = 5
N_RUNS = 20

def bench_run():
    t0 = time.time()
    deg_score = de_net_neuron(im_lr)
    vae_de_mod_all, unet_de_mod_all = compute_modulation(deg_score)
    latent = vae_enc_neuron(im_lr_resize_norm, vae_de_mod_all)
    pos_pred = unet_neuron(latent, timestep, pos_enc_cached, unet_de_mod_all)
    neg_pred = unet_neuron(latent, timestep, neg_enc_cached, unet_de_mod_all)
    model_pred = neg_pred + 1.07 * (pos_pred - neg_pred)
    x_denoised = sched.step(model_pred.cpu(), torch.tensor([999]), latent.cpu(), return_dict=True).prev_sample
    _ = vae_dec_neuron(x_denoised)
    return time.time() - t0

# Warmup
for i in range(N_WARMUP):
    bench_run()
    print(f"  Warmup {i+1}/{N_WARMUP}")

# Measured runs
times = []
for i in range(N_RUNS):
    t = bench_run()
    times.append(t)

mean_t = np.mean(times)
std_t = np.std(times)
print(f"\n=== S3Diff Benchmark (128x128 -> 512x512) ===")
print(f"Mean latency:  {mean_t:.3f}s +/- {std_t:.3f}s")
print(f"Throughput:    {1.0/mean_t:.2f} img/s")
print(f"Min:           {min(times):.3f}s")
print(f"Max:           {max(times):.3f}s")

# [md cell 29]: ## 9. Results Summary |  | ### Performance |  | | Metric | Value | | |--------|-------| | | Mean latency | ~0.47s | | | Throughput | ~2.1 img/s | | | CPU baseline | 11.5s | | | Speedup | ~24x | | | Resolution | 128x128 -

