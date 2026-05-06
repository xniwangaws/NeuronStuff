"""Diagnose where NaN appears in Jim's pipeline at LR=128 (smaller, faster)."""
import os, sys, math, time
sys.path.insert(0, os.path.expanduser("~/s3diff_jim"))
os.environ["LR"] = "128"

import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig
from huggingface_hub import hf_hub_download, snapshot_download
import torch_neuronx

LR_H = LR_W = 128
SF = 4
HR_H, HR_W = LR_H * SF, LR_W * SF
PAD_H = (math.ceil(HR_H / 64)) * 64
PAD_W = (math.ceil(HR_W / 64)) * 64
LATENT_H, LATENT_W = PAD_H // 8, PAD_W // 8
COMPILE_DIR = os.path.expanduser("~/s3diff_jim/compile_128")

print(f"LR={LR_H}, PAD={PAD_H}, LATENT={LATENT_H}")

pretrained_path = hf_hub_download(repo_id="zhangap/S3Diff", filename="s3diff.pkl")
sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")

tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
sched = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
sched.set_timesteps(1, device="cpu")

sd = torch.load(pretrained_path, map_location="cpu", weights_only=False)
lora_rank_unet = sd["rank_unet"]
lora_rank_vae = sd["rank_vae"]
num_embeddings = 64
W = torch.nn.Parameter(sd["w"], requires_grad=False)

import torch.nn as nn
vae_de_mlp = nn.Sequential(nn.Linear(num_embeddings * 4, 256), nn.ReLU(True))
unet_de_mlp = nn.Sequential(nn.Linear(num_embeddings * 4, 256), nn.ReLU(True))
vae_block_mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU(True))
unet_block_mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU(True))
vae_fuse_mlp = nn.Linear(256 + 64, lora_rank_vae ** 2)
unet_fuse_mlp = nn.Linear(256 + 64, lora_rank_unet ** 2)
vae_block_embeddings = nn.Embedding(6, 64)
unet_block_embeddings = nn.Embedding(10, 64)

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

def compute_modulation(deg_score):
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


de_net_neuron = torch.jit.load(os.path.join(COMPILE_DIR, "de_net.pt"))
text_enc_neuron = torch.jit.load(os.path.join(COMPILE_DIR, "text_encoder.pt"))
vae_enc_neuron = torch.jit.load(os.path.join(COMPILE_DIR, "vae_encoder.pt"))
unet_neuron = torch.jit.load(os.path.join(COMPILE_DIR, "unet.pt"))
vae_dec_neuron = torch.jit.load(os.path.join(COMPILE_DIR, "vae_decoder.pt"))


def show(name, t):
    t = t.float()
    print(f"  {name:20s} shape={tuple(t.shape)} dtype={t.dtype} "
          f"min={t.min().item():.4f} max={t.max().item():.4f} mean={t.mean().item():.4f} "
          f"nan={torch.isnan(t).any().item()} inf={torch.isinf(t).any().item()}")


# Try with real cat LQ image downsampled to 128x128
cat_path = os.path.expanduser("~/s3diff_jim/lq/cat_LQ_256.png")
im = Image.open(cat_path).convert("RGB").resize((128, 128), Image.BICUBIC)
to_tensor = transforms.ToTensor()
im_lr = to_tensor(im).unsqueeze(0)
show("im_lr", im_lr)

im_lr_resize = F.interpolate(im_lr, size=(128 * SF, 128 * SF), mode='bilinear', align_corners=False)
im_lr_resize_norm = (im_lr_resize * 2 - 1.0).clamp(-1, 1)
resize_h, resize_w = im_lr_resize_norm.shape[2:]
im_lr_resize_norm = F.pad(im_lr_resize_norm, pad=(0, PAD_W - resize_w, 0, PAD_H - resize_h), mode='reflect')
show("im_lr_resize_norm", im_lr_resize_norm)

with torch.no_grad():
    deg_score = de_net_neuron(im_lr)
    show("deg_score", deg_score)
    vae_de_mod_all, unet_de_mod_all = compute_modulation(deg_score)
    show("vae_de_mod_all", vae_de_mod_all)
    show("unet_de_mod_all", unet_de_mod_all)
    pos_tokens = tokenizer("high quality, highly detailed, clean", max_length=tokenizer.model_max_length,
                           padding="max_length", truncation=True, return_tensors="pt").input_ids
    neg_tokens = tokenizer("blurry, dotted, noise, raster lines, unclear, lowres, over-smoothed",
                           max_length=tokenizer.model_max_length,
                           padding="max_length", truncation=True, return_tensors="pt").input_ids
    pos_enc = text_enc_neuron(pos_tokens)
    neg_enc = text_enc_neuron(neg_tokens)
    show("pos_enc", pos_enc)
    show("neg_enc", neg_enc)
    latent = vae_enc_neuron(im_lr_resize_norm, vae_de_mod_all)
    show("latent (after VAE enc)", latent)
    timestep = torch.tensor([999], dtype=torch.long)
    pos_pred = unet_neuron(latent, timestep, pos_enc, unet_de_mod_all)
    neg_pred = unet_neuron(latent, timestep, neg_enc, unet_de_mod_all)
    show("pos_pred", pos_pred)
    show("neg_pred", neg_pred)
    model_pred = neg_pred + 1.07 * (pos_pred - neg_pred)
    show("model_pred", model_pred)
    x_denoised = sched.step(model_pred.cpu(), torch.tensor([999]), latent.cpu(), return_dict=True).prev_sample
    show("x_denoised", x_denoised)
    output = vae_dec_neuron(x_denoised)
    show("output (VAE dec raw)", output)
    output_clamp = output.clamp(-1, 1)
    show("output clamped", output_clamp)
    out_final = (output_clamp[0] * 0.5 + 0.5).cpu().clamp(0, 1)
    show("out_final", out_final)
