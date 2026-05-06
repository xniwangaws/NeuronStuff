"""Test the LR=256 VAE encoder, which was compiled with --model-type=unet-inference."""
import os, torch, torch_neuronx
import numpy as np, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from diffusers import AutoencoderKL
from peft import LoraConfig

COMPILE_DIR = os.path.expanduser("~/s3diff_jim/compile_256")
LR = 256
PAD = 1024

# Load compute_modulation weights
pretrained_path = hf_hub_download(repo_id="zhangap/S3Diff", filename="s3diff.pkl")
sd = torch.load(pretrained_path, map_location="cpu", weights_only=False)
lora_rank_vae = sd["rank_vae"]
lora_rank_unet = sd["rank_unet"]
num_embeddings = 64
W = nn.Parameter(sd["w"], requires_grad=False)
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
vae_enc_neuron = torch.jit.load(os.path.join(COMPILE_DIR, "vae_encoder.pt"))

def show(name, t):
    t = t.float()
    print(f"  {name:25s} shape={tuple(t.shape)} min={t.min().item():.4f} max={t.max().item():.4f} mean={t.mean().item():.4f} nan={torch.isnan(t).any().item()}")

cat_path = os.path.expanduser("~/s3diff_jim/lq/cat_LQ_256.png")
im = Image.open(cat_path).convert("RGB")
to_tensor = transforms.ToTensor()
im_lr = to_tensor(im).unsqueeze(0)
im_lr_resize = F.interpolate(im_lr, size=(LR*4, LR*4), mode='bilinear', align_corners=False)
im_lr_resize_norm = (im_lr_resize * 2 - 1.0).clamp(-1, 1)
rh, rw = im_lr_resize_norm.shape[2:]
im_lr_resize_norm = F.pad(im_lr_resize_norm, pad=(0, PAD-rw, 0, PAD-rh), mode='reflect')
with torch.no_grad():
    deg_score = de_net_neuron(im_lr)
    show("deg_score", deg_score)
    v_mod, _ = compute_modulation(deg_score)
    show("v_mod", v_mod)
    latent = vae_enc_neuron(im_lr_resize_norm, v_mod)
    show("latent (LR=256 VAE enc)", latent)
