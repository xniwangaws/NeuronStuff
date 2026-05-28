"""Sanity check: run our wrapper chain on CPU eagerly (no Neuron) and compare to vae.decode.
Detects whether a wrapper signature mismatch is the source of the cosine drop."""
import torch
from diffusers import AutoencoderKL
import sys
sys.path.insert(0, "/home/ubuntu/work_a")
from trace_subneffs import (
    PostQuantConvWrap, ConvInWrap, ResnetBlockWrap, AttentionWrap,
    UpBlockWrap, ConvOutWrap,
)

VAE_PATH = "/home/ubuntu/sdxl-base"

vae = AutoencoderKL.from_pretrained(
    VAE_PATH, subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16
)
vae.eval()

torch.manual_seed(0)
latent = torch.randn(1, 4, 256, 256, dtype=torch.bfloat16)

with torch.no_grad():
    ref = vae.decode(latent).sample

# Build wrapper chain (eager).
wrappers = [
    PostQuantConvWrap(vae),
    ConvInWrap(vae),
    ResnetBlockWrap(vae.decoder.mid_block.resnets[0]),
    AttentionWrap(vae.decoder.mid_block.attentions[0]),
    ResnetBlockWrap(vae.decoder.mid_block.resnets[1]),
    UpBlockWrap(vae, 0),
    UpBlockWrap(vae, 1),
    UpBlockWrap(vae, 2),
    UpBlockWrap(vae, 3),
    ConvOutWrap(vae),
]
x = latent
for i, w in enumerate(wrappers):
    w.eval()
    with torch.no_grad():
        x = w(x)
    if isinstance(x, (tuple, list)):
        x = x[0]
    print(f"  step {i:2d} {type(w).__name__:25s} -> {list(x.shape)}")

cos = float(torch.nn.functional.cosine_similarity(
    ref.flatten().float(), x.flatten().float(), dim=0))
print(f"eager-chain vs vae.decode cosine: {cos:.6f}")
print(f"  max_abs_diff = {(ref.float() - x.float()).abs().max().item():.6f}")
print(f"  mean_abs_diff = {(ref.float() - x.float()).abs().mean().item():.6f}")
