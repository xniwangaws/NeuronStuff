"""Run each Neuron sub-NEFF and compare its output to the eager wrapper output,
using the SAME inputs (the eager-chain output of the previous stage). Tells us
which sub-NEFF is responsible for accuracy drop."""
import torch
import sys
from pathlib import Path
sys.path.insert(0, "/home/ubuntu/work_a")
from diffusers import AutoencoderKL
from trace_subneffs import (
    PostQuantConvWrap, ConvInWrap, ResnetBlockWrap, AttentionWrap,
    UpBlockWrap, ConvOutWrap,
)

VAE_PATH = "/home/ubuntu/sdxl-base"
WORK = Path("/home/ubuntu/work_a")

vae = AutoencoderKL.from_pretrained(
    VAE_PATH, subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16
)
vae.eval()

torch.manual_seed(0)
latent = torch.randn(1, 4, 256, 256, dtype=torch.bfloat16)

stages = [
    ("01_post_quant_conv", PostQuantConvWrap(vae)),
    ("02_conv_in",         ConvInWrap(vae)),
    ("03a_mid_resnet0",    ResnetBlockWrap(vae.decoder.mid_block.resnets[0])),
    ("03b_mid_attn",       AttentionWrap(vae.decoder.mid_block.attentions[0])),
    ("03c_mid_resnet1",    ResnetBlockWrap(vae.decoder.mid_block.resnets[1])),
    ("04_up_block_0",      UpBlockWrap(vae, 0)),
    ("05_up_block_1",      UpBlockWrap(vae, 1)),
    ("06_up_block_2",      UpBlockWrap(vae, 2)),
    ("07_up_block_3",      UpBlockWrap(vae, 3)),
    ("08_conv_out_block",  ConvOutWrap(vae)),
]

def cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0))

x_eager = latent
for name, w in stages:
    w.eval()
    pt = WORK / f"traced_{name}.pt"
    neff = torch.jit.load(str(pt))

    with torch.no_grad():
        eager_out = w(x_eager)
    if isinstance(eager_out, (tuple, list)):
        eager_out = eager_out[0]

    with torch.no_grad():
        neff_out = neff(x_eager)
    if isinstance(neff_out, (tuple, list)):
        neff_out = neff_out[0]

    cos = cosine(eager_out, neff_out)
    md  = float((eager_out.float() - neff_out.float()).abs().max())
    eager_max = float(eager_out.float().abs().max())
    rel = md / max(eager_max, 1e-6)
    print(f"  {name:24s} on eager-input: cos={cos:.6f}  max_abs={md:.4f}  rel_to_eager_max={rel:.4f}  eager_max={eager_max:.2f}  shape={list(neff_out.shape)}")

    x_eager = eager_out  # advance with eager (clean) so next stage's input is also clean
