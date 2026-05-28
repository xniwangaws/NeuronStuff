"""
Minimal reproducer for SDXL VAE decoder NEFF compile failure at 2K
(input latent 256x256 -> output 2048x2048).

Triggers NCC_EVRF007 even with --inst-count-limit=15000000 set.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python repro_sdxl_2k_vae.py 2>&1 | tee error.log

Tested on:
- trn2.3xlarge (1 chip, LNC=2)
- DLAMI: Deep Learning AMI Neuron Multi-Framework Ubuntu 24.04
- Neuron SDK 2.32 (torch-neuronx 2.9.0.2.14.27725)
- diffusers 0.34.0
"""
import os
import sys
import time

import torch
import torch_neuronx
from diffusers import AutoencoderKL

SDXL_PATH = "/home/ubuntu/sdxl-base"
WORKDIR = "/tmp/sdxl_2k_ticket_workdir"

# 2048x2048 image -> latent (B, 4, 256, 256) for VAE decoder
LATENT_SHAPE = (1, 4, 256, 256)


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] Loading SDXL VAE from {SDXL_PATH}", flush=True)
    vae = AutoencoderKL.from_pretrained(SDXL_PATH, subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16)
    vae.eval()

    # Standalone VAE decoder + post_quant_conv (matches diffusers' vae.decode path).
    # post_quant_conv is the projection from latent to decoder input.
    decoder = vae.decoder
    post_quant_conv = vae.post_quant_conv

    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, post_quant_conv, decoder):
            super().__init__()
            self.post_quant_conv = post_quant_conv
            self.decoder = decoder

        def forward(self, latent):
            z = self.post_quant_conv(latent)
            return self.decoder(z)

    wrapper = VAEDecoderWrapper(post_quant_conv, decoder).eval()

    example = torch.randn(LATENT_SHAPE, dtype=torch.bfloat16)
    print(f"[{time.strftime('%H:%M:%S')}] Input latent shape: {tuple(example.shape)} bf16 -> output expected {LATENT_SHAPE[2]*8}x{LATENT_SHAPE[3]*8}", flush=True)

    # Sanity-check forward on CPU before tracing
    print(f"[{time.strftime('%H:%M:%S')}] Sanity CPU forward...", flush=True)
    with torch.no_grad():
        out = wrapper(example)
    print(f"[{time.strftime('%H:%M:%S')}] CPU output shape: {tuple(out.shape)}", flush=True)
    del out

    # Compile flags include --inst-count-limit=15M which is the workaround
    # that lifted the 5M ceiling for SDXL UNet 1K but NOT for VAE decoder 2K.
    compiler_args = [
        "--model-type=unet-inference",
        "--lnc=2",
        "-O1",
        "--tensorizer-options=--inst-count-limit=15000000",
        "--internal-max-instruction-limit=15000000",
        "--verbose=DEBUG",
    ]

    print(f"[{time.strftime('%H:%M:%S')}] Tracing VAE decoder with compiler_args={compiler_args}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] compiler_workdir={WORKDIR}", flush=True)
    t0 = time.time()
    try:
        traced = torch_neuronx.trace(
            wrapper,
            example,
            compiler_workdir=WORKDIR,
            compiler_args=compiler_args,
        )
        print(f"[{time.strftime('%H:%M:%S')}] *** SURPRISE: compile succeeded in {time.time()-t0:.1f}s ***", flush=True)
        torch.jit.save(traced, os.path.join(WORKDIR, "vae_decoder.pt"))
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Compile FAILED after {time.time()-t0:.1f}s", flush=True)
        print(f"Exception class: {type(e).__name__}", flush=True)
        print(f"Exception message: {e}", flush=True)
        sys.exit(70)


if __name__ == "__main__":
    main()
