"""
Treatment reproducer for kaena-30652: Luo's --tiled-inst-limit flag test.
"""
import os
import sys
import time

import torch
import torch_neuronx
from diffusers import AutoencoderKL

SDXL_PATH = "/home/ubuntu/sdxl-base"
WORKDIR = "/tmp/sdxl_2k_treatment_workdir"

LATENT_SHAPE = (1, 4, 256, 256)


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] Loading SDXL VAE from {SDXL_PATH}", flush=True)
    vae = AutoencoderKL.from_pretrained(SDXL_PATH, subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16)
    vae.eval()

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

    print(f"[{time.strftime('%H:%M:%S')}] Sanity CPU forward...", flush=True)
    with torch.no_grad():
        out = wrapper(example)
    print(f"[{time.strftime('%H:%M:%S')}] CPU output shape: {tuple(out.shape)}", flush=True)
    del out

    # Luo's proposed flag from kaena-30652
    compiler_args = [
        "--model-type=unet-inference",
        "--lnc=2",
        "-O1",
        "--internal-hlo2tensorizer-options=--tiled-inst-limit 10000000",
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
        elapsed = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] *** SUCCESS: compile completed in {elapsed:.1f}s ***", flush=True)
        out_path = os.path.join(WORKDIR, "vae_decoder.pt")
        torch.jit.save(traced, out_path)
        print(f"[{time.strftime('%H:%M:%S')}] Saved traced model to {out_path}", flush=True)
        # Print neff size if exists
        for root, dirs, files in os.walk(WORKDIR):
            for f in files:
                if f.endswith(".neff"):
                    p = os.path.join(root, f)
                    print(f"NEFF: {p} size_mb={os.path.getsize(p)/1024/1024:.2f}", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Compile FAILED after {time.time()-t0:.1f}s", flush=True)
        print(f"Exception class: {type(e).__name__}", flush=True)
        print(f"Exception message: {e}", flush=True)
        sys.exit(70)


if __name__ == "__main__":
    main()
