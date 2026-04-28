"""Trace the FLUX.2 VAE decoder on Neuron via torch_neuronx.trace.

FLUX.2-dev VAE: `AutoencoderKLFlux2`, 8× spatial compression, 32 latent channels.
We trace the decoder only (encoder not needed at inference for text-to-image).

Input at 1024² image → latent (1, 32, 128, 128) (after /8 compression).
Input at 2048² image → latent (1, 32, 256, 256).

Usage: python trace_vae.py --resolution {1024,2048}
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch_neuronx
from diffusers import AutoencoderKLFlux2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, required=True, choices=[256, 512, 1024, 2048])
    parser.add_argument("--model", default="/home/ubuntu/flux2_weights")
    parser.add_argument("--out-dir", default="vae_traced")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] AutoencoderKLFlux2 from {args.model}")
    vae = AutoencoderKLFlux2.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.eval()

    latent_h = args.resolution // 8
    latent_w = args.resolution // 8
    latent_channels = vae.config.latent_channels if hasattr(vae.config, "latent_channels") else 32

    print(f"[probe] latent shape: (1, {latent_channels}, {latent_h}, {latent_w}); VAE config: "
          f"block_out_channels={vae.config.block_out_channels if hasattr(vae.config, 'block_out_channels') else '?'}")

    # Dummy latent
    example_latent = torch.randn(1, latent_channels, latent_h, latent_w, dtype=torch.bfloat16)

    # CPU sanity first
    print(f"[cpu] running decoder on CPU for sanity...")
    t = time.perf_counter()
    with torch.no_grad():
        cpu_out = vae.decode(example_latent, return_dict=False)[0]
    print(f"[cpu] out shape={tuple(cpu_out.shape)} dtype={cpu_out.dtype} in {time.perf_counter()-t:.1f}s")

    # Wrap decoder in a module that takes a plain tensor and returns a plain tensor
    class VAEDecoderWrap(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent):
            return self.vae.decode(latent, return_dict=False)[0]

    wrap = VAEDecoderWrap(vae).eval()

    print(f"[trace] torch_neuronx.trace(...) for {args.resolution}²")
    t0 = time.perf_counter()
    compiled = torch_neuronx.trace(
        wrap,
        example_inputs=example_latent,
        compiler_args=[
            "--auto-cast", "matmult",  # included for consistency though bf16 is no-op
            "--model-type", "transformer",
        ],
        inline_weights_to_neff=True,
    )
    compile_time = time.perf_counter() - t0
    print(f"[trace] compiled in {compile_time:.1f}s")

    # Save the compiled model
    path = out / f"vae_decoder_{args.resolution}.pt"
    torch.jit.save(compiled, path)
    print(f"[save] {path} ({path.stat().st_size/1e9:.2f} GB)")

    # Run on Neuron and compare against CPU
    print(f"[neuron] running 3 warmup + 5 measured iterations...")
    for _ in range(3):
        _ = compiled(example_latent)

    ts = []
    for _ in range(5):
        t = time.perf_counter()
        neuron_out = compiled(example_latent)
        dt = time.perf_counter() - t
        ts.append(dt)
    print(f"[neuron] mean={sum(ts)/len(ts)*1000:.1f}ms  ts={[f'{x*1000:.1f}' for x in ts]}")

    # Accuracy check
    rel_err = (cpu_out - neuron_out).abs().max().item() / cpu_out.abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        cpu_out.float().flatten().unsqueeze(0),
        neuron_out.float().flatten().unsqueeze(0),
    ).item()
    print(f"[acc] max_rel_err={rel_err:.4e}  cos_sim={cos_sim:.6f}")


if __name__ == "__main__":
    main()
