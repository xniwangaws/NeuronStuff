"""End-to-end validation for the chained per-block SDXL VAE decoder at 2048x2048.

Compares the chained Neuron version against CPU bf16 reference (vae.decode(latent)),
prints cosine sim + max abs diff, and times cold + warm runs. Saves a decoded PNG
of a deterministic latent so we can eyeball quality.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL

from chained_vae_decoder import ChainedVAEDecoder, DEFAULT_ORDER

VAE_PATH = "/home/ubuntu/sdxl-base"
WORK = Path("/home/ubuntu/work_a")


def cosine(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def to_png(image_bf16, path):
    """image_bf16 in [-1, 1], shape [1,3,H,W]. Save uint8 PNG."""
    from PIL import Image
    x = image_bf16.detach().to(torch.float32).clamp(-1, 1)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray(x).save(path)


def main():
    print("Loading CPU reference VAE in bf16...")
    vae = AutoencoderKL.from_pretrained(
        VAE_PATH, subfolder="vae", variant="fp16", torch_dtype=torch.bfloat16
    )
    vae.eval()

    torch.manual_seed(0)
    latent = torch.randn(1, 4, 256, 256, dtype=torch.bfloat16)

    print("Running CPU bf16 reference (vae.decode)...")
    t0 = time.time()
    with torch.no_grad():
        ref = vae.decode(latent).sample
    cpu_t = time.time() - t0
    print(f"  CPU forward: {cpu_t:.1f}s  out_shape={list(ref.shape)} dtype={ref.dtype}")

    print("Loading chained Neuron decoder...")
    chain = ChainedVAEDecoder(WORK, order=DEFAULT_ORDER)
    print(f"  loaded {len(chain.names)} sub-NEFFs")

    print("Cold run (chained Neuron)...")
    t0 = time.time()
    with torch.no_grad():
        out_cold = chain(latent)
    cold_t = time.time() - t0
    print(f"  cold: {cold_t:.2f}s  out_shape={list(out_cold.shape)} dtype={out_cold.dtype}")

    print("Warm runs (5x)...")
    warm_times = []
    with torch.no_grad():
        for i in range(5):
            t0 = time.time()
            out = chain(latent)
            dt = time.time() - t0
            warm_times.append(dt)
            print(f"  warm[{i}]: {dt:.3f}s")

    cos = cosine(ref, out)
    max_diff = float((ref.to(torch.float32) - out.to(torch.float32)).abs().max())
    mean_diff = float((ref.to(torch.float32) - out.to(torch.float32)).abs().mean())
    print(f"\n=== Numerical accuracy ===")
    print(f"  cosine_similarity = {cos:.6f}  (target >= 0.999)")
    print(f"  max_abs_diff      = {max_diff:.6f}")
    print(f"  mean_abs_diff     = {mean_diff:.6f}")

    # Save PNGs.
    to_png(ref, WORK / "ref_cpu_bf16.png")
    to_png(out, WORK / "neuron_chained.png")
    print(f"  saved ref/neuron PNGs to {WORK}")

    summary = {
        "input_shape": list(latent.shape),
        "output_shape": list(out.shape),
        "cosine_similarity": cos,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "cpu_ref_seconds": cpu_t,
        "neuron_cold_seconds": cold_t,
        "neuron_warm_seconds": warm_times,
        "neuron_warm_mean_seconds": float(np.mean(warm_times)),
        "neuron_warm_min_seconds": float(np.min(warm_times)),
        "subnefs": chain.names,
    }
    (WORK / "test_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {WORK / 'test_summary.json'}")


if __name__ == "__main__":
    main()
