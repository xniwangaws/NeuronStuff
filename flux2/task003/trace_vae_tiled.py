"""Trace FLUX.2 VAE decoder on Neuron with explicit tiling at 512² tile size.

Strategy: set vae.tile_sample_min_size=512 and vae.tile_latent_min_size=64 so
that tiling is activated for output >= 512². Trace only a single-tile-size
forward (512² tile). At inference, reuse the traced NEFF repeatedly for each
tile via the diffusers' built-in tiling logic.

This avoids the NCC_IXTP002 "too many instructions" error we saw at single-
pass 1024² compilation.

Usage: python trace_vae_tiled.py [--tile 512]
"""

import argparse
import time
from pathlib import Path

import torch
import torch_neuronx
from diffusers import AutoencoderKLFlux2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", type=int, default=512, help="Pixel tile size (divide by 8 for latent tile)")
    parser.add_argument("--model", default="/home/ubuntu/flux2_weights")
    parser.add_argument("--out-dir", default="vae_traced_tiled")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] AutoencoderKLFlux2 from {args.model}")
    vae = AutoencoderKLFlux2.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.eval()

    latent_tile = args.tile // 8
    print(f"[config] tile={args.tile}px, latent_tile={latent_tile}")

    # A single-tile forward. This mimics what diffusers' tiled_decode does internally,
    # but we trace only the inner decoder call (skip the blend logic, which is done on CPU).
    class SingleTileDecoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent_tile):
            return self.vae.decode(latent_tile, return_dict=False)[0]

    wrap = SingleTileDecoder(vae).eval()
    example_latent = torch.randn(1, 32, latent_tile, latent_tile, dtype=torch.bfloat16)

    print(f"[trace] compiling single-tile decoder at {args.tile}²...")
    t0 = time.perf_counter()
    compiled = torch_neuronx.trace(
        wrap,
        example_inputs=example_latent,
        compiler_args=[
            "--auto-cast", "matmult",
            "--model-type", "transformer",
        ],
        inline_weights_to_neff=True,
    )
    dt = time.perf_counter() - t0
    print(f"[trace] compiled in {dt:.1f}s")

    path = out / f"vae_decoder_tile{args.tile}.pt"
    torch.jit.save(compiled, path)
    print(f"[save] {path} ({path.stat().st_size / 1e9:.2f} GB)")

    # Benchmark: single tile
    for _ in range(3):
        _ = compiled(example_latent)
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        _ = compiled(example_latent)
        ts.append(time.perf_counter() - t)
    print(f"[bench] single-tile: mean={sum(ts)/len(ts)*1000:.1f}ms per tile")

    # Simulate full-image decode: 1024² = 4 tiles, 2048² = 16 tiles
    for out_res in (1024, 2048):
        n_tiles = (out_res // args.tile) ** 2
        total = n_tiles * (sum(ts) / len(ts))
        print(f"[estimate] {out_res}² output = {n_tiles} tiles × {sum(ts)/len(ts)*1000:.1f}ms = {total*1000:.0f}ms decode-only")


if __name__ == "__main__":
    main()
