"""Pixel-level diff: Neuron klein vs H100 BF16 reference.

Computes MSE / PSNR / SSIM per seed, aggregates mean + std.
Does NOT require CUDA / Neuron — runs locally.
"""
import json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/Users/xniwang/NeuronStuff/flux2/task015_klein_jim_pr146/results")
SEEDS = list(range(42, 52))

def load(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0

def mse(a, b):
    return float(np.mean((a - b) ** 2))

def psnr(a, b):
    m = mse(a, b)
    return float(20 * np.log10(1.0 / np.sqrt(m))) if m > 0 else float("inf")

def ssim_naive(a, b, win=11):
    """Simple SSIM (no Gaussian window; rough indicator)."""
    from scipy.ndimage import uniform_filter
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    def uf(x): return np.stack([uniform_filter(x[..., c], size=win) for c in range(3)], -1)
    mu_a, mu_b = uf(a), uf(b)
    va = uf(a * a) - mu_a * mu_a
    vb = uf(b * b) - mu_b * mu_b
    cab = uf(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))
    return float(np.mean(s))

def compare(ref_dir, cmp_dir, label):
    msesm, psnrsm, ssims = [], [], []
    for seed in SEEDS:
        ref = ROOT / ref_dir / f"seed{seed}_cat.png"
        cmp = ROOT / cmp_dir / f"seed{seed}_cat.png"
        if not (ref.exists() and cmp.exists()):
            print(f"  skip seed {seed}: missing file")
            continue
        a, b = load(ref), load(cmp)
        if a.shape != b.shape:
            print(f"  skip seed {seed}: shape {a.shape} vs {b.shape}")
            continue
        m, p, s = mse(a, b), psnr(a, b), ssim_naive(a, b)
        msesm.append(m); psnrsm.append(p); ssims.append(s)
        print(f"  seed {seed}: MSE={m:.5f}  PSNR={p:.2f}dB  SSIM={s:.4f}")
    def ms(x): return f"{np.mean(x):.4f}±{np.std(x):.4f}" if x else "n/a"
    print(f"[{label}] mean: MSE={ms(msesm)} PSNR={ms(psnrsm)}dB SSIM={ms(ssims)}  n={len(msesm)}")
    return {"label": label, "n": len(msesm),
            "mse_mean": float(np.mean(msesm)) if msesm else None,
            "psnr_mean": float(np.mean(psnrsm)) if psnrsm else None,
            "ssim_mean": float(np.mean(ssims)) if ssims else None}

def main():
    print("=== klein 1K BF16 vs H100 BF16 (reference) ===")
    r1 = compare("h100_1024_bf16", "klein_1k_50step", "Neuron vs H100 BF16")
    print()
    print("=== H100 FP8 vs H100 BF16 (reference) — intra-GPU sanity ===")
    r2 = compare("h100_1024_bf16", "h100_1024_fp8", "H100 FP8 vs H100 BF16")
    print()
    print("=== klein 2K BF16 vs H100 BF16 (2K reference) ===")
    r3 = compare("h100_2048_bf16", "klein_2k_50step", "Neuron 2K vs H100 2K BF16")

    out = {"1k_neuron_vs_h100bf16": r1, "1k_h100fp8_vs_h100bf16": r2, "2k_neuron_vs_h100bf16": r3}
    with open(ROOT / "pixel_diff.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {ROOT/'pixel_diff.json'}")

if __name__ == "__main__":
    main()
