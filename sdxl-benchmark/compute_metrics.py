"""
Informational PSNR (+ optional LPIPS) between trn2/l4 and h100 outputs.

NOT a pass/fail gate — customer principle #2 is "no pixel-level alignment".
Report as context only in results/README.md.
"""

import argparse
import csv
import math
import os

import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--n_prompts", type=int, default=5)
    p.add_argument("--lpips", action="store_true", help="also compute LPIPS (needs lpips pkg)")
    return p.parse_args()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10((255.0 ** 2) / mse)


def maybe_lpips():
    try:
        import lpips  # type: ignore
        import torch

        net = lpips.LPIPS(net="alex")
        net.eval()

        def to_t(arr: np.ndarray) -> "torch.Tensor":
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            return t

        def fn(a, b):
            with torch.no_grad():
                return float(net(to_t(a), to_t(b)).item())

        return fn
    except Exception as e:
        print(f"[lpips] disabled: {e}")
        return None


def load(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    args = parse_args()
    lpips_fn = maybe_lpips() if args.lpips else None
    out = os.path.join(args.root, "metrics.csv")
    rows = [["pair", "prompt_id", "psnr_db", "lpips"]]

    for k in range(args.n_prompts):
        h100 = os.path.join(args.root, "images", "h100", f"prompt{k}_steps{args.steps}.png")
        if not os.path.exists(h100):
            print(f"[skip] no H100 for prompt{k}")
            continue
        a = load(h100)
        for pair, dev in [("trn2_vs_h100", "neuron"), ("l4_vs_h100", "l4")]:
            other = os.path.join(args.root, "images", dev, f"prompt{k}_steps{args.steps}.png")
            if not os.path.exists(other):
                print(f"[skip] no {dev} for prompt{k}")
                continue
            b = load(other)
            if b.shape != a.shape:
                print(f"[skip] shape mismatch {pair} prompt{k}")
                continue
            p = psnr(a, b)
            lp = f"{lpips_fn(a, b):.4f}" if lpips_fn else ""
            rows.append([pair, k, f"{p:.2f}", lp])
            print(f"{pair} prompt{k}: PSNR={p:.2f} dB LPIPS={lp}")

    with open(out, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
