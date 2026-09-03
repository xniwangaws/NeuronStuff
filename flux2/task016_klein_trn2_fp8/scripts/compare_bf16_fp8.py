#!/usr/bin/env python3
"""Compare same-seed BF16 and FP8 FLUX.2 benchmark images."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-dir", type=Path, required=True)
    parser.add_argument("--fp8-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def global_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Dependency-free global SSIM indicator over all RGB pixels."""
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = float(reference.mean())
    mu_y = float(candidate.mean())
    var_x = float(reference.var())
    var_y = float(candidate.var())
    covariance = float(
        ((reference - mu_x) * (candidate - mu_y)).mean()
    )
    numerator = (2 * mu_x * mu_y + c1) * (2 * covariance + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return numerator / denominator


def main() -> None:
    args = parse_args()
    bf16_images = {
        path.name: path for path in sorted(args.bf16_dir.glob("seed*_cat.png"))
    }
    fp8_images = {
        path.name: path for path in sorted(args.fp8_dir.glob("seed*_cat.png"))
    }

    rows = []
    for name in sorted(bf16_images.keys() & fp8_images.keys()):
        reference = load_rgb(bf16_images[name])
        candidate = load_rgb(fp8_images[name])
        if reference.shape != candidate.shape:
            raise ValueError(
                f"Shape mismatch for {name}: {reference.shape} vs {candidate.shape}"
            )
        error = reference - candidate
        mse = float(np.mean(error**2))
        psnr = math.inf if mse == 0 else float(10 * math.log10(1.0 / mse))
        rows.append(
            {
                "image": name,
                "mse": mse,
                "mae": float(np.mean(np.abs(error))),
                "psnr_db": psnr,
                "global_ssim": global_ssim(reference, candidate),
            }
        )

    if not rows:
        raise SystemExit("No matching seed*_cat.png files found.")

    finite_psnr = [row["psnr_db"] for row in rows if math.isfinite(row["psnr_db"])]
    summary = {
        "bf16_dir": str(args.bf16_dir),
        "fp8_dir": str(args.fp8_dir),
        "sample_count": len(rows),
        "mean_mse": float(np.mean([row["mse"] for row in rows])),
        "mean_mae": float(np.mean([row["mae"] for row in rows])),
        "mean_psnr_db": (
            float(np.mean(finite_psnr)) if finite_psnr else math.inf
        ),
        "mean_global_ssim": float(
            np.mean([row["global_ssim"] for row in rows])
        ),
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
