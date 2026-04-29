#!/usr/bin/env python3
"""Accuracy metrics for the FLUX.2-dev benchmark matrix (1024² only).

Reference: H100 BF16 (highest-quality GPU baseline).
For each other device/precision, per-prompt PSNR and pixel-L1 are computed,
plus the mean across the 10 prompts. Results written to results/metrics.json.

LPIPS was considered but is not used here: the `lpips` package is not
installed system-wide on this laptop and requires torch + a VGG weight
download, which this laptop-only task explicitly avoids. Pixel-L1 (on 0-1
normalized RGB) and PSNR together give a reasonable proxy for perceptual
similarity between images sampled from the same seeds.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

TASK_ROOT = Path("/Users/xniwang/oppo-opencode/working/flux2")
OUT_DIR = TASK_ROOT / "task011" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE = ("H100 BF16", TASK_ROOT / "task002/results/h100_bf16_1024")
CANDIDATES = [
    ("L4 NF4",      TASK_ROOT / "task001/results/l4_nf4_1024"),
    ("H100 FP8",    TASK_ROOT / "task002/results/h100_fp8_1024"),
    ("Neuron BF16", TASK_ROOT / "task010/results/neuron_bf16_1024"),
]
NUM_PROMPTS = 10


def prompt_filename(p: int) -> str:
    return f"seed{42 + p:04d}_p{p:02d}.png"


def load_rgb01(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.float32) / 255.0


def psnr(ref: np.ndarray, cand: np.ndarray) -> float:
    mse = float(np.mean((ref - cand) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def l1(ref: np.ndarray, cand: np.ndarray) -> float:
    return float(np.mean(np.abs(ref - cand)))


def evaluate(label: str, cand_dir: Path, ref_dir: Path) -> dict | None:
    if not (cand_dir / prompt_filename(0)).exists():
        print(f"skip {label}: {cand_dir} (no images present)")
        return None
    per_prompt = []
    for p in range(NUM_PROMPTS):
        ref = load_rgb01(ref_dir / prompt_filename(p))
        cand = load_rgb01(cand_dir / prompt_filename(p))
        if cand.shape != ref.shape:
            # L4 NF4 and H100 1024 are both 1024²; but guard anyway.
            cand_im = Image.fromarray((cand * 255).astype(np.uint8)).resize(
                (ref.shape[1], ref.shape[0]), Image.LANCZOS)
            cand = np.asarray(cand_im, dtype=np.float32) / 255.0
        per_prompt.append({
            "prompt_idx": p,
            "psnr_db": psnr(ref, cand),
            "l1": l1(ref, cand),
        })
    psnrs = [x["psnr_db"] for x in per_prompt]
    l1s = [x["l1"] for x in per_prompt]
    return {
        "label": label,
        "directory": str(cand_dir),
        "mean_psnr_db": float(np.mean(psnrs)),
        "mean_l1": float(np.mean(l1s)),
        "min_psnr_db": float(np.min(psnrs)),
        "max_l1": float(np.max(l1s)),
        "per_prompt": per_prompt,
    }


def main() -> None:
    ref_label, ref_dir = REFERENCE
    if not (ref_dir / prompt_filename(0)).exists():
        raise SystemExit(f"reference images missing: {ref_dir}")

    results = {
        "reference": {"label": ref_label, "directory": str(ref_dir)},
        "metric_notes": (
            "PSNR and pixel-L1 on 0-1 normalized RGB. LPIPS skipped because "
            "the `lpips` package is not installed locally and requires torch "
            "plus a model download; this task is laptop-only."
        ),
        "resolution": 1024,
        "comparisons": [],
    }
    for label, cand_dir in CANDIDATES:
        entry = evaluate(label, cand_dir, ref_dir)
        if entry is not None:
            results["comparisons"].append(entry)
            print(f"{label:15s}  mean PSNR {entry['mean_psnr_db']:6.2f} dB   "
                  f"mean L1 {entry['mean_l1']:.4f}")

    out = OUT_DIR / "metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
