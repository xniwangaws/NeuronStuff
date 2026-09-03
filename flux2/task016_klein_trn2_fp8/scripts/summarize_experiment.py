#!/usr/bin/env python3
"""Summarize BF16, MLP FP8, and all-linear FP8 benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--all-linear", type=Path, required=True)
    parser.add_argument("--all-linear-comparison", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--hourly-price", type=float, default=53.64 / 24)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def mode_summary(result: dict, bf16_mean: float, hourly_price: float) -> dict:
    mean_seconds = float(result["mean_seconds"])
    return {
        "mean_seconds": mean_seconds,
        "median_seconds": float(result["median_seconds"]),
        "min_seconds": float(result["min_seconds"]),
        "max_seconds": float(result["max_seconds"]),
        "stdev_seconds": float(result["stdev_seconds"]),
        "valid_images": int(result["valid_images"]),
        "sample_count": int(result["sample_count"]),
        "speedup_vs_bf16": bf16_mean / mean_seconds,
        "latency_reduction_percent": (1.0 - mean_seconds / bf16_mean) * 100,
        "cost_per_image_usd": mean_seconds / 3600 * hourly_price,
    }


def main() -> None:
    args = parse_args()
    bf16 = read_json(args.bf16)
    mlp = read_json(args.mlp)
    all_linear = read_json(args.all_linear)
    comparison = read_json(args.all_linear_comparison)
    manifest = read_json(args.checkpoint_manifest)

    bf16_mean = float(bf16["mean_seconds"])
    summary = {
        "resolution": bf16["metadata"]["resolution"],
        "steps": bf16["metadata"]["steps"],
        "hourly_price_usd": args.hourly_price,
        "bf16": {
            **mode_summary(bf16, bf16_mean, args.hourly_price),
            "speedup_vs_bf16": 1.0,
            "latency_reduction_percent": 0.0,
        },
        "fp8_mlp_dynamic": mode_summary(
            mlp,
            bf16_mean,
            args.hourly_price,
        ),
        "fp8_all_transformer_linear_dynamic": {
            **mode_summary(all_linear, bf16_mean, args.hourly_price),
            "scope_note": (
                "All Transformer Linear/GEMM projections are W8A8; "
                "normalization, RoPE, softmax, residual math, and the "
                "attention kernel remain BF16/FP32."
            ),
            "checkpoint": {
                "format": manifest["format"],
                "weight_dtype": manifest["weight_dtype"],
                "scale_dtype": manifest["scale_dtype"],
                "weight_quantization": manifest["weight_quantization"],
                "fp8_weight_tensor_count": manifest[
                    "fp8_weight_tensor_count"
                ],
                "fp8_mlp_weight_tensor_count": manifest[
                    "fp8_mlp_weight_tensor_count"
                ],
                "fp8_attention_weight_tensor_count": manifest[
                    "fp8_attention_weight_tensor_count"
                ],
                "fp8_auxiliary_weight_tensor_count": manifest[
                    "fp8_auxiliary_weight_tensor_count"
                ],
                "total_tensor_bytes": manifest["fp8_weight_bytes"]
                + manifest["scale_bytes"],
            },
            "quality_vs_bf16": {
                "mean_mse": comparison["mean_mse"],
                "mean_mae": comparison["mean_mae"],
                "mean_psnr_db": comparison["mean_psnr_db"],
                "mean_global_ssim": comparison["mean_global_ssim"],
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
