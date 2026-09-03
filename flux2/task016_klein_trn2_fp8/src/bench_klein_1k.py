#!/usr/bin/env python3
"""Reproducible 1024x1024 benchmark for FLUX.2-klein-base-9B on Neuron."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from application import NeuronFlux2KleinApplication, create_flux2_klein_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/mnt/nvme/flux2-klein/weights",
        help="Downloaded Hugging Face model directory.",
    )
    parser.add_argument(
        "--compile-dir",
        default="/mnt/nvme/flux2-klein/compiled_bf16",
    )
    parser.add_argument(
        "--transformer-checkpoint",
        default=os.environ.get("FLUX2_FP8_CHECKPOINT"),
        help=(
            "Optional prequantized NxDI-format transformer checkpoint directory. "
            "The original model directory is still used for config, text encoder, "
            "tokenizer, scheduler, and VAE."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/nvme/flux2-klein/outputs_bf16",
    )
    parser.add_argument(
        "--prompt",
        default="A cat holding a sign that says hello world",
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--warmups", type=int, default=1)
    return parser.parse_args()


def image_metrics(image) -> dict[str, float | bool]:
    arr = np.asarray(image).astype(np.float32)
    left = arr[:, :-1, :].reshape(-1)
    right = arr[:, 1:, :].reshape(-1)
    correlation = float(np.corrcoef(left, right)[0, 1])
    std = float(arr.std())
    mean = float(arr.mean())
    valid = bool(
        np.isfinite(arr).all()
        and np.isfinite(correlation)
        and std > 40.0
        and correlation > 0.60
    )
    return {
        "mean": mean,
        "std": std,
        "neighbor_correlation": correlation,
        "valid_image": valid,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "model": args.model,
        "transformer_checkpoint": args.transformer_checkpoint,
        "compile_dir": args.compile_dir,
        "resolution": [args.height, args.width],
        "steps": args.steps,
        "guidance": args.guidance,
        "tp": args.tp,
        "seeds": args.seeds,
        "prompt": args.prompt,
        "neuron_logical_nc_config": os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
        "neuron_rt_virtual_core_size": os.environ.get(
            "NEURON_RT_VIRTUAL_CORE_SIZE"
        ),
        "neuron_rt_visible_cores": os.environ.get("NEURON_RT_VISIBLE_CORES"),
        "flux2_fp8_mlp": os.environ.get("FLUX2_FP8_MLP", "0"),
        "flux2_fp8_scope": os.environ.get("FLUX2_FP8_SCOPE"),
        "flux2_fp8_activation": os.environ.get("FLUX2_FP8_ACTIVATION", "none"),
        "unsafe_fp8fncast": os.environ.get("UNSAFE_FP8FNCAST"),
    }

    config = create_flux2_klein_config(
        model_path=args.model,
        backbone_tp_degree=args.tp,
        dtype=torch.bfloat16,
        height=args.height,
        width=args.width,
    )
    app = NeuronFlux2KleinApplication(
        model_path=args.model,
        backbone_config=config,
        height=args.height,
        width=args.width,
        transformer_checkpoint=args.transformer_checkpoint,
    )

    started = time.perf_counter()
    app.compile(args.compile_dir)
    compile_seconds = time.perf_counter() - started

    started = time.perf_counter()
    app.load(args.compile_dir)
    load_seconds = time.perf_counter() - started

    for warmup_index in range(args.warmups):
        generator = torch.Generator(device="cpu").manual_seed(
            args.seeds[0] + 10000 + warmup_index
        )
        app(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        )

    samples = []
    for seed in args.seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        started = time.perf_counter()
        result = app(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        )
        latency = time.perf_counter() - started
        image = result.images[0]
        image_path = output_dir / f"seed{seed}_cat.png"
        image.save(image_path)
        metrics = image_metrics(image)
        sample = {
            "seed": seed,
            "latency_seconds": latency,
            "image": str(image_path),
            **metrics,
        }
        samples.append(sample)
        print(json.dumps(sample, sort_keys=True), flush=True)

    latencies = [sample["latency_seconds"] for sample in samples]
    summary = {
        "metadata": metadata,
        "compile_seconds": compile_seconds,
        "load_seconds": load_seconds,
        "mean_seconds": statistics.fmean(latencies),
        "median_seconds": statistics.median(latencies),
        "min_seconds": min(latencies),
        "max_seconds": max(latencies),
        "stdev_seconds": statistics.pstdev(latencies),
        "valid_images": sum(sample["valid_image"] for sample in samples),
        "sample_count": len(samples),
        "samples": samples,
    }
    summary_path = output_dir / "results.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
