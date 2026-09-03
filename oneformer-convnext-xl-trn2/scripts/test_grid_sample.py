#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neuron_port.ops import CustomGridSample, RawGridSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--implementation",
        choices=("raw", "custom"),
        default="raw",
    )
    parser.add_argument(
        "--output-dir",
        default="agent_artifacts/traces/grid_sample",
    )
    return parser.parse_args()


def make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    input_tensor = torch.randn(8, 32, 64, 64)
    grid = torch.empty(8, 256, 4, 2).uniform_(-1.15, 1.15)
    return input_tensor, grid


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_flat = actual.float().flatten()
    expected_flat = expected.float().flatten()
    return {
        "max_abs_error": (actual_flat - expected_flat).abs().max().item(),
        "mean_abs_error": (actual_flat - expected_flat).abs().mean().item(),
        "cosine_similarity": nn.functional.cosine_similarity(
            actual_flat,
            expected_flat,
            dim=0,
        ).item(),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_tensor, grid = make_inputs()
    raw_module = RawGridSample().eval()
    custom_module = CustomGridSample().eval()

    with torch.no_grad():
        raw_output = raw_module(input_tensor, grid)
        custom_output = custom_module(input_tensor, grid)
    cpu_metrics = metrics(custom_output, raw_output)
    print(json.dumps({"cpu_custom_vs_raw": cpu_metrics}, indent=2))

    if not args.compile:
        if cpu_metrics["max_abs_error"] > 1e-5:
            raise ValueError("Custom bilinear sampler does not match grid_sample")
        return

    import torch_neuronx

    module = raw_module if args.implementation == "raw" else custom_module
    compiler_args = "--model-type=unet-inference -O1 --auto-cast=none"
    traced = torch_neuronx.trace(
        module,
        (input_tensor, grid),
        compiler_args=compiler_args,
        compiler_workdir=str(output_dir / f"{args.implementation}_workdir"),
    )
    trace_path = output_dir / f"{args.implementation}.pt"
    torch.jit.save(traced, str(trace_path))

    with torch.no_grad():
        neuron_output = traced(input_tensor, grid)
    neuron_metrics = metrics(neuron_output, raw_output)
    report = {
        "implementation": args.implementation,
        "compiler_args": compiler_args,
        "trace_path": str(trace_path),
        "cpu_custom_vs_raw": cpu_metrics,
        "neuron_vs_raw": neuron_metrics,
    }
    (output_dir / f"{args.implementation}_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
