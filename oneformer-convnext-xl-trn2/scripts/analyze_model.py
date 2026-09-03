#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import torch
import torch_neuronx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neuron_port.modeling import load_oneformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="shi-labs/oneformer_ade20k_swin_tiny",
    )
    parser.add_argument(
        "--inputs",
        default="agent_artifacts/data/reference/inputs.pt",
    )
    parser.add_argument("--cache-dir", default="agent_artifacts/data/hf_cache")
    parser.add_argument(
        "--output",
        default="agent_artifacts/results/neuron_analyze.json",
    )
    parser.add_argument(
        "--compiler-workdir",
        default="agent_artifacts/traces/analyze",
    )
    parser.add_argument("--custom-grid-sample", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    pixel_values = inputs["pixel_values"].contiguous()
    task_inputs = inputs["task_inputs"].contiguous()
    _, wrapper = load_oneformer(
        args.model_id,
        cache_dir=args.cache_dir,
        use_custom_grid_sample=args.custom_grid_sample,
        local_files_only=True,
    )

    report = torch_neuronx.analyze(
        wrapper,
        (pixel_values, task_inputs),
        compiler_workdir=args.compiler_workdir,
        max_workers=args.max_workers,
        is_hf_transformers=True,
        cleanup=False,
    )
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
