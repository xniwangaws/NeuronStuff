#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from transformers.models.oneformer import modeling_oneformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neuron_port.modeling import load_oneformer, patch_oneformer_for_neuron


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
    parser.add_argument("--max-abs-threshold", type=float, default=5e-4)
    parser.add_argument("--cosine-threshold", type=float, default=0.999)
    return parser.parse_args()


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "cosine_similarity": nn.functional.cosine_similarity(
            actual.float().flatten(),
            expected.float().flatten(),
            dim=0,
        ).item(),
    }


def main() -> None:
    args = parse_args()
    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    pixel_values = inputs["pixel_values"]
    task_inputs = inputs["task_inputs"]

    _, wrapper = load_oneformer(
        args.model_id,
        cache_dir=args.cache_dir,
        use_custom_grid_sample=False,
        local_files_only=True,
    )
    with torch.no_grad():
        raw_class, raw_mask = wrapper(pixel_values, task_inputs)

    patch_oneformer_for_neuron(modeling_oneformer)
    with torch.no_grad():
        custom_class, custom_mask = wrapper(pixel_values, task_inputs)

    report = {
        "class_logits": metrics(custom_class, raw_class),
        "mask_logits": metrics(custom_mask, raw_mask),
    }
    report["passed"] = (
        report["class_logits"]["max_abs_error"] <= args.max_abs_threshold
        and report["mask_logits"]["max_abs_error"] <= args.max_abs_threshold
        and report["class_logits"]["cosine_similarity"]
        >= args.cosine_threshold
        and report["mask_logits"]["cosine_similarity"]
        >= args.cosine_threshold
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise ValueError("Full custom attention path diverged from raw OneFormer")


if __name__ == "__main__":
    main()
