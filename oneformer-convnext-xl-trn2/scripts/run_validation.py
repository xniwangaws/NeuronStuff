#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import torch
import torch_neuronx  # noqa: F401 - registers Neuron TorchScript classes
from torch import Tensor, nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compiled-model",
        default="agent_artifacts/traces/oneformer_512.pt",
    )
    parser.add_argument(
        "--inputs",
        default="agent_artifacts/data/reference/inputs.pt",
    )
    parser.add_argument(
        "--golden",
        default="agent_artifacts/data/reference/golden.pt",
    )
    parser.add_argument(
        "--output",
        default="agent_artifacts/results/validation.json",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--cosine-threshold", type=float, default=0.99)
    parser.add_argument("--pixel-threshold", type=float, default=0.95)
    return parser.parse_args()


def tensor_metrics(actual: Tensor, expected: Tensor) -> dict[str, float]:
    actual_flat = actual.float().flatten()
    expected_flat = expected.float().flatten()
    difference = (actual_flat - expected_flat).abs()
    return {
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "cosine_similarity": nn.functional.cosine_similarity(
            actual_flat,
            expected_flat,
            dim=0,
        ).item(),
    }


def semantic_map(
    class_logits: Tensor,
    mask_logits: Tensor,
    target_size: tuple[int, int] | None = None,
) -> Tensor:
    if target_size is not None:
        mask_logits = nn.functional.interpolate(
            mask_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    class_probabilities = class_logits.softmax(dim=-1)[..., :-1]
    mask_probabilities = mask_logits.sigmoid()
    semantic_scores = torch.einsum(
        "bqc,bqhw->bchw",
        class_probabilities,
        mask_probabilities,
    )
    return semantic_scores.argmax(dim=1)


def percentile(sorted_values: list[float], quantile: float) -> float:
    index = round((len(sorted_values) - 1) * quantile)
    return sorted_values[index]


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    golden = torch.load(args.golden, map_location="cpu", weights_only=True)
    compiled = torch.jit.load(args.compiled_model)

    pixel_values = inputs["pixel_values"].contiguous()
    task_inputs = inputs["task_inputs"].contiguous()

    with torch.no_grad():
        for _ in range(args.warmup):
            warmup_outputs = compiled(pixel_values, task_inputs)
            warmup_outputs[0].cpu()

        latencies_ms = []
        neuron_outputs = None
        for _ in range(args.runs):
            start = time.perf_counter()
            neuron_outputs = compiled(pixel_values, task_inputs)
            neuron_outputs[0].cpu()
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    class_logits, mask_logits = neuron_outputs
    class_metrics = tensor_metrics(class_logits, golden["class_logits"])
    mask_metrics = tensor_metrics(mask_logits, golden["mask_logits"])
    target_size = tuple(golden["semantic"].shape[-2:])
    neuron_semantic = semantic_map(
        class_logits,
        mask_logits,
        target_size=target_size,
    )
    golden_semantic = golden["semantic"].unsqueeze(0)
    pixel_agreement = (
        neuron_semantic == golden_semantic
    ).float().mean().item()

    latencies_ms.sort()
    passed = (
        class_metrics["cosine_similarity"] >= args.cosine_threshold
        and mask_metrics["cosine_similarity"] >= args.cosine_threshold
        and pixel_agreement >= args.pixel_threshold
    )
    report = {
        "passed": passed,
        "thresholds": {
            "cosine_similarity": args.cosine_threshold,
            "semantic_pixel_agreement": args.pixel_threshold,
        },
        "class_logits": class_metrics,
        "mask_logits": mask_metrics,
        "semantic_pixel_agreement": pixel_agreement,
        "latency_ms": {
            "min": min(latencies_ms),
            "mean": sum(latencies_ms) / len(latencies_ms),
            "p50": percentile(latencies_ms, 0.50),
            "p90": percentile(latencies_ms, 0.90),
            "max": max(latencies_ms),
            "runs": args.runs,
        },
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    if not passed:
        raise ValueError("Neuron validation did not meet the configured gates")


if __name__ == "__main__":
    main()
