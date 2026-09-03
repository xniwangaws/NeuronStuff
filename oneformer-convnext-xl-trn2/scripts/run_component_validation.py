#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch_neuronx  # noqa: F401 - registers Neuron TorchScript classes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_validation import percentile, semantic_map, tensor_metrics


def synchronize_output(output) -> None:
    if isinstance(output, (tuple, list)):
        output[0].cpu()
    else:
        output.cpu()


def benchmark_callable(fn, warmup: int, runs: int) -> dict[str, float]:
    for _ in range(warmup):
        synchronize_output(fn())

    latencies_ms = []
    for _ in range(runs):
        start = time.perf_counter()
        synchronize_output(fn())
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    latencies_ms.sort()
    return {
        "min": min(latencies_ms),
        "mean": sum(latencies_ms) / len(latencies_ms),
        "p50": percentile(latencies_ms, 0.50),
        "p90": percentile(latencies_ms, 0.90),
        "max": max(latencies_ms),
        "runs": runs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--components-dir",
        default="agent_artifacts/traces/components",
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
        default="agent_artifacts/results/component_validation.json",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--cosine-threshold", type=float, default=0.99)
    parser.add_argument("--pixel-threshold", type=float, default=0.95)
    parser.add_argument("--fine-pixel-split", action="store_true")
    return parser.parse_args()


def run_pipeline(
    pixel_model: torch.jit.ScriptModule | None,
    backbone_model: torch.jit.ScriptModule | None,
    pixel_decoder_model: torch.jit.ScriptModule | None,
    task_model: torch.jit.ScriptModule,
    transformer_model: torch.jit.ScriptModule,
    pixel_values: torch.Tensor,
    task_inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if backbone_model is not None:
        backbone_outputs = backbone_model(pixel_values)
        pixel_outputs = pixel_decoder_model(*backbone_outputs)
    else:
        pixel_outputs = pixel_model(pixel_values)
    task_token = task_model(task_inputs)
    return transformer_model(*pixel_outputs, task_token)


def main() -> None:
    args = parse_args()
    components_dir = Path(args.components_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    golden = torch.load(args.golden, map_location="cpu", weights_only=True)
    if args.fine_pixel_split:
        backbone_model = torch.jit.load(str(components_dir / "backbone.pt"))
        pixel_decoder_model = torch.jit.load(
            str(components_dir / "pixel_decoder.pt")
        )
        pixel_model = None
    else:
        pixel_model = torch.jit.load(str(components_dir / "pixel_level.pt"))
        backbone_model = None
        pixel_decoder_model = None
    task_model = torch.jit.load(str(components_dir / "task_encoder.pt"))
    transformer_model = torch.jit.load(str(components_dir / "transformer.pt"))

    pixel_values = inputs["pixel_values"].contiguous()
    task_inputs = inputs["task_inputs"].contiguous()

    with torch.no_grad():
        component_latency_ms = {}
        if args.fine_pixel_split:
            backbone_outputs = backbone_model(pixel_values)
            pixel_outputs = pixel_decoder_model(*backbone_outputs)
            component_latency_ms["backbone"] = benchmark_callable(
                lambda: backbone_model(pixel_values),
                args.warmup,
                args.runs,
            )
            component_latency_ms["pixel_decoder"] = benchmark_callable(
                lambda: pixel_decoder_model(*backbone_outputs),
                args.warmup,
                args.runs,
            )
        else:
            backbone_outputs = None
            pixel_outputs = pixel_model(pixel_values)
            component_latency_ms["pixel_level"] = benchmark_callable(
                lambda: pixel_model(pixel_values),
                args.warmup,
                args.runs,
            )
        task_token = task_model(task_inputs)
        component_latency_ms["task_encoder"] = benchmark_callable(
            lambda: task_model(task_inputs),
            args.warmup,
            args.runs,
        )
        component_latency_ms["transformer"] = benchmark_callable(
            lambda: transformer_model(*pixel_outputs, task_token),
            args.warmup,
            args.runs,
        )

        for _ in range(args.warmup):
            warmup_outputs = run_pipeline(
                pixel_model,
                backbone_model,
                pixel_decoder_model,
                task_model,
                transformer_model,
                pixel_values,
                task_inputs,
            )
            warmup_outputs[0].cpu()

        latencies_ms = []
        neuron_outputs = None
        for _ in range(args.runs):
            start = time.perf_counter()
            neuron_outputs = run_pipeline(
                pixel_model,
                backbone_model,
                pixel_decoder_model,
                task_model,
                transformer_model,
                pixel_values,
                task_inputs,
            )
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
    pixel_agreement = (
        neuron_semantic[0] == golden["semantic"]
    ).float().mean().item()

    latencies_ms.sort()
    passed = (
        class_metrics["cosine_similarity"] >= args.cosine_threshold
        and mask_metrics["cosine_similarity"] >= args.cosine_threshold
        and pixel_agreement >= args.pixel_threshold
    )
    report = {
        "passed": passed,
        "class_logits": class_metrics,
        "mask_logits": mask_metrics,
        "semantic_pixel_agreement": pixel_agreement,
        "component_latency_ms": component_latency_ms,
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
        raise ValueError("Component validation did not meet the configured gates")


if __name__ == "__main__":
    main()
