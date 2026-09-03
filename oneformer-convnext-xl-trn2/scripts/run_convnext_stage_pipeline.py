#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch_neuronx  # noqa: F401 - registers Neuron TorchScript classes
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neuron_port.modeling import load_oneformer
from scripts.run_validation import percentile, tensor_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--components-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--custom-grid-sample", action="store_true")
    return parser.parse_args()


def synchronize_output(output) -> None:
    if isinstance(output, (tuple, list)):
        for value in output:
            value.cpu()
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


class NeuronConvNextStagePipeline:
    def __init__(
        self,
        components_dir: Path,
        stage_depths: list[int],
        chunk_size: int,
    ):
        self.embeddings = torch.jit.load(
            str(components_dir / "embeddings.pt")
        )
        self.stages = []
        for stage_index, depth in enumerate(stage_depths):
            stage_chunks = []
            for chunk_index, start in enumerate(range(0, depth, chunk_size)):
                end = min(start + chunk_size, depth)
                name = (
                    f"stage{stage_index}_chunk{chunk_index}_"
                    f"layers{start}_{end}"
                )
                stage_chunks.append(
                    (
                        name,
                        end == depth,
                        torch.jit.load(str(components_dir / f"{name}.pt")),
                    )
                )
            self.stages.append(stage_chunks)

    def __call__(
        self,
        pixel_values: Tensor,
        capture_intermediates: bool = False,
    ):
        component_inputs = {}
        component_outputs = {}

        if capture_intermediates:
            component_inputs["embeddings"] = pixel_values
        features = self.embeddings(pixel_values)
        if capture_intermediates:
            component_outputs["embeddings"] = features

        normalized_features = []
        for stage_chunks in self.stages:
            for name, is_final, module in stage_chunks:
                if capture_intermediates:
                    component_inputs[name] = features
                output = module(features)
                if capture_intermediates:
                    component_outputs[name] = output
                if is_final:
                    features, normalized = output
                    normalized_features.append(normalized)
                else:
                    features = output

        result = tuple(normalized_features)
        if capture_intermediates:
            return result, component_inputs, component_outputs
        return result


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    components_dir = Path(args.components_dir)

    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    pixel_values = inputs["pixel_values"].contiguous()
    _, wrapper = load_oneformer(
        args.model_id,
        cache_dir=args.cache_dir,
        use_custom_grid_sample=args.custom_grid_sample,
        local_files_only=True,
    )
    backbone = wrapper.core.pixel_level_module.encoder.eval()
    stage_depths = [len(stage.layers) for stage in backbone.encoder.stages]
    pipeline = NeuronConvNextStagePipeline(
        components_dir,
        stage_depths,
        args.chunk_size,
    )

    with torch.no_grad():
        reference_features = tuple(backbone(pixel_values).feature_maps)
        (
            neuron_features,
            component_inputs,
            component_outputs,
        ) = pipeline(pixel_values, capture_intermediates=True)

        component_latency_ms = {
            "embeddings": benchmark_callable(
                lambda: pipeline.embeddings(
                    component_inputs["embeddings"]
                ),
                args.warmup,
                args.runs,
            )
        }
        for stage_chunks in pipeline.stages:
            for name, _, module in stage_chunks:
                component_input = component_inputs[name]
                component_latency_ms[name] = benchmark_callable(
                    lambda module=module, value=component_input: module(value),
                    args.warmup,
                    args.runs,
                )

        end_to_end_latency_ms = benchmark_callable(
            lambda: pipeline(pixel_values),
            args.warmup,
            args.runs,
        )

    if len(neuron_features) != len(reference_features):
        raise ValueError("Neuron pipeline returned the wrong feature count")

    feature_names = list(backbone.out_features)
    feature_metrics = {
        name: tensor_metrics(actual, expected)
        for name, actual, expected in zip(
            feature_names,
            neuron_features,
            reference_features,
        )
    }
    report = {
        "model_id": args.model_id,
        "precision": "bf16-all",
        "chunk_size": args.chunk_size,
        "stage_depths": stage_depths,
        "feature_names": feature_names,
        "feature_shapes": {
            name: list(value.shape)
            for name, value in zip(feature_names, neuron_features)
        },
        "feature_metrics": feature_metrics,
        "component_latency_ms": component_latency_ms,
        "component_mean_sum_ms": sum(
            value["mean"] for value in component_latency_ms.values()
        ),
        "end_to_end_latency_ms": end_to_end_latency_ms,
        "all_feature_cosine_at_least_0_99": all(
            value["cosine_similarity"] >= 0.99
            for value in feature_metrics.values()
        ),
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
