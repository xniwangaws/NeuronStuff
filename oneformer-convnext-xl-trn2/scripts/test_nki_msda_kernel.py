#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch_neuronx
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neuron_port.modeling import load_oneformer
from neuron_port.nki_ops import (
    MSDA_LEVEL_START_INDEX,
    MSDA_SPATIAL_SHAPES,
    NkiMsdaOutputProjectionCore,
    sanitize_far_oob_samples,
)
from neuron_port.ops import multi_scale_deformable_attention_bilinear
from scripts.run_full_oneformer_pipeline import (
    build_pixel_position_embeddings,
)
from scripts.run_validation import percentile, tensor_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--existing-pixel-dir", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--compiler-workdir", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--lnc", type=int, choices=(1, 2), default=2)
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--custom-grid-sample", action="store_true")
    return parser.parse_args()


def synchronize_output(output: Tensor) -> None:
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


def main() -> None:
    args = parse_args()
    output_model = Path(args.output_model)
    output_report = Path(args.output_report)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    pixel_values = inputs["pixel_values"].contiguous()
    _, wrapper = load_oneformer(
        args.model_id,
        cache_dir=args.cache_dir,
        use_custom_grid_sample=args.custom_grid_sample,
        local_files_only=True,
    )
    backbone = wrapper.core.pixel_level_module.encoder.eval()
    decoder = wrapper.core.pixel_level_module.decoder.eval()
    attention = decoder.encoder.layers[0].self_attn

    existing_dir = Path(args.existing_pixel_dir)
    compiled_input = torch.jit.load(str(existing_dir / "input.pt"))
    compiled_projection = torch.jit.load(
        str(existing_dir / "layer_0_projection.pt")
    )
    compiled_samplers = [
        torch.jit.load(str(existing_dir / f"sampler_{index}.pt"))
        for index in range(3)
    ]
    compiled_combine = torch.jit.load(
        str(existing_dir / "layer_0_combine.pt")
    )

    with torch.no_grad():
        backbone_outputs = tuple(backbone(pixel_values).feature_maps)
        position_embeddings = build_pixel_position_embeddings(
            decoder,
            backbone_outputs,
        )
        hidden = compiled_input(
            backbone_outputs[1],
            backbone_outputs[2],
            backbone_outputs[3],
        )
        value, locations, weights = compiled_projection(
            hidden,
            position_embeddings,
        )
        cpu_attention = multi_scale_deformable_attention_bilinear(
            value,
            MSDA_SPATIAL_SHAPES,
            locations,
            weights,
        )
        expected = attention.output_proj(cpu_attention)
        bf16_attention = multi_scale_deformable_attention_bilinear(
            value.to(torch.bfloat16),
            MSDA_SPATIAL_SHAPES,
            locations.to(torch.float32),
            weights.to(torch.bfloat16),
        )
        bf16_expected = attention.output_proj(
            bf16_attention.to(torch.float32)
        )
        safe_locations, safe_weights = sanitize_far_oob_samples(
            locations,
            weights,
        )
        sanitized_attention = multi_scale_deformable_attention_bilinear(
            value,
            MSDA_SPATIAL_SHAPES,
            safe_locations,
            safe_weights,
        )
        sanitized_expected = attention.output_proj(
            sanitized_attention
        )
        old_sampled = [
            sampler(value, locations)
            for sampler in compiled_samplers
        ]
        old_output = compiled_combine(*old_sampled, weights)

    module = NkiMsdaOutputProjectionCore(attention, args.lnc).eval()
    compiler_args = (
        "--model-type=unet-inference -O1 "
        "--auto-cast=all --auto-cast-type=bf16"
    )
    compile_seconds = 0.0
    if output_model.exists() and not args.force_recompile:
        compiled_nki = torch.jit.load(str(output_model))
    else:
        start = time.perf_counter()
        compiled_nki = torch_neuronx.trace(
            module,
            (value, locations, weights),
            compiler_args=compiler_args,
            compiler_workdir=args.compiler_workdir,
        )
        compile_seconds = time.perf_counter() - start
        torch.jit.save(compiled_nki, str(output_model))

    with torch.no_grad():
        nki_output = compiled_nki(value, locations, weights)
        nki_latency = benchmark_callable(
            lambda: compiled_nki(value, locations, weights),
            args.warmup,
            args.runs,
        )

        def old_pipeline():
            sampled = [
                sampler(value, locations)
                for sampler in compiled_samplers
            ]
            return compiled_combine(*sampled, weights)

        old_latency = benchmark_callable(
            old_pipeline,
            args.warmup,
            args.runs,
        )

    report = {
        "input_shapes": {
            "value": list(value.shape),
            "sampling_locations": list(locations.shape),
            "attention_weights": list(weights.shape),
        },
        "input_dtypes": {
            "value": str(value.dtype),
            "sampling_locations": str(locations.dtype),
            "attention_weights": str(weights.dtype),
        },
        "kernel_input_dtypes": {
            "value": "torch.bfloat16",
            "sampling_locations": "torch.float32",
            "attention_weights": "torch.bfloat16",
        },
        "far_oob_sanitization": True,
        "lnc": args.lnc,
        "compiler_args": compiler_args,
        "compile_seconds": compile_seconds,
        "artifact_bytes": output_model.stat().st_size,
        "old_pipeline_vs_cpu": tensor_metrics(old_output, expected),
        "bf16_reference_vs_cpu": tensor_metrics(
            bf16_expected,
            expected,
        ),
        "sanitized_reference_vs_cpu": tensor_metrics(
            sanitized_expected,
            expected,
        ),
        "nki_vs_cpu": tensor_metrics(nki_output, expected),
        "nki_vs_bf16_reference": tensor_metrics(
            nki_output,
            bf16_expected,
        ),
        "nki_vs_old_pipeline": tensor_metrics(nki_output, old_output),
        "latency_ms": {
            "old_three_samplers_plus_combine": old_latency,
            "nki_msda_plus_output_projection": nki_latency,
            "speedup": old_latency["mean"] / nki_latency["mean"],
        },
    }
    output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
