#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch_neuronx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neuron_port.convnext_stage2_nki import (
    NkiConvNextStage2FusedLayerCore,
)
from neuron_port.convnext_stage1_nki import (
    NkiConvNextStage1FusedLayerCore,
)
from neuron_port.modeling import load_oneformer
from scripts.run_validation import percentile, tensor_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--stage-index",
        type=int,
        choices=(1, 2),
        default=2,
    )
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--custom-grid-sample", action="store_true")
    return parser.parse_args()


def synchronize_output(output: torch.Tensor) -> None:
    output.cpu()


def benchmark_callable(fn, warmup: int, runs: int) -> dict[str, float]:
    for _ in range(warmup):
        synchronize_output(fn())
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        synchronize_output(fn())
        latencies.append((time.perf_counter() - start) * 1000.0)
    latencies.sort()
    return {
        "min": min(latencies),
        "mean": sum(latencies) / len(latencies),
        "p50": percentile(latencies, 0.50),
        "p90": percentile(latencies, 0.90),
        "max": max(latencies),
        "runs": runs,
    }


def place_on_device(module, device_id: int) -> None:
    torch_neuronx.move_trace_to_device(module, device_id)
    torch_neuronx.set_neuron_cores(
        module,
        start_nc=device_id,
        nc_count=1,
    )


def trace_component(
    name: str,
    module: torch.nn.Module,
    example_input: torch.Tensor,
    output_dir: Path,
) -> tuple[torch.jit.ScriptModule, dict]:
    artifact_path = output_dir / f"{name}.pt"
    start = time.perf_counter()
    compiled = torch_neuronx.trace(
        module,
        (example_input,),
        compiler_args=(
            "--model-type=unet-inference -O1 "
            "--auto-cast=all --auto-cast-type=bf16"
        ),
        compiler_workdir=str(output_dir / f"{name}_workdir"),
    )
    compile_seconds = time.perf_counter() - start
    torch.jit.save(compiled, str(artifact_path))
    return compiled, {
        "name": name,
        "compile_seconds": compile_seconds,
        "artifact_bytes": artifact_path.stat().st_size,
    }


def stage_layer_input(
    backbone,
    pixel_values: torch.Tensor,
    stage_index: int,
) -> torch.Tensor:
    features = backbone.embeddings(pixel_values)
    for prior_stage_index in range(stage_index):
        stage = backbone.encoder.stages[prior_stage_index]
        for downsampling in stage.downsampling_layer:
            features = downsampling(features)
        for layer in stage.layers:
            features = layer(features)

    target_stage = backbone.encoder.stages[stage_index]
    for downsampling in target_stage.downsampling_layer:
        features = downsampling(features)
    return features


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    pixel_values = inputs["pixel_values"].contiguous()
    _, wrapper = load_oneformer(
        args.model_id,
        cache_dir=args.cache_dir,
        use_custom_grid_sample=args.custom_grid_sample,
        local_files_only=True,
    )
    backbone = wrapper.core.pixel_level_module.encoder.eval()
    layer = backbone.encoder.stages[args.stage_index].layers[
        args.layer_index
    ].eval()
    nki_core = (
        NkiConvNextStage1FusedLayerCore
        if args.stage_index == 1
        else NkiConvNextStage2FusedLayerCore
    )
    nki_layer = nki_core(layer, lnc=2).eval()

    with torch.no_grad():
        layer_input = stage_layer_input(
            backbone,
            pixel_values,
            args.stage_index,
        )
        expected = layer(layer_input)

    compiled_nki, nki_compile = trace_component(
        "nki_fused_full_layer",
        nki_layer,
        layer_input,
        output_dir,
    )
    compiled_native, native_compile = trace_component(
        "native_full_layer",
        layer,
        layer_input,
        output_dir,
    )

    place_on_device(compiled_nki, args.device_id)
    place_on_device(compiled_native, args.device_id)
    device = torch.device(f"privateuseone:{args.device_id}")
    neuron_input = layer_input.to(device)

    with torch.no_grad():
        nki_output = compiled_nki(neuron_input).cpu()
        native_output = compiled_native(neuron_input).cpu()
        latency = {
            "nki_fused_full_layer": benchmark_callable(
                lambda: compiled_nki(neuron_input),
                args.warmup,
                args.runs,
            ),
            "native_full_layer": benchmark_callable(
                lambda: compiled_native(neuron_input),
                args.warmup,
                args.runs,
            ),
        }

    report = {
        "model_id": args.model_id,
        "stage": args.stage_index,
        "layer_index": args.layer_index,
        "shape": list(layer_input.shape),
        "precision": "fp32-io+bf16-pointwise",
        "lnc": 2,
        "compile": [nki_compile, native_compile],
        "metrics": {
            "nki_vs_cpu": tensor_metrics(nki_output, expected),
            "native_vs_cpu": tensor_metrics(native_output, expected),
            "nki_vs_native_neuron": tensor_metrics(
                nki_output,
                native_output,
            ),
        },
        "latency_ms": latency,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
