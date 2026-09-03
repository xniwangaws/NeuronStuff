#!/usr/bin/env python3

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch_neuronx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neuron_port.components import PixelDecoderCore
from neuron_port.modeling import load_oneformer
from neuron_port.nki_ops import (
    MSDA_SPATIAL_SHAPES,
    NkiFusedPixelDecoderEncoderLayerCore,
    NkiFusedPixelDecoderEncoderStackCore,
)
from neuron_port.ops import multi_scale_deformable_attention_bilinear
from scripts.compile_pixel_decoder_micro_pipeline import (
    AttentionProjectionCore,
    EncoderLayerPostCore,
    PixelDecoderInputCore,
)
from scripts.compile_pixel_decoder_pipeline import PixelDecoderOutputCore
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-layers", type=int, default=6)
    parser.add_argument("--lnc", type=int, choices=(1, 2), default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--fuse-stack", action="store_true")
    parser.add_argument("--force-recompile", action="store_true")
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


def metrics_tree(actual, expected):
    if isinstance(actual, (tuple, list)):
        return [
            tensor_metrics(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        ]
    return tensor_metrics(actual, expected)


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_layers <= 6:
        raise ValueError("--max-layers must be in [1, 6]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "metadata.json"
    compiler_args = (
        "--model-type=unet-inference -O1 "
        "--auto-cast=all --auto-cast-type=bf16"
    )

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
    full_decoder = PixelDecoderCore(wrapper.core).eval()
    input_core = PixelDecoderInputCore(decoder).eval()
    output_core = PixelDecoderOutputCore(decoder).eval()

    with torch.no_grad():
        backbone_outputs = tuple(backbone(pixel_values).feature_maps)
        expected_decoder_outputs = full_decoder(*backbone_outputs)
        input_args = (
            backbone_outputs[1],
            backbone_outputs[2],
            backbone_outputs[3],
        )
        cpu_hidden = input_core(*input_args)
        position_embeddings = build_pixel_position_embeddings(
            decoder,
            backbone_outputs,
        )
        spatial_shapes = torch.tensor(
            MSDA_SPATIAL_SHAPES,
            dtype=torch.long,
        )
        valid_ratios = torch.ones(
            (cpu_hidden.shape[0], len(MSDA_SPATIAL_SHAPES), 2),
            dtype=cpu_hidden.dtype,
        )
        reference_points = decoder.encoder.get_reference_points(
            spatial_shapes,
            valid_ratios,
            device=cpu_hidden.device,
        )

        cpu_records = []
        for layer in decoder.encoder.layers:
            projection = AttentionProjectionCore(
                layer.self_attn,
                reference_points,
            ).eval()
            post = EncoderLayerPostCore(layer).eval()
            value, locations, weights = projection(
                cpu_hidden,
                position_embeddings,
            )
            attention_output = multi_scale_deformable_attention_bilinear(
                value,
                MSDA_SPATIAL_SHAPES,
                locations,
                weights,
            )
            attention_output = layer.self_attn.output_proj(
                attention_output
            )
            next_hidden = post(cpu_hidden, attention_output)
            cpu_records.append(
                {
                    "input": cpu_hidden,
                    "output": next_hidden,
                }
            )
            cpu_hidden = next_hidden
        cpu_pipeline_outputs = output_core(
            cpu_hidden,
            backbone_outputs[0],
        )

    existing_dir = Path(args.existing_pixel_dir)
    reused = []
    for name in ("input.pt", "output.pt"):
        source = existing_dir / name
        destination = output_dir / name
        if not destination.exists():
            shutil.copy2(source, destination)
        reused.append(
            {
                "name": name,
                "source": str(source),
                "artifact_bytes": destination.stat().st_size,
            }
        )

    compiled_input = torch.jit.load(str(output_dir / "input.pt"))
    compiled_output = torch.jit.load(str(output_dir / "output.pt"))
    with torch.no_grad():
        initial_neuron_hidden = compiled_input(*input_args)
        neuron_hidden = initial_neuron_hidden

    layer_results = []
    compiled_layers = []
    layer_runtime_inputs = []
    for index, (layer, record) in enumerate(
        zip(decoder.encoder.layers, cpu_records)
    ):
        if index >= args.max_layers:
            break
        module = NkiFusedPixelDecoderEncoderLayerCore(
            layer,
            reference_points,
            args.lnc,
        ).eval()
        artifact_path = output_dir / f"layer_{index}.pt"
        workdir = output_dir / f"layer_{index}_workdir"
        start = time.perf_counter()
        if artifact_path.exists() and not args.force_recompile:
            compiled = torch.jit.load(str(artifact_path))
            compile_seconds = 0.0
            skipped_existing = True
        else:
            compiled = torch_neuronx.trace(
                module,
                (record["input"], position_embeddings),
                compiler_args=compiler_args,
                compiler_workdir=str(workdir),
            )
            compile_seconds = time.perf_counter() - start
            torch.jit.save(compiled, str(artifact_path))
            skipped_existing = False

        layer_input = neuron_hidden
        with torch.no_grad():
            neuron_hidden = compiled(
                layer_input,
                position_embeddings,
            )
        layer_results.append(
            {
                "name": f"layer_{index}",
                "compile_seconds": compile_seconds,
                "skipped_existing": skipped_existing,
                "artifact_bytes": artifact_path.stat().st_size,
                "metrics_vs_cpu_layer_output": tensor_metrics(
                    neuron_hidden,
                    record["output"],
                ),
            }
        )
        compiled_layers.append(compiled)
        layer_runtime_inputs.append(layer_input)
        gc.collect()

    layer_latencies = {
        f"layer_{index}": benchmark_callable(
            lambda module=module, hidden=hidden: module(
                hidden,
                position_embeddings,
            ),
            args.warmup,
            args.runs,
        )
        for index, (module, hidden) in enumerate(
            zip(compiled_layers, layer_runtime_inputs)
        )
    }

    complete = args.max_layers == len(decoder.encoder.layers)
    compiled_stack = None
    stack_result = None
    stack_latency = None
    if complete and args.fuse_stack:
        stack_module = NkiFusedPixelDecoderEncoderStackCore(
            decoder.encoder.layers,
            reference_points,
            args.lnc,
        ).eval()
        stack_path = output_dir / "encoder_stack.pt"
        stack_workdir = output_dir / "encoder_stack_workdir"
        start = time.perf_counter()
        if stack_path.exists() and not args.force_recompile:
            compiled_stack = torch.jit.load(str(stack_path))
            compile_seconds = 0.0
            skipped_existing = True
        else:
            compiled_stack = torch_neuronx.trace(
                stack_module,
                (cpu_records[0]["input"], position_embeddings),
                compiler_args=compiler_args,
                compiler_workdir=str(stack_workdir),
            )
            compile_seconds = time.perf_counter() - start
            torch.jit.save(compiled_stack, str(stack_path))
            skipped_existing = False
        with torch.no_grad():
            stack_output = compiled_stack(
                initial_neuron_hidden,
                position_embeddings,
            )
        stack_result = {
            "name": "encoder_stack",
            "compile_seconds": compile_seconds,
            "skipped_existing": skipped_existing,
            "artifact_bytes": stack_path.stat().st_size,
            "metrics_vs_cpu_stack_output": tensor_metrics(
                stack_output,
                cpu_hidden,
            ),
        }
        stack_latency = benchmark_callable(
            lambda: compiled_stack(
                initial_neuron_hidden,
                position_embeddings,
            ),
            args.warmup,
            args.runs,
        )

    report = {
        "model_id": args.model_id,
        "backend": "nki-ms-deformable-attention",
        "precision": "bf16-all",
        "lnc": args.lnc,
        "compiler_args": compiler_args,
        "compiled_layers": len(compiled_layers),
        "fused_stack": args.fuse_stack,
        "complete": complete,
        "reused_components": reused,
        "cpu_split_pipeline_vs_full_decoder": metrics_tree(
            cpu_pipeline_outputs,
            expected_decoder_outputs,
        ),
        "layers": layer_results,
        "layer_latency_ms": layer_latencies,
    }
    if stack_result is not None:
        report["stack"] = stack_result
        report["stack_latency_ms"] = stack_latency

    if complete:
        with torch.no_grad():
            output_hidden = (
                compiled_stack(initial_neuron_hidden, position_embeddings)
                if compiled_stack is not None
                else neuron_hidden
            )
            neuron_outputs = compiled_output(
                output_hidden,
                backbone_outputs[0],
            )
            input_latency = benchmark_callable(
                lambda: compiled_input(*input_args),
                args.warmup,
                args.runs,
            )
            output_latency = benchmark_callable(
                lambda: compiled_output(
                    neuron_hidden,
                    backbone_outputs[0],
                ),
                args.warmup,
                args.runs,
            )

            def full_pixel_decoder():
                hidden = compiled_input(*input_args)
                if compiled_stack is not None:
                    hidden = compiled_stack(
                        hidden,
                        position_embeddings,
                    )
                else:
                    for layer in compiled_layers:
                        hidden = layer(hidden, position_embeddings)
                return compiled_output(hidden, backbone_outputs[0])

            full_latency = benchmark_callable(
                full_pixel_decoder,
                args.warmup,
                args.runs,
            )
        report.update(
            {
                "component_count": 3 if compiled_stack is not None else 8,
                "runtime_invocation_count": (
                    3 if compiled_stack is not None else 8
                ),
                "neuron_pipeline_vs_full_decoder": metrics_tree(
                    neuron_outputs,
                    expected_decoder_outputs,
                ),
                "component_latency_ms": {
                    "input": input_latency,
                    **(
                        {"encoder_stack": stack_latency}
                        if compiled_stack is not None
                        else layer_latencies
                    ),
                    "output": output_latency,
                },
                "component_mean_sum_ms": (
                    input_latency["mean"]
                    + (
                        stack_latency["mean"]
                        if compiled_stack is not None
                        else sum(
                            latency["mean"]
                            for latency in layer_latencies.values()
                        )
                    )
                    + output_latency["mean"]
                ),
                "full_pixel_decoder_latency_ms": full_latency,
            }
        )

    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
