#!/usr/bin/env python3

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_neuronx
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neuron_port.modeling import load_oneformer
from scripts.compile_pixel_decoder_pipeline import PixelDecoderOutputCore
from scripts.run_full_oneformer_pipeline import (
    build_pixel_position_embeddings,
)
from scripts.run_validation import percentile, tensor_metrics


def fixed_bilinear_upsample_2x(input_tensor: Tensor) -> Tensor:
    """Exact scale-2 align_corners=False bilinear interpolation."""

    left = torch.cat(
        (input_tensor[..., :1], input_tensor[..., :-1]),
        dim=-1,
    )
    right = torch.cat(
        (input_tensor[..., 1:], input_tensor[..., -1:]),
        dim=-1,
    )
    even_x = 0.25 * left + 0.75 * input_tensor
    odd_x = 0.75 * input_tensor + 0.25 * right
    horizontal = torch.stack((even_x, odd_x), dim=-1).flatten(-2)

    top = torch.cat(
        (horizontal[:, :, :1], horizontal[:, :, :-1]),
        dim=2,
    )
    bottom = torch.cat(
        (horizontal[:, :, 1:], horizontal[:, :, -1:]),
        dim=2,
    )
    even_y = 0.25 * top + 0.75 * horizontal
    odd_y = 0.75 * horizontal + 0.25 * bottom
    return torch.stack((even_y, odd_y), dim=3).flatten(2, 3)


class PixelDecoderOutputFixedCore(PixelDecoderOutputCore):
    def forward(
        self,
        hidden_states: Tensor,
        feature_4: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        feature_32 = (
            hidden_states[:, 0:400]
            .transpose(1, 2)
            .reshape(hidden_states.shape[0], 256, 20, 20)
        )
        feature_16 = (
            hidden_states[:, 400:2000]
            .transpose(1, 2)
            .reshape(hidden_states.shape[0], 256, 40, 40)
        )
        feature_8 = (
            hidden_states[:, 2000:8400]
            .transpose(1, 2)
            .reshape(hidden_states.shape[0], 256, 80, 80)
        )
        lateral = self.lateral_conv(feature_4)
        feature_4_output = self.output_conv(
            lateral + fixed_bilinear_upsample_2x(feature_8)
        )
        return (
            feature_32,
            feature_16,
            feature_8,
            self.mask_projection(feature_4_output),
        )


class PixelDecoderOutputPrepareCore(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.lateral_conv = decoder.lateral_convs[0]

    def forward(
        self,
        hidden_states: Tensor,
        feature_4: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        feature_32 = (
            hidden_states[:, 0:400]
            .transpose(1, 2)
            .reshape(hidden_states.shape[0], 256, 20, 20)
        )
        feature_16 = (
            hidden_states[:, 400:2000]
            .transpose(1, 2)
            .reshape(hidden_states.shape[0], 256, 40, 40)
        )
        feature_8 = (
            hidden_states[:, 2000:8400]
            .transpose(1, 2)
            .reshape(hidden_states.shape[0], 256, 80, 80)
        )
        return (
            feature_32,
            feature_16,
            feature_8,
            self.lateral_conv(feature_4),
        )


class FixedBilinearUpsample2xCore(nn.Module):
    def forward(self, feature_8: Tensor) -> Tensor:
        return fixed_bilinear_upsample_2x(feature_8)


class PixelDecoderOutputFinishCore(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.output_conv = decoder.output_convs[0]
        self.mask_projection = decoder.mask_projection

    def forward(
        self,
        lateral: Tensor,
        upsampled_feature_8: Tensor,
    ) -> Tensor:
        feature_4_output = self.output_conv(
            lateral + upsampled_feature_8
        )
        return self.mask_projection(feature_4_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pixel-decoder-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--device-id", type=int, default=0)
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
    return [
        tensor_metrics(actual_value, expected_value)
        for actual_value, expected_value in zip(actual, expected)
    ]


def trace_component(
    name: str,
    module: nn.Module,
    example_inputs: tuple[Tensor, ...],
    output_dir: Path,
    compiler_args: str,
):
    artifact_path = output_dir / f"{name}.pt"
    start = time.perf_counter()
    if artifact_path.exists():
        compiled = torch.jit.load(str(artifact_path))
        compile_seconds = 0.0
        skipped_existing = True
    else:
        compiled = torch_neuronx.trace(
            module,
            example_inputs,
            compiler_args=compiler_args,
            compiler_workdir=str(output_dir / f"{name}_workdir"),
        )
        compile_seconds = time.perf_counter() - start
        torch.jit.save(compiled, str(artifact_path))
        skipped_existing = False
    return compiled, {
        "name": name,
        "compile_seconds": compile_seconds,
        "skipped_existing": skipped_existing,
        "artifact_bytes": artifact_path.stat().st_size,
    }


def place_on_device(module, device_id: int) -> None:
    torch_neuronx.move_trace_to_device(module, device_id)
    torch_neuronx.set_neuron_cores(
        module,
        start_nc=device_id,
        nc_count=1,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    pixel_dir = Path(args.pixel_decoder_dir)
    compiled_input = torch.jit.load(str(pixel_dir / "input.pt"))
    compiled_layers = [
        torch.jit.load(str(pixel_dir / f"layer_{index}.pt"))
        for index in range(6)
    ]
    old_output = torch.jit.load(str(pixel_dir / "output.pt"))

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
        for layer in compiled_layers:
            hidden = layer(hidden, position_embeddings)
        hidden = hidden.cpu()
        feature_4 = backbone_outputs[0]

        native_module = PixelDecoderOutputCore(decoder).eval()
        fixed_module = PixelDecoderOutputFixedCore(decoder).eval()
        prepare_module = PixelDecoderOutputPrepareCore(decoder).eval()
        upsample_module = FixedBilinearUpsample2xCore().eval()
        finish_module = PixelDecoderOutputFinishCore(decoder).eval()

        native_cpu_output = native_module(hidden, feature_4)
        fixed_cpu_output = fixed_module(hidden, feature_4)
        prepared_cpu = prepare_module(hidden, feature_4)
        upsampled_cpu = upsample_module(prepared_cpu[2])
        split_cpu_output = (
            prepared_cpu[0],
            prepared_cpu[1],
            prepared_cpu[2],
            finish_module(prepared_cpu[3], upsampled_cpu),
        )

    random_input = torch.randn(1, 3, 5, 7)
    fixed_random = fixed_bilinear_upsample_2x(random_input)
    native_random = F.interpolate(
        random_input,
        scale_factor=2.0,
        mode="bilinear",
        align_corners=False,
    )

    compile_results = []
    native_new, result = trace_component(
        "native_output_new",
        native_module,
        (hidden, feature_4),
        output_dir,
        compiler_args,
    )
    compile_results.append(result)
    fixed_output, result = trace_component(
        "fixed_output",
        fixed_module,
        (hidden, feature_4),
        output_dir,
        compiler_args,
    )
    compile_results.append(result)
    prepare, result = trace_component(
        "prepare",
        prepare_module,
        (hidden, feature_4),
        output_dir,
        compiler_args,
    )
    compile_results.append(result)
    upsample, result = trace_component(
        "upsample",
        upsample_module,
        (prepared_cpu[2],),
        output_dir,
        compiler_args,
    )
    compile_results.append(result)
    finish, result = trace_component(
        "finish",
        finish_module,
        (prepared_cpu[3], upsampled_cpu),
        output_dir,
        compiler_args,
    )
    compile_results.append(result)
    gc.collect()

    all_modules = (
        old_output,
        native_new,
        fixed_output,
        prepare,
        upsample,
        finish,
    )
    for module in all_modules:
        place_on_device(module, args.device_id)

    private_backend = torch._C._get_privateuse1_backend_name()
    neuron_device = torch.device(
        f"{private_backend}:{args.device_id}"
    )
    hidden_device = hidden.to(neuron_device)
    feature_4_device = feature_4.to(neuron_device)

    with torch.no_grad():
        old_neuron_output = old_output(
            hidden_device,
            feature_4_device,
        )
        native_neuron_output = native_new(
            hidden_device,
            feature_4_device,
        )
        fixed_neuron_output = fixed_output(
            hidden_device,
            feature_4_device,
        )
        prepared_device = prepare(
            hidden_device,
            feature_4_device,
        )
        upsampled_device = upsample(prepared_device[2])
        split_neuron_output = (
            prepared_device[0],
            prepared_device[1],
            prepared_device[2],
            finish(prepared_device[3], upsampled_device),
        )

        def split_pipeline():
            values = prepare(hidden_device, feature_4_device)
            upsampled = upsample(values[2])
            mask = finish(values[3], upsampled)
            return values[0], values[1], values[2], mask

        latency_ms = {
            "old_output": benchmark_callable(
                lambda: old_output(hidden_device, feature_4_device),
                args.warmup,
                args.runs,
            ),
            "native_output_new": benchmark_callable(
                lambda: native_new(hidden_device, feature_4_device),
                args.warmup,
                args.runs,
            ),
            "fixed_output": benchmark_callable(
                lambda: fixed_output(hidden_device, feature_4_device),
                args.warmup,
                args.runs,
            ),
            "split_output": benchmark_callable(
                split_pipeline,
                args.warmup,
                args.runs,
            ),
            "prepare": benchmark_callable(
                lambda: prepare(hidden_device, feature_4_device),
                args.warmup,
                args.runs,
            ),
            "upsample": benchmark_callable(
                lambda: upsample(prepared_device[2]),
                args.warmup,
                args.runs,
            ),
            "finish": benchmark_callable(
                lambda: finish(
                    prepared_device[3],
                    upsampled_device,
                ),
                args.warmup,
                args.runs,
            ),
        }

    report = {
        "compiler_args": compiler_args,
        "device": str(neuron_device),
        "fixed_upsample_cpu_vs_interpolate": tensor_metrics(
            fixed_random,
            native_random,
        ),
        "fixed_cpu_output_vs_native": metrics_tree(
            fixed_cpu_output,
            native_cpu_output,
        ),
        "split_cpu_output_vs_native": metrics_tree(
            split_cpu_output,
            native_cpu_output,
        ),
        "old_neuron_output_vs_native_cpu": metrics_tree(
            tuple(value.cpu() for value in old_neuron_output),
            native_cpu_output,
        ),
        "native_new_neuron_output_vs_native_cpu": metrics_tree(
            tuple(value.cpu() for value in native_neuron_output),
            native_cpu_output,
        ),
        "fixed_neuron_output_vs_native_cpu": metrics_tree(
            tuple(value.cpu() for value in fixed_neuron_output),
            native_cpu_output,
        ),
        "split_neuron_output_vs_native_cpu": metrics_tree(
            tuple(value.cpu() for value in split_neuron_output),
            native_cpu_output,
        ),
        "compile_results": compile_results,
        "latency_ms": latency_ms,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
