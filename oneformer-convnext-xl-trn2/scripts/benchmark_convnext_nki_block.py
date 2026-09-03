#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch_neuronx
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neuron_port.convnext_nki import (
    NkiConvNextStage0DepthwiseCore,
    NkiConvNextStage0FusedLayerCore,
    NkiConvNextStage0LayerCore,
    NkiConvNextStage0PostDwCore,
)
from neuron_port.modeling import load_oneformer
from scripts.run_validation import percentile, tensor_metrics


class NativeConvNextPostDwCore(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layernorm = layer.layernorm
        self.pwconv1 = layer.pwconv1
        self.act = layer.act
        self.pwconv2 = layer.pwconv2
        self.layer_scale_parameter = layer.layer_scale_parameter
        self.drop_path = layer.drop_path

    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor,
    ) -> Tensor:
        hidden_states = hidden_states.permute(0, 2, 3, 1)
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        if self.layer_scale_parameter is not None:
            hidden_states = self.layer_scale_parameter * hidden_states
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        return residual + self.drop_path(hidden_states)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--custom-grid-sample", action="store_true")
    return parser.parse_args()


def synchronize_output(output: Tensor) -> None:
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
    module: nn.Module,
    example_inputs: tuple[Tensor, ...],
    output_dir: Path,
) -> tuple[torch.jit.ScriptModule, dict]:
    artifact_path = output_dir / f"{name}.pt"
    start = time.perf_counter()
    compiled = torch_neuronx.trace(
        module,
        example_inputs,
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
    layer = backbone.encoder.stages[0].layers[0].eval()
    native_post = NativeConvNextPostDwCore(layer).eval()
    nki_depthwise = NkiConvNextStage0DepthwiseCore(layer, lnc=2).eval()
    nki_post = NkiConvNextStage0PostDwCore(layer, lnc=2).eval()
    nki_fused = NkiConvNextStage0FusedLayerCore(layer, lnc=2).eval()
    nki_full = NkiConvNextStage0LayerCore(layer, lnc=2).eval()

    with torch.no_grad():
        layer_input = backbone.embeddings(pixel_values)
        depthwise_output = layer.dwconv(layer_input)
        expected_post = native_post(depthwise_output, layer_input)
        expected_full = layer(layer_input)

    compiled_nki_fused, nki_fused_compile = trace_component(
        "nki_fused_full_layer",
        nki_fused,
        (layer_input,),
        output_dir,
    )
    compiled_nki_depthwise, nki_depthwise_compile = trace_component(
        "nki_depthwise",
        nki_depthwise,
        (layer_input,),
        output_dir,
    )
    compiled_native_depthwise, native_depthwise_compile = trace_component(
        "native_depthwise",
        layer.dwconv,
        (layer_input,),
        output_dir,
    )
    compiled_native_post, native_post_compile = trace_component(
        "native_post_dw",
        native_post,
        (depthwise_output, layer_input),
        output_dir,
    )
    compiled_nki_post, nki_post_compile = trace_component(
        "nki_post_dw",
        nki_post,
        (depthwise_output, layer_input),
        output_dir,
    )
    compiled_native_full, native_full_compile = trace_component(
        "native_full_layer",
        layer,
        (layer_input,),
        output_dir,
    )
    compiled_nki_full, nki_full_compile = trace_component(
        "nki_hybrid_full_layer",
        nki_full,
        (layer_input,),
        output_dir,
    )

    device = torch.device(f"privateuseone:{args.device_id}")
    for module in (
        compiled_nki_fused,
        compiled_nki_depthwise,
        compiled_native_depthwise,
        compiled_native_post,
        compiled_nki_post,
        compiled_native_full,
        compiled_nki_full,
    ):
        place_on_device(module, args.device_id)

    neuron_depthwise = depthwise_output.to(device)
    neuron_layer_input = layer_input.to(device)
    with torch.no_grad():
        nki_fused_output = compiled_nki_fused(
            neuron_layer_input,
        ).cpu()
        nki_depthwise_output = compiled_nki_depthwise(
            neuron_layer_input,
        ).cpu()
        native_depthwise_output = compiled_native_depthwise(
            neuron_layer_input,
        ).cpu()
        native_post_output = compiled_native_post(
            neuron_depthwise,
            neuron_layer_input,
        ).cpu()
        nki_post_output = compiled_nki_post(
            neuron_depthwise,
            neuron_layer_input,
        ).cpu()
        native_full_output = compiled_native_full(
            neuron_layer_input,
        ).cpu()
        nki_full_output = compiled_nki_full(
            neuron_layer_input,
        ).cpu()

        latency = {
            "nki_fused_full_layer": benchmark_callable(
                lambda: compiled_nki_fused(neuron_layer_input),
                args.warmup,
                args.runs,
            ),
            "nki_depthwise": benchmark_callable(
                lambda: compiled_nki_depthwise(neuron_layer_input),
                args.warmup,
                args.runs,
            ),
            "native_depthwise": benchmark_callable(
                lambda: compiled_native_depthwise(neuron_layer_input),
                args.warmup,
                args.runs,
            ),
            "native_post_dw": benchmark_callable(
                lambda: compiled_native_post(
                    neuron_depthwise,
                    neuron_layer_input,
                ),
                args.warmup,
                args.runs,
            ),
            "nki_post_dw": benchmark_callable(
                lambda: compiled_nki_post(
                    neuron_depthwise,
                    neuron_layer_input,
                ),
                args.warmup,
                args.runs,
            ),
            "native_full_layer": benchmark_callable(
                lambda: compiled_native_full(neuron_layer_input),
                args.warmup,
                args.runs,
            ),
            "nki_hybrid_full_layer": benchmark_callable(
                lambda: compiled_nki_full(neuron_layer_input),
                args.warmup,
                args.runs,
            ),
        }

    report = {
        "model_id": args.model_id,
        "shape": list(layer_input.shape),
        "precision": "bf16-all",
        "lnc": 2,
        "compile": [
            nki_fused_compile,
            nki_depthwise_compile,
            native_depthwise_compile,
            native_post_compile,
            nki_post_compile,
            native_full_compile,
            nki_full_compile,
        ],
        "metrics": {
            "nki_fused_full_vs_cpu": tensor_metrics(
                nki_fused_output,
                expected_full,
            ),
            "nki_fused_full_vs_native_neuron": tensor_metrics(
                nki_fused_output,
                native_full_output,
            ),
            "nki_depthwise_vs_cpu": tensor_metrics(
                nki_depthwise_output,
                depthwise_output,
            ),
            "native_depthwise_vs_cpu": tensor_metrics(
                native_depthwise_output,
                depthwise_output,
            ),
            "nki_depthwise_vs_native_neuron": tensor_metrics(
                nki_depthwise_output,
                native_depthwise_output,
            ),
            "native_post_vs_cpu": tensor_metrics(
                native_post_output,
                expected_post,
            ),
            "nki_post_vs_cpu": tensor_metrics(
                nki_post_output,
                expected_post,
            ),
            "native_full_vs_cpu": tensor_metrics(
                native_full_output,
                expected_full,
            ),
            "nki_hybrid_full_vs_cpu": tensor_metrics(
                nki_full_output,
                expected_full,
            ),
            "nki_hybrid_full_vs_native_neuron": tensor_metrics(
                nki_full_output,
                native_full_output,
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
