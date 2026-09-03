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
from scripts.compile_transformer_pipeline import AttentionMaskCore
try:
    from scripts.run_full_oneformer_pipeline import (
        CompiledNkiPixelDecoderPipeline,
        CompiledTransformerPipeline,
        build_pixel_position_embeddings,
    )
except ImportError:
    from scripts.run_full_oneformer_hbm_pipeline import (
        CompiledNkiPixelDecoderPipeline,
        CompiledTransformerPipeline,
        build_pixel_position_embeddings,
    )
from scripts.run_validation import percentile, tensor_metrics


def fixed_bilinear_downsample(
    input_tensor: Tensor,
    factor: int,
) -> Tensor:
    """Exact align_corners=False bilinear downsample for even factors."""

    offset = factor // 2 - 1
    row0 = offset
    row1 = offset + 1
    col0 = offset
    col1 = offset + 1
    return 0.25 * (
        input_tensor[:, :, row0::factor, col0::factor]
        + input_tensor[:, :, row0::factor, col1::factor]
        + input_tensor[:, :, row1::factor, col0::factor]
        + input_tensor[:, :, row1::factor, col1::factor]
    )


class FixedAttentionMaskCore(nn.Module):
    def __init__(
        self,
        decoder: nn.Module,
        factor: int,
    ):
        super().__init__()
        self.decoder_norm = decoder.decoder_norm
        self.mask_embed = decoder.mask_embed
        self.num_heads = decoder.num_heads
        self.factor = factor

    def forward(
        self,
        output: Tensor,
        mask_features: Tensor,
    ) -> Tensor:
        decoder_output = self.decoder_norm(output).transpose(0, 1)
        mask_embedding = self.mask_embed(decoder_output)
        masks = torch.einsum(
            "bqc,bchw->bqhw",
            mask_embedding,
            mask_features,
        )
        attention_mask = fixed_bilinear_downsample(
            masks,
            self.factor,
        )
        return (
            attention_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pixel-decoder-dir", required=True)
    parser.add_argument("--remaining-dir", required=True)
    parser.add_argument("--transformer-dir", required=True)
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


def bool_metrics(actual: Tensor, expected: Tensor) -> dict[str, float]:
    return {
        "agreement": (actual == expected).float().mean().item(),
        "different_elements": int((actual != expected).sum().item()),
        "total_elements": actual.numel(),
    }


def trace_component(
    name: str,
    module: nn.Module,
    example_inputs: tuple[Tensor, Tensor],
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
        "--model-type=transformer -O1 "
        "--auto-cast=all --auto-cast-type=bf16"
    )

    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    pixel_values = inputs["pixel_values"].contiguous()
    task_inputs = inputs["task_inputs"].contiguous()
    _, wrapper = load_oneformer(
        args.model_id,
        cache_dir=args.cache_dir,
        use_custom_grid_sample=args.custom_grid_sample,
        local_files_only=True,
    )
    backbone = wrapper.core.pixel_level_module.encoder.eval()
    decoder = wrapper.core.transformer_module.decoder

    compiled_pixel = CompiledNkiPixelDecoderPipeline(
        Path(args.pixel_decoder_dir)
    )
    compiled_task = torch.jit.load(
        str(Path(args.remaining_dir) / "task_encoder.pt")
    )
    compiled_transformer = CompiledTransformerPipeline(
        Path(args.transformer_dir)
    )

    with torch.no_grad():
        backbone_outputs = tuple(backbone(pixel_values).feature_maps)
        position_embeddings = build_pixel_position_embeddings(
            wrapper.core.pixel_level_module.decoder,
            backbone_outputs,
        )
        pixel_outputs = compiled_pixel(
            backbone_outputs,
            position_embeddings,
        )
        task_token = compiled_task(task_inputs)
        mask_features = pixel_outputs[3]
        query_output, mask_position, task_sequence = (
            compiled_transformer.query_input(
                mask_features,
                task_token,
            )
        )
        for query_layer in compiled_transformer.query_layers:
            query_output = query_layer(
                query_output,
                mask_position,
            )
        (
            output,
            _,
            _,
            _,
        ) = compiled_transformer.main_prepare(
            query_output,
            task_sequence,
            pixel_outputs[0],
            pixel_outputs[1],
            pixel_outputs[2],
        )
        output = output.cpu()
        mask_features = mask_features.cpu()

    target_sizes = (20, 40, 80)
    factors = (8, 4, 2)
    cpu_native = []
    cpu_fixed = []
    fixed_modules = []
    compile_results = []
    compiled_fixed = []
    for index, (target, factor) in enumerate(
        zip(target_sizes, factors)
    ):
        native_module = AttentionMaskCore(
            decoder,
            target,
            target,
        ).eval()
        fixed_module = FixedAttentionMaskCore(
            decoder,
            factor,
        ).eval()
        with torch.no_grad():
            cpu_native.append(native_module(output, mask_features))
            cpu_fixed.append(fixed_module(output, mask_features))
        fixed_modules.append(fixed_module)
        compiled, result = trace_component(
            f"mask_{index}",
            fixed_module,
            (output, mask_features),
            output_dir,
            compiler_args,
        )
        compiled_fixed.append(compiled)
        compile_results.append(result)
        gc.collect()

    random_masks = torch.randn(1, 3, 160, 160)
    resize_equivalence = {}
    for target, factor in zip(target_sizes, factors):
        fixed = fixed_bilinear_downsample(random_masks, factor)
        native = F.interpolate(
            random_masks,
            size=(target, target),
            mode="bilinear",
            align_corners=False,
        )
        resize_equivalence[str(target)] = tensor_metrics(
            fixed,
            native,
        )

    old_masks = [
        torch.jit.load(
            str(Path(args.transformer_dir) / f"mask_{index}.pt")
        )
        for index in range(3)
    ]
    for module in (*old_masks, *compiled_fixed):
        place_on_device(module, args.device_id)

    private_backend = torch._C._get_privateuse1_backend_name()
    neuron_device = torch.device(
        f"{private_backend}:{args.device_id}"
    )
    output_device = output.to(neuron_device)
    mask_features_device = mask_features.to(neuron_device)

    old_outputs = []
    fixed_outputs = []
    latency_ms = {}
    with torch.no_grad():
        for index, (old_mask, fixed_mask) in enumerate(
            zip(old_masks, compiled_fixed)
        ):
            old_output = old_mask(
                output_device,
                mask_features_device,
            )
            fixed_output = fixed_mask(
                output_device,
                mask_features_device,
            )
            old_outputs.append(old_output.cpu())
            fixed_outputs.append(fixed_output.cpu())
            latency_ms[f"mask_{index}_old"] = benchmark_callable(
                lambda module=old_mask: module(
                    output_device,
                    mask_features_device,
                ),
                args.warmup,
                args.runs,
            )
            latency_ms[f"mask_{index}_fixed"] = benchmark_callable(
                lambda module=fixed_mask: module(
                    output_device,
                    mask_features_device,
                ),
                args.warmup,
                args.runs,
            )

    report = {
        "compiler_args": compiler_args,
        "device": str(neuron_device),
        "resize_equivalence": resize_equivalence,
        "cpu_fixed_vs_native": {
            str(target): bool_metrics(fixed, native)
            for target, fixed, native in zip(
                target_sizes,
                cpu_fixed,
                cpu_native,
            )
        },
        "old_neuron_vs_native_cpu": {
            str(target): bool_metrics(actual, expected)
            for target, actual, expected in zip(
                target_sizes,
                old_outputs,
                cpu_native,
            )
        },
        "fixed_neuron_vs_native_cpu": {
            str(target): bool_metrics(actual, expected)
            for target, actual, expected in zip(
                target_sizes,
                fixed_outputs,
                cpu_native,
            )
        },
        "compile_results": compile_results,
        "latency_ms": latency_ms,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
