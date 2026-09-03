#!/usr/bin/env python3

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch_neuronx
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neuron_port.convnext_nki import NkiConvNextStage0FusedLayerCore
from neuron_port.convnext_stage1_nki import (
    NkiConvNextStage1FusedLayerCore,
)
from neuron_port.convnext_stage2_nki import (
    NkiConvNextStage2FusedLayerCore,
)
from neuron_port.modeling import load_oneformer


class ConvNextEmbeddingCore(nn.Module):
    def __init__(self, backbone: nn.Module, layout: str):
        super().__init__()
        embeddings = backbone.embeddings
        self.patch_embeddings = embeddings.patch_embeddings
        self.layernorm = (
            ChannelsFirstLayerNorm(embeddings.layernorm)
            if layout == "channels-first"
            else embeddings.layernorm
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        return self.layernorm(self.patch_embeddings(pixel_values))


class ChannelsFirstLayerNorm(nn.Module):
    """LayerNorm over C without NCHW-to-NHWC permutations."""

    def __init__(self, layernorm: nn.LayerNorm):
        super().__init__()
        self.weight = nn.Parameter(
            layernorm.weight.detach().clone(),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            layernorm.bias.detach().clone(),
            requires_grad=False,
        )
        self.eps = layernorm.eps

    def forward(self, features: Tensor) -> Tensor:
        mean = features.mean(dim=1, keepdim=True)
        centered = features - mean
        variance = (centered * centered).mean(dim=1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        return (
            normalized * self.weight.view(1, -1, 1, 1)
            + self.bias.view(1, -1, 1, 1)
        )


def linear_as_conv2d(linear: nn.Linear) -> nn.Conv2d:
    convolution = nn.Conv2d(
        linear.in_features,
        linear.out_features,
        kernel_size=1,
        bias=linear.bias is not None,
    )
    convolution.weight = nn.Parameter(
        linear.weight.detach().reshape(
            linear.out_features,
            linear.in_features,
            1,
            1,
        ).clone(),
        requires_grad=False,
    )
    if linear.bias is not None:
        convolution.bias = nn.Parameter(
            linear.bias.detach().clone(),
            requires_grad=False,
        )
    return convolution


class ChannelsFirstConvNextLayer(nn.Module):
    """The mathematically equivalent all-NCHW ConvNeXt block."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.dwconv = layer.dwconv
        self.layernorm = ChannelsFirstLayerNorm(layer.layernorm)
        self.pwconv1 = linear_as_conv2d(layer.pwconv1)
        self.act = layer.act
        self.pwconv2 = linear_as_conv2d(layer.pwconv2)
        self.layer_scale_parameter = (
            nn.Parameter(
                layer.layer_scale_parameter.detach().clone(),
                requires_grad=False,
            )
            if layer.layer_scale_parameter is not None
            else None
        )
        self.drop_path = layer.drop_path

    def forward(self, features: Tensor) -> Tensor:
        residual = features
        features = self.dwconv(features)
        features = self.layernorm(features)
        features = self.pwconv1(features)
        features = self.act(features)
        features = self.pwconv2(features)
        if self.layer_scale_parameter is not None:
            features = (
                self.layer_scale_parameter.view(1, -1, 1, 1) * features
            )
        return residual + self.drop_path(features)


class ConvNextStageChunkCore(nn.Module):
    def __init__(
        self,
        stage: nn.Module,
        start: int,
        end: int,
        output_norm: nn.Module | None,
        layout: str,
        use_stage0_nki: bool = False,
        use_stage1_nki: bool = False,
        use_stage2_nki: bool = False,
    ):
        super().__init__()
        downsampling = list(stage.downsampling_layer) if start == 0 else []
        if layout == "channels-first" and downsampling:
            downsampling[0] = ChannelsFirstLayerNorm(downsampling[0])
        self.downsampling = nn.ModuleList(downsampling)
        layers = list(stage.layers[start:end])
        if use_stage0_nki:
            layers = [
                NkiConvNextStage0FusedLayerCore(layer, lnc=2)
                for layer in layers
            ]
        elif use_stage1_nki:
            layers = [
                NkiConvNextStage1FusedLayerCore(layer, lnc=2)
                for layer in layers
            ]
        elif use_stage2_nki:
            layers = [
                NkiConvNextStage2FusedLayerCore(layer, lnc=2)
                for layer in layers
            ]
        elif layout == "channels-first":
            layers = [ChannelsFirstConvNextLayer(layer) for layer in layers]
        self.layers = nn.ModuleList(layers)
        self.output_norm = (
            ChannelsFirstLayerNorm(output_norm)
            if layout == "channels-first" and output_norm is not None
            else output_norm
        )

    def forward(self, features: Tensor):
        for layer in self.downsampling:
            features = layer(features)
        for layer in self.layers:
            features = layer(features)
        if self.output_norm is None:
            return features
        return features, self.output_norm(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument(
        "--layout",
        choices=("native", "channels-first"),
        default="native",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "bf16-all"),
        default="bf16-all",
    )
    parser.add_argument("--opt-level", choices=("1", "2"), default="1")
    parser.add_argument(
        "--stage0-nki",
        action="store_true",
        help="Replace all stage-0 ConvNeXt blocks with fused LNC2 NKI kernels.",
    )
    parser.add_argument(
        "--stage1-nki",
        action="store_true",
        help="Replace all stage-1 ConvNeXt blocks with fused LNC2 NKI kernels.",
    )
    parser.add_argument(
        "--stage2-nki",
        action="store_true",
        help="Replace all stage-2 ConvNeXt blocks with fused LNC2 NKI kernels.",
    )
    parser.add_argument("--custom-grid-sample", action="store_true")
    return parser.parse_args()


def compiler_args(args: argparse.Namespace) -> str:
    base = f"--model-type=unet-inference -O{args.opt_level}"
    if args.precision == "bf16-all":
        return f"{base} --auto-cast=all --auto-cast-type=bf16"
    if args.precision == "bf16":
        return f"{base} --auto-cast=matmult --auto-cast-type=bf16"
    return f"{base} --auto-cast=none"


def shape_tree(output) -> list[list[int]]:
    if isinstance(output, (tuple, list)):
        return [list(value.shape) for value in output]
    return [list(output.shape)]


def tensor_metrics(actual: Tensor, expected: Tensor) -> dict[str, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    difference = (actual_float - expected_float).abs()
    return {
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "cosine_similarity": nn.functional.cosine_similarity(
            actual_float.flatten(),
            expected_float.flatten(),
            dim=0,
        ).item(),
    }


def write_progress(
    progress_path: Path,
    current: str | None,
    completed: list[dict],
) -> None:
    progress_path.write_text(
        json.dumps(
            {
                "current": current,
                "completed": completed,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"

    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    pixel_values = inputs["pixel_values"].contiguous()
    _, wrapper = load_oneformer(
        args.model_id,
        cache_dir=args.cache_dir,
        use_custom_grid_sample=args.custom_grid_sample,
        local_files_only=True,
    )
    backbone = wrapper.core.pixel_level_module.encoder
    embedding_core = ConvNextEmbeddingCore(backbone, args.layout).eval()

    with torch.no_grad():
        embedding_output = embedding_core(pixel_values)
        reference_features = backbone(pixel_values).feature_maps

    compile_specs = [
        {
            "name": "embeddings",
            "module": embedding_core,
            "input": pixel_values,
            "cpu_output": embedding_output,
        }
    ]
    raw_features = embedding_output
    manual_features = []
    stage_names = list(backbone.stage_names[1:])
    if len(stage_names) != len(backbone.encoder.stages):
        raise ValueError(
            "ConvNeXt stage-name count does not match encoder stages"
        )

    for stage_index, (stage_name, stage) in enumerate(
        zip(stage_names, backbone.encoder.stages)
    ):
        layer_count = len(stage.layers)
        norm = (
            backbone.hidden_states_norms[stage_name]
            if stage_name in backbone.hidden_states_norms
            else None
        )
        for chunk_index, start in enumerate(
            range(0, layer_count, args.chunk_size)
        ):
            end = min(start + args.chunk_size, layer_count)
            is_final = end == layer_count
            cpu_chunk = ConvNextStageChunkCore(
                stage,
                start,
                end,
                norm if is_final else None,
                args.layout,
            ).eval()
            chunk = ConvNextStageChunkCore(
                stage,
                start,
                end,
                norm if is_final else None,
                args.layout,
                use_stage0_nki=(
                    args.stage0_nki and stage_index == 0
                ),
                use_stage1_nki=(
                    args.stage1_nki and stage_index == 1
                ),
                use_stage2_nki=(
                    args.stage2_nki and stage_index == 2
                ),
            ).eval()
            chunk_input = raw_features
            compile_input = chunk_input
            with torch.no_grad():
                chunk_output = cpu_chunk(chunk_input)
            if is_final:
                raw_features, normalized_features = chunk_output
                if norm is not None:
                    manual_features.append(normalized_features)
            else:
                raw_features = chunk_output
            compile_specs.append(
                {
                    "name": (
                        f"stage{stage_index}_chunk{chunk_index}_"
                        f"layers{start}_{end}"
                    ),
                    "module": chunk,
                    "input": compile_input,
                    "cpu_output": chunk_output,
                }
            )

    if len(manual_features) != len(reference_features):
        raise ValueError(
            "Manual ConvNeXt stage pipeline did not reproduce all features"
        )
    manual_vs_backbone = {
        stage_name: tensor_metrics(actual, expected)
        for stage_name, actual, expected in zip(
            list(backbone.out_features),
            manual_features,
            reference_features,
        )
    }

    completed = []
    compile_flags = compiler_args(args)
    for spec in compile_specs:
        name = spec["name"]
        output_path = output_dir / f"{name}.pt"
        workdir = output_dir / f"{name}_workdir"
        write_progress(progress_path, name, completed)
        print(
            json.dumps(
                {
                    "compiling": name,
                    "input_shape": list(spec["input"].shape),
                    "output_shapes": shape_tree(spec["cpu_output"]),
                    "compiler_args": compile_flags,
                }
            ),
            flush=True,
        )
        if output_path.exists():
            completed.append(
                {
                    "name": name,
                    "skipped_existing": True,
                    "artifact_bytes": output_path.stat().st_size,
                }
            )
            continue

        start = time.perf_counter()
        traced = torch_neuronx.trace(
            spec["module"],
            (spec["input"],),
            compiler_args=compile_flags,
            compiler_workdir=str(workdir),
        )
        torch.jit.save(traced, str(output_path))
        elapsed = time.perf_counter() - start
        completed.append(
            {
                "name": name,
                "compile_seconds": elapsed,
                "artifact_bytes": output_path.stat().st_size,
                "input_shape": list(spec["input"].shape),
                "output_shapes": shape_tree(spec["cpu_output"]),
            }
        )
        del traced
        gc.collect()
        write_progress(progress_path, None, completed)

    metadata = {
        "model_id": args.model_id,
        "precision": args.precision,
        "layout": args.layout,
        "chunk_size": args.chunk_size,
        "stage0_nki": args.stage0_nki,
        "stage1_nki": args.stage1_nki,
        "stage2_nki": args.stage2_nki,
        "compiler_args": compile_flags,
        "manual_vs_backbone": manual_vs_backbone,
        "components": completed,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    write_progress(progress_path, None, completed)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
