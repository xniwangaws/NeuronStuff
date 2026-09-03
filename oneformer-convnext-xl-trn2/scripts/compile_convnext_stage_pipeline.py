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

from neuron_port.modeling import load_oneformer


class ConvNextEmbeddingCore(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.embeddings = backbone.embeddings

    def forward(self, pixel_values: Tensor) -> Tensor:
        return self.embeddings(pixel_values)


class ConvNextStageChunkCore(nn.Module):
    def __init__(
        self,
        stage: nn.Module,
        start: int,
        end: int,
        output_norm: nn.Module | None,
    ):
        super().__init__()
        self.downsampling = nn.ModuleList(
            list(stage.downsampling_layer) if start == 0 else []
        )
        self.layers = nn.ModuleList(list(stage.layers[start:end]))
        self.output_norm = output_norm

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
        "--precision",
        choices=("fp32", "bf16", "bf16-all"),
        default="bf16-all",
    )
    parser.add_argument("--opt-level", choices=("1", "2"), default="1")
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
    embedding_core = ConvNextEmbeddingCore(backbone).eval()

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
            chunk = ConvNextStageChunkCore(
                stage,
                start,
                end,
                norm if is_final else None,
            ).eval()
            chunk_input = raw_features
            with torch.no_grad():
                chunk_output = chunk(chunk_input)
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
                    "input": chunk_input,
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
        "chunk_size": args.chunk_size,
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
