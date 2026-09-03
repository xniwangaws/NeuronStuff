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

from neuron_port.components import PixelDecoderCore
from neuron_port.modeling import load_oneformer
from neuron_port.ops import bilinear_grid_sample_2d
from scripts.compile_pixel_decoder_pipeline import PixelDecoderOutputCore
from scripts.run_validation import tensor_metrics


class PixelDecoderInputCore(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.input_projections = decoder.input_projections

    def forward(
        self,
        feature_8: Tensor,
        feature_16: Tensor,
        feature_32: Tensor,
    ) -> Tensor:
        sources = (
            self.input_projections[0](feature_32),
            self.input_projections[1](feature_16),
            self.input_projections[2](feature_8),
        )
        return torch.cat(
            [source.flatten(2).transpose(1, 2) for source in sources],
            dim=1,
        )


class AttentionProjectionCore(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        reference_points: Tensor,
    ):
        super().__init__()
        self.value_proj = attention.value_proj
        self.sampling_offsets = attention.sampling_offsets
        self.attention_weights = attention.attention_weights
        self.num_heads = attention.n_heads
        self.num_levels = attention.n_levels
        self.num_points = attention.n_points
        self.head_dim = attention.d_model // attention.n_heads
        self.register_buffer("reference_points", reference_points)
        self.register_buffer(
            "offset_normalizer",
            torch.tensor(
                [[20, 20], [40, 40], [80, 80]],
                dtype=reference_points.dtype,
            ),
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        query = hidden_states + position_embeddings
        batch_size, sequence_length, _ = hidden_states.shape
        value = self.value_proj(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )
        sampling_offsets = self.sampling_offsets(query).reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        attention_weights = self.attention_weights(query).reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.num_levels * self.num_points,
        )
        attention_weights = torch.softmax(
            attention_weights,
            dim=-1,
        ).reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )
        sampling_locations = (
            self.reference_points[:, :, None, :, None, :]
            + sampling_offsets
            / self.offset_normalizer[None, None, None, :, None, :]
        )
        return value, sampling_locations, attention_weights


class AttentionSamplerCore(nn.Module):
    def __init__(
        self,
        level: int,
        start: int,
        end: int,
        height: int,
        width: int,
        implementation: str,
    ):
        super().__init__()
        self.level = level
        self.start = start
        self.end = end
        self.height = height
        self.width = width
        self.implementation = implementation

    def forward(
        self,
        value: Tensor,
        sampling_locations: Tensor,
    ) -> Tensor:
        batch_size = value.shape[0]
        num_heads = value.shape[2]
        head_dim = value.shape[3]
        value_level = (
            value[:, self.start : self.end]
            .flatten(2)
            .transpose(1, 2)
            .reshape(
                batch_size * num_heads,
                head_dim,
                self.height,
                self.width,
            )
        )
        sampling_grid = (
            (2.0 * sampling_locations[:, :, :, self.level] - 1.0)
            .transpose(1, 2)
            .flatten(0, 1)
        )
        if self.implementation == "raw":
            return F.grid_sample(
                value_level,
                sampling_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        return bilinear_grid_sample_2d(value_level, sampling_grid)


class AttentionCombineCore(nn.Module):
    def __init__(self, attention: nn.Module):
        super().__init__()
        self.output_proj = attention.output_proj
        self.num_heads = attention.n_heads
        self.head_dim = attention.d_model // attention.n_heads

    def forward(
        self,
        sampled_20: Tensor,
        sampled_40: Tensor,
        sampled_80: Tensor,
        attention_weights: Tensor,
    ) -> Tensor:
        batch_size = attention_weights.shape[0]
        num_queries = attention_weights.shape[1]
        weights = attention_weights.transpose(1, 2).reshape(
            batch_size * self.num_heads,
            1,
            num_queries,
            12,
        )
        sampled = torch.stack(
            (sampled_20, sampled_40, sampled_80),
            dim=-2,
        ).flatten(-2)
        output = (
            (sampled * weights)
            .sum(-1)
            .reshape(
                batch_size,
                self.num_heads * self.head_dim,
                num_queries,
            )
            .transpose(1, 2)
            .contiguous()
        )
        return self.output_proj(output)


class EncoderLayerPostCore(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.self_attn_layer_norm = layer.self_attn_layer_norm
        self.fc1 = layer.fc1
        self.fc2 = layer.fc2
        self.final_layer_norm = layer.final_layer_norm

    def forward(
        self,
        hidden_states: Tensor,
        attention_output: Tensor,
    ) -> Tensor:
        hidden_states = self.self_attn_layer_norm(
            hidden_states + attention_output
        )
        residual = hidden_states
        hidden_states = torch.relu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        return self.final_layer_norm(residual + hidden_states)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "bf16-all"),
        default="bf16-all",
    )
    parser.add_argument("--custom-grid-sample", action="store_true")
    parser.add_argument(
        "--sampler-implementation",
        choices=("custom", "raw"),
        default="custom",
    )
    return parser.parse_args()


def compiler_args(precision: str) -> str:
    base = "--model-type=unet-inference -O1"
    if precision == "bf16-all":
        return f"{base} --auto-cast=all --auto-cast-type=bf16"
    if precision == "bf16":
        return f"{base} --auto-cast=matmult --auto-cast-type=bf16"
    return f"{base} --auto-cast=none"


def shape_tree(output) -> list[list[int]]:
    if isinstance(output, (tuple, list)):
        return [list(value.shape) for value in output]
    return [list(output.shape)]


def metrics_tree(actual, expected):
    if isinstance(actual, (tuple, list)):
        return [
            tensor_metrics(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        ]
    return tensor_metrics(actual, expected)


def write_progress(
    path: Path,
    current: str | None,
    completed: list[dict],
) -> None:
    path.write_text(
        json.dumps(
            {"current": current, "completed": completed},
            indent=2,
        )
        + "\n"
    )


def trace_component(
    name: str,
    module: nn.Module,
    example_inputs: tuple[Tensor, ...],
    expected,
    output_dir: Path,
    flags: str,
) -> tuple[torch.jit.ScriptModule, dict]:
    artifact_path = output_dir / f"{name}.pt"
    result = {
        "name": name,
        "input_shapes": [list(value.shape) for value in example_inputs],
        "output_shapes": shape_tree(expected),
        "compiler_args": flags,
    }
    start = time.perf_counter()
    if artifact_path.exists():
        compiled = torch.jit.load(str(artifact_path))
        result["skipped_existing"] = True
    else:
        compiled = torch_neuronx.trace(
            module,
            example_inputs,
            compiler_args=flags,
            compiler_workdir=str(output_dir / f"{name}_workdir"),
        )
        torch.jit.save(compiled, str(artifact_path))
        result["compile_seconds"] = time.perf_counter() - start
    result["artifact_bytes"] = artifact_path.stat().st_size
    return compiled, result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    flags = compiler_args(args.precision)

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
        input_hidden = input_core(*input_args)

        source_features = (
            backbone_outputs[3],
            backbone_outputs[2],
            backbone_outputs[1],
        )
        position_embeddings = []
        for level, source in enumerate(source_features):
            position = decoder.position_embedding(
                source.shape,
                source.device,
                source.dtype,
            )
            position = position.flatten(2).transpose(1, 2)
            position_embeddings.append(
                position + decoder.level_embed[level].view(1, 1, -1)
            )
        position_embeddings = torch.cat(position_embeddings, dim=1)
        spatial_shapes = torch.tensor(
            [[20, 20], [40, 40], [80, 80]],
            dtype=torch.long,
        )
        valid_ratios = torch.ones(
            (input_hidden.shape[0], 3, 2),
            dtype=input_hidden.dtype,
        )
        reference_points = decoder.encoder.get_reference_points(
            spatial_shapes,
            valid_ratios,
            device=input_hidden.device,
        )

        samplers = [
            AttentionSamplerCore(
                0,
                0,
                400,
                20,
                20,
                args.sampler_implementation,
            ).eval(),
            AttentionSamplerCore(
                1,
                400,
                2000,
                40,
                40,
                args.sampler_implementation,
            ).eval(),
            AttentionSamplerCore(
                2,
                2000,
                8400,
                80,
                80,
                args.sampler_implementation,
            ).eval(),
        ]
        layer_cores = []
        cpu_hidden = input_hidden
        cpu_layer_records = []
        for layer in decoder.encoder.layers:
            projection = AttentionProjectionCore(
                layer.self_attn,
                reference_points,
            ).eval()
            combine = AttentionCombineCore(layer.self_attn).eval()
            post = EncoderLayerPostCore(layer).eval()
            value, locations, weights = projection(
                cpu_hidden,
                position_embeddings,
            )
            sampled = [
                sampler(value, locations)
                for sampler in samplers
            ]
            attention_output = combine(*sampled, weights)
            next_hidden = post(cpu_hidden, attention_output)
            layer_cores.append((projection, combine, post))
            cpu_layer_records.append(
                {
                    "input": cpu_hidden,
                    "value": value,
                    "locations": locations,
                    "weights": weights,
                    "sampled": sampled,
                    "attention_output": attention_output,
                    "output": next_hidden,
                }
            )
            cpu_hidden = next_hidden
        cpu_pipeline_outputs = output_core(
            cpu_hidden,
            backbone_outputs[0],
        )

    completed = []
    compiled = {}

    def compile_one(name, module, example_inputs, expected):
        write_progress(progress_path, name, completed)
        print(json.dumps({"compiling": name}), flush=True)
        compiled_module, result = trace_component(
            name,
            module,
            tuple(example_inputs),
            expected,
            output_dir,
            flags,
        )
        compiled[name] = compiled_module
        completed.append(result)
        write_progress(progress_path, None, completed)
        gc.collect()

    compile_one(
        "input",
        input_core,
        input_args,
        input_hidden,
    )
    first_record = cpu_layer_records[0]
    for level, sampler in enumerate(samplers):
        compile_one(
            f"sampler_{level}",
            sampler,
            (first_record["value"], first_record["locations"]),
            first_record["sampled"][level],
        )
    for index, ((projection, combine, post), record) in enumerate(
        zip(layer_cores, cpu_layer_records)
    ):
        compile_one(
            f"layer_{index}_projection",
            projection,
            (record["input"], position_embeddings),
            (
                record["value"],
                record["locations"],
                record["weights"],
            ),
        )
        compile_one(
            f"layer_{index}_combine",
            combine,
            (*record["sampled"], record["weights"]),
            record["attention_output"],
        )
        compile_one(
            f"layer_{index}_post",
            post,
            (record["input"], record["attention_output"]),
            record["output"],
        )
    compile_one(
        "output",
        output_core,
        (cpu_hidden, backbone_outputs[0]),
        cpu_pipeline_outputs,
    )

    with torch.no_grad():
        neuron_hidden = compiled["input"](*input_args)
        for index in range(len(layer_cores)):
            value, locations, weights = compiled[
                f"layer_{index}_projection"
            ](neuron_hidden, position_embeddings)
            sampled = [
                compiled[f"sampler_{level}"](value, locations)
                for level in range(3)
            ]
            attention_output = compiled[f"layer_{index}_combine"](
                *sampled,
                weights,
            )
            neuron_hidden = compiled[f"layer_{index}_post"](
                neuron_hidden,
                attention_output,
            )
        neuron_outputs = compiled["output"](
            neuron_hidden,
            backbone_outputs[0],
        )

    metadata = {
        "model_id": args.model_id,
        "precision": args.precision,
        "sampler_implementation": args.sampler_implementation,
        "encoder_layers": len(layer_cores),
        "cpu_pipeline_vs_full_decoder": metrics_tree(
            cpu_pipeline_outputs,
            expected_decoder_outputs,
        ),
        "neuron_pipeline_vs_full_decoder": metrics_tree(
            neuron_outputs,
            expected_decoder_outputs,
        ),
        "components": completed,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    write_progress(progress_path, None, completed)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
