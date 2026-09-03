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
from scripts.run_validation import tensor_metrics


class PixelDecoderInputCore(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.input_projections = decoder.input_projections

    def forward(
        self,
        feature_4: Tensor,
        feature_8: Tensor,
        feature_16: Tensor,
        feature_32: Tensor,
    ) -> Tensor:
        del feature_4
        sources = (
            self.input_projections[0](feature_32),
            self.input_projections[1](feature_16),
            self.input_projections[2](feature_8),
        )
        return torch.cat(
            [source.flatten(2).transpose(1, 2) for source in sources],
            dim=1,
        )


class PixelDecoderEncoderLayerCore(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        attention_mask: Tensor,
        position_embeddings: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
    ):
        super().__init__()
        self.layer = layer
        self.register_buffer("attention_mask", attention_mask)
        self.register_buffer("position_embeddings", position_embeddings)
        self.register_buffer("reference_points", reference_points)
        self.register_buffer("spatial_shapes", spatial_shapes)
        self.register_buffer("level_start_index", level_start_index)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.layer(
            hidden_states,
            self.attention_mask,
            position_embeddings=self.position_embeddings,
            reference_points=self.reference_points,
            spatial_shapes=self.spatial_shapes,
            level_start_index=self.level_start_index,
            output_attentions=False,
        )[0]


class PixelDecoderOutputCore(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        if decoder.num_fpn_levels != 1:
            raise ValueError("This fixed target expects one extra FPN level")
        self.lateral_conv = decoder.lateral_convs[0]
        self.output_conv = decoder.output_convs[0]
        self.mask_projection = decoder.mask_projection

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
            lateral
            + F.interpolate(
                feature_8,
                size=(160, 160),
                mode="bilinear",
                align_corners=False,
            )
        )
        return (
            feature_32,
            feature_16,
            feature_8,
            self.mask_projection(feature_4_output),
        )


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


def trace_and_save(
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
        input_hidden = input_core(*backbone_outputs)

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
        level_start_index = torch.tensor(
            [0, 400, 2000],
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (input_hidden.shape[0], input_hidden.shape[1]),
            dtype=torch.bool,
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

        layer_cores = [
            PixelDecoderEncoderLayerCore(
                layer,
                attention_mask,
                position_embeddings,
                reference_points,
                spatial_shapes,
                level_start_index,
            ).eval()
            for layer in decoder.encoder.layers
        ]
        cpu_hidden = input_hidden
        cpu_layer_outputs = []
        for layer_core in layer_cores:
            cpu_hidden = layer_core(cpu_hidden)
            cpu_layer_outputs.append(cpu_hidden)
        cpu_pipeline_outputs = output_core(
            cpu_hidden,
            backbone_outputs[0],
        )

    cpu_pipeline_metrics = metrics_tree(
        cpu_pipeline_outputs,
        expected_decoder_outputs,
    )
    completed = []
    compiled_modules = []

    specs = [
        (
            "input",
            input_core,
            backbone_outputs,
            input_hidden,
        )
    ]
    layer_input = input_hidden
    for index, (layer_core, layer_output) in enumerate(
        zip(layer_cores, cpu_layer_outputs)
    ):
        specs.append(
            (
                f"encoder_layer_{index}",
                layer_core,
                (layer_input,),
                layer_output,
            )
        )
        layer_input = layer_output
    specs.append(
        (
            "output",
            output_core,
            (cpu_hidden, backbone_outputs[0]),
            cpu_pipeline_outputs,
        )
    )

    for name, module, example_inputs, expected in specs:
        write_progress(progress_path, name, completed)
        print(json.dumps({"compiling": name}), flush=True)
        compiled, result = trace_and_save(
            name,
            module,
            tuple(example_inputs),
            expected,
            output_dir,
            flags,
        )
        compiled_modules.append(compiled)
        completed.append(result)
        write_progress(progress_path, None, completed)
        gc.collect()

    with torch.no_grad():
        neuron_hidden = compiled_modules[0](*backbone_outputs)
        for compiled_layer in compiled_modules[1:-1]:
            neuron_hidden = compiled_layer(neuron_hidden)
        neuron_outputs = compiled_modules[-1](
            neuron_hidden,
            backbone_outputs[0],
        )

    metadata = {
        "model_id": args.model_id,
        "precision": args.precision,
        "encoder_layers": len(layer_cores),
        "cpu_pipeline_vs_full_decoder": cpu_pipeline_metrics,
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
