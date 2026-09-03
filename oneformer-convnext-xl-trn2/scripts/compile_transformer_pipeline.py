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

from neuron_port.components import (
    PixelDecoderCore,
    TaskEncoderCore,
    TransformerCore,
)
from neuron_port.modeling import load_oneformer
from scripts.run_validation import tensor_metrics


class QueryInputCore(nn.Module):
    def __init__(
        self,
        transformer_module: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.query_input_projection = decoder.query_input_projection
        self.task_norm = (
            decoder.decoder_norm if decoder.use_task_norm else nn.Identity()
        )
        self.object_query_count = (
            transformer_module.queries_embedder.weight.shape[0] - 1
        )

    def forward(
        self,
        mask_features: Tensor,
        task_token: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        task_sequence = self.task_norm(task_token).unsqueeze(0)
        output = task_sequence.repeat(self.object_query_count, 1, 1)
        mask_position = (
            self.query_input_projection(mask_features)
            .flatten(2)
            .permute(2, 0, 1)
        )
        return output, mask_position, task_sequence


class QueryLayerCore(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        memory: Tensor,
        query_position: Tensor,
    ):
        super().__init__()
        self.layer = layer
        self.register_buffer("memory", memory)
        self.register_buffer("query_position", query_position)

    def forward(
        self,
        output: Tensor,
        mask_position: Tensor,
    ) -> Tensor:
        return self.layer(
            output,
            self.memory,
            memory_key_padding_mask=None,
            pos=mask_position,
            query_pos=self.query_position,
        )


class MainDecoderPrepareCore(nn.Module):
    def __init__(
        self,
        transformer_module: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.query_norm = decoder.query_transformer.decoder.norm
        self.input_projections = nn.ModuleList(
            list(transformer_module.input_projections)
        )
        self.level_embed = transformer_module.level_embed

    def forward(
        self,
        query_output: Tensor,
        task_sequence: Tensor,
        feature_20: Tensor,
        feature_40: Tensor,
        feature_80: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        object_queries = self.query_norm(query_output)
        output = torch.cat((object_queries, task_sequence), dim=0)
        features = (feature_20, feature_40, feature_80)
        memories = []
        for index, feature in enumerate(features):
            projected = self.input_projections[index](feature)
            projected = (
                projected.flatten(2)
                + self.level_embed.weight[index][None, :, None]
            )
            memories.append(projected.permute(2, 0, 1))
        return output, memories[0], memories[1], memories[2]


class AttentionMaskCore(nn.Module):
    def __init__(
        self,
        decoder: nn.Module,
        target_height: int,
        target_width: int,
    ):
        super().__init__()
        self.decoder_norm = decoder.decoder_norm
        self.mask_embed = decoder.mask_embed
        self.num_heads = decoder.num_heads
        self.target_height = target_height
        self.target_width = target_width

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
        attention_mask = F.interpolate(
            masks,
            size=(self.target_height, self.target_width),
            mode="bilinear",
            align_corners=False,
        )
        return (
            attention_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()


class MainDecoderLayerCore(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        position_embedding: Tensor,
        query_embedding: Tensor,
    ):
        super().__init__()
        self.layer = layer
        self.register_buffer("position_embedding", position_embedding)
        self.register_buffer("query_embedding", query_embedding)

    def forward(
        self,
        output: Tensor,
        memory: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        all_masked = attention_mask.sum(-1) == attention_mask.shape[-1]
        safe_attention_mask = (
            attention_mask & ~all_masked.unsqueeze(-1)
        )
        output, _ = self.layer.cross_attn(
            output,
            memory,
            memory_mask=safe_attention_mask,
            memory_key_padding_mask=None,
            pos=self.position_embedding,
            query_pos=self.query_embedding,
        )
        output, _ = self.layer.self_attn(
            output,
            output_mask=None,
            output_key_padding_mask=None,
            query_pos=self.query_embedding,
        )
        return self.layer.ffn(output)


class FinalPredictionCore(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder_norm = decoder.decoder_norm
        self.class_embed = decoder.class_embed
        self.mask_embed = decoder.mask_embed

    def forward(
        self,
        output: Tensor,
        mask_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        decoder_output = self.decoder_norm(output).transpose(0, 1)
        class_logits = self.class_embed(decoder_output)
        mask_embedding = self.mask_embed(decoder_output)
        mask_logits = torch.einsum(
            "bqc,bchw->bqhw",
            mask_embedding,
            mask_features,
        )
        return class_logits, mask_logits


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
    base = "--model-type=transformer -O1"
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
    task_inputs = inputs["task_inputs"].contiguous()
    _, wrapper = load_oneformer(
        args.model_id,
        cache_dir=args.cache_dir,
        use_custom_grid_sample=args.custom_grid_sample,
        local_files_only=True,
    )
    backbone = wrapper.core.pixel_level_module.encoder.eval()
    pixel_decoder = PixelDecoderCore(wrapper.core).eval()
    task_encoder = TaskEncoderCore(wrapper.core).eval()
    full_transformer = TransformerCore(wrapper.core).eval()
    transformer_module = wrapper.core.transformer_module
    decoder = transformer_module.decoder

    with torch.no_grad():
        backbone_outputs = tuple(backbone(pixel_values).feature_maps)
        pixel_outputs = pixel_decoder(*backbone_outputs)
        task_token = task_encoder(task_inputs)
        expected_outputs = full_transformer(*pixel_outputs, task_token)

        mask_features = pixel_outputs[3]
        query_memory = transformer_module.position_embedder(
            mask_features.shape,
            mask_features.device,
            mask_features.dtype,
            None,
        ).flatten(2).permute(2, 0, 1)
        batch_size = mask_features.shape[0]
        query_position = (
            transformer_module.queries_embedder.weight[:-1]
            .unsqueeze(1)
            .repeat(1, batch_size, 1)
        )
        main_query_embedding = (
            transformer_module.queries_embedder.weight
            .unsqueeze(1)
            .repeat(1, batch_size, 1)
        )
        main_positions = [
            transformer_module.position_embedder(
                feature.shape,
                feature.device,
                feature.dtype,
                None,
            ).flatten(2).permute(2, 0, 1)
            for feature in pixel_outputs[:3]
        ]

        query_input_core = QueryInputCore(
            transformer_module,
            decoder,
        ).eval()
        query_layer_cores = [
            QueryLayerCore(layer, query_memory, query_position).eval()
            for layer in decoder.query_transformer.decoder.layers
        ]
        prepare_core = MainDecoderPrepareCore(
            transformer_module,
            decoder,
        ).eval()
        mask_cores = [
            AttentionMaskCore(decoder, 20, 20).eval(),
            AttentionMaskCore(decoder, 40, 40).eval(),
            AttentionMaskCore(decoder, 80, 80).eval(),
        ]
        main_layer_cores = [
            MainDecoderLayerCore(
                layer,
                main_positions[index % 3],
                main_query_embedding,
            ).eval()
            for index, layer in enumerate(decoder.layers)
        ]
        final_core = FinalPredictionCore(decoder).eval()

        query_output, mask_position, task_sequence = query_input_core(
            mask_features,
            task_token,
        )
        cpu_query_records = []
        for query_layer in query_layer_cores:
            next_query_output = query_layer(
                query_output,
                mask_position,
            )
            cpu_query_records.append(
                {
                    "input": query_output,
                    "output": next_query_output,
                }
            )
            query_output = next_query_output

        (
            decoder_output,
            memory_20,
            memory_40,
            memory_80,
        ) = prepare_core(
            query_output,
            task_sequence,
            *pixel_outputs[:3],
        )
        memories = (memory_20, memory_40, memory_80)
        attention_mask = mask_cores[0](
            decoder_output,
            mask_features,
        )
        cpu_main_records = []
        for index, main_layer in enumerate(main_layer_cores):
            next_decoder_output = main_layer(
                decoder_output,
                memories[index % 3],
                attention_mask,
            )
            next_attention_mask = (
                mask_cores[(index + 1) % 3](
                    next_decoder_output,
                    mask_features,
                )
                if index + 1 < len(main_layer_cores)
                else None
            )
            cpu_main_records.append(
                {
                    "output_input": decoder_output,
                    "memory": memories[index % 3],
                    "attention_mask": attention_mask,
                    "output": next_decoder_output,
                    "next_attention_mask": next_attention_mask,
                }
            )
            decoder_output = next_decoder_output
            if next_attention_mask is not None:
                attention_mask = next_attention_mask
        cpu_pipeline_outputs = final_core(
            decoder_output,
            mask_features,
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

    initial_query_outputs = query_input_core(mask_features, task_token)
    compile_one(
        "query_input",
        query_input_core,
        (mask_features, task_token),
        initial_query_outputs,
    )
    for index, (query_layer, record) in enumerate(
        zip(query_layer_cores, cpu_query_records)
    ):
        compile_one(
            f"query_layer_{index}",
            query_layer,
            (record["input"], mask_position),
            record["output"],
        )
    prepare_outputs = prepare_core(
        query_output,
        task_sequence,
        *pixel_outputs[:3],
    )
    compile_one(
        "main_prepare",
        prepare_core,
        (query_output, task_sequence, *pixel_outputs[:3]),
        prepare_outputs,
    )
    for index, mask_core in enumerate(mask_cores):
        compile_one(
            f"mask_{index}",
            mask_core,
            (decoder_output, mask_features),
            mask_core(decoder_output, mask_features),
        )
    for index, (main_layer, record) in enumerate(
        zip(main_layer_cores, cpu_main_records)
    ):
        compile_one(
            f"decoder_layer_{index}",
            main_layer,
            (
                record["output_input"],
                record["memory"],
                record["attention_mask"],
            ),
            record["output"],
        )
    compile_one(
        "final_prediction",
        final_core,
        (decoder_output, mask_features),
        cpu_pipeline_outputs,
    )

    with torch.no_grad():
        (
            neuron_query_output,
            neuron_mask_position,
            neuron_task_sequence,
        ) = compiled["query_input"](mask_features, task_token)
        for index in range(len(query_layer_cores)):
            neuron_query_output = compiled[f"query_layer_{index}"](
                neuron_query_output,
                neuron_mask_position,
            )
        (
            neuron_decoder_output,
            neuron_memory_20,
            neuron_memory_40,
            neuron_memory_80,
        ) = compiled["main_prepare"](
            neuron_query_output,
            neuron_task_sequence,
            *pixel_outputs[:3],
        )
        neuron_memories = (
            neuron_memory_20,
            neuron_memory_40,
            neuron_memory_80,
        )
        neuron_attention_mask = compiled["mask_0"](
            neuron_decoder_output,
            mask_features,
        )
        for index in range(len(main_layer_cores)):
            neuron_decoder_output = compiled[
                f"decoder_layer_{index}"
            ](
                neuron_decoder_output,
                neuron_memories[index % 3],
                neuron_attention_mask,
            )
            if index + 1 < len(main_layer_cores):
                neuron_attention_mask = compiled[
                    f"mask_{(index + 1) % 3}"
                ](
                    neuron_decoder_output,
                    mask_features,
                )
        neuron_outputs = compiled["final_prediction"](
            neuron_decoder_output,
            mask_features,
        )

    metadata = {
        "model_id": args.model_id,
        "precision": args.precision,
        "query_layers": len(query_layer_cores),
        "decoder_layers": len(main_layer_cores),
        "cpu_pipeline_vs_full_transformer": metrics_tree(
            cpu_pipeline_outputs,
            expected_outputs,
        ),
        "neuron_pipeline_vs_full_transformer": metrics_tree(
            neuron_outputs,
            expected_outputs,
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
