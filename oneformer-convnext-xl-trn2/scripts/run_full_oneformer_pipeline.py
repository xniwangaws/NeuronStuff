#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch_neuronx  # noqa: F401 - registers Neuron TorchScript classes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neuron_port.components import (
    PixelDecoderCore,
    TaskEncoderCore,
    TransformerCore,
)
from neuron_port.modeling import load_oneformer
from scripts.run_convnext_stage_pipeline import (
    NeuronConvNextStagePipeline,
)
from scripts.run_validation import percentile, semantic_map, tensor_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--backbone-dir", required=True)
    parser.add_argument("--pixel-decoder-dir", required=True)
    parser.add_argument("--remaining-dir", required=True)
    parser.add_argument("--transformer-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
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


class CompiledPixelDecoderPipeline:
    def __init__(self, directory: Path):
        self.input = torch.jit.load(str(directory / "input.pt"))
        self.samplers = [
            torch.jit.load(str(directory / f"sampler_{index}.pt"))
            for index in range(3)
        ]
        self.projections = [
            torch.jit.load(
                str(directory / f"layer_{index}_projection.pt")
            )
            for index in range(6)
        ]
        self.combiners = [
            torch.jit.load(str(directory / f"layer_{index}_combine.pt"))
            for index in range(6)
        ]
        self.posts = [
            torch.jit.load(str(directory / f"layer_{index}_post.pt"))
            for index in range(6)
        ]
        self.output = torch.jit.load(str(directory / "output.pt"))

    def __call__(self, features, position_embeddings):
        hidden = self.input(features[1], features[2], features[3])
        for projection, combiner, post in zip(
            self.projections,
            self.combiners,
            self.posts,
        ):
            value, locations, weights = projection(
                hidden,
                position_embeddings,
            )
            sampled = [
                sampler(value, locations)
                for sampler in self.samplers
            ]
            attention_output = combiner(*sampled, weights)
            hidden = post(hidden, attention_output)
        return self.output(hidden, features[0])


class CompiledTransformerPipeline:
    def __init__(self, directory: Path):
        self.query_input = torch.jit.load(
            str(directory / "query_input.pt")
        )
        self.query_layers = [
            torch.jit.load(str(directory / f"query_layer_{index}.pt"))
            for index in range(2)
        ]
        self.main_prepare = torch.jit.load(
            str(directory / "main_prepare.pt")
        )
        self.masks = [
            torch.jit.load(str(directory / f"mask_{index}.pt"))
            for index in range(3)
        ]
        self.decoder_layers = [
            torch.jit.load(str(directory / f"decoder_layer_{index}.pt"))
            for index in range(9)
        ]
        self.final_prediction = torch.jit.load(
            str(directory / "final_prediction.pt")
        )

    def __call__(self, pixel_outputs, task_token):
        mask_features = pixel_outputs[3]
        query_output, mask_position, task_sequence = self.query_input(
            mask_features,
            task_token,
        )
        for query_layer in self.query_layers:
            query_output = query_layer(query_output, mask_position)
        (
            output,
            memory_20,
            memory_40,
            memory_80,
        ) = self.main_prepare(
            query_output,
            task_sequence,
            pixel_outputs[0],
            pixel_outputs[1],
            pixel_outputs[2],
        )
        memories = (memory_20, memory_40, memory_80)
        attention_mask = self.masks[0](output, mask_features)
        for index, decoder_layer in enumerate(self.decoder_layers):
            output = decoder_layer(
                output,
                memories[index % 3],
                attention_mask,
            )
            if index + 1 < len(self.decoder_layers):
                attention_mask = self.masks[(index + 1) % 3](
                    output,
                    mask_features,
                )
        return self.final_prediction(output, mask_features)


def build_pixel_position_embeddings(decoder, backbone_outputs):
    source_features = (
        backbone_outputs[3],
        backbone_outputs[2],
        backbone_outputs[1],
    )
    positions = []
    for level, source in enumerate(source_features):
        position = decoder.position_embedding(
            source.shape,
            source.device,
            source.dtype,
        )
        position = position.flatten(2).transpose(1, 2)
        positions.append(
            position + decoder.level_embed[level].view(1, 1, -1)
        )
    return torch.cat(positions, dim=1)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    cpu_pixel_decoder = PixelDecoderCore(wrapper.core).eval()
    cpu_task_encoder = TaskEncoderCore(wrapper.core).eval()
    cpu_transformer = TransformerCore(wrapper.core).eval()

    stage_depths = [len(stage.layers) for stage in backbone.encoder.stages]
    compiled_backbone = NeuronConvNextStagePipeline(
        Path(args.backbone_dir),
        stage_depths,
        1,
    )
    compiled_pixel_decoder = CompiledPixelDecoderPipeline(
        Path(args.pixel_decoder_dir)
    )
    compiled_task_encoder = torch.jit.load(
        str(Path(args.remaining_dir) / "task_encoder.pt")
    )
    compiled_transformer = CompiledTransformerPipeline(
        Path(args.transformer_dir)
    )

    with torch.no_grad():
        cpu_backbone_outputs = tuple(backbone(pixel_values).feature_maps)
        position_embeddings = build_pixel_position_embeddings(
            wrapper.core.pixel_level_module.decoder,
            cpu_backbone_outputs,
        )
        cpu_pixel_outputs = cpu_pixel_decoder(*cpu_backbone_outputs)
        cpu_task_token = cpu_task_encoder(task_inputs)
        cpu_outputs = cpu_transformer(
            *cpu_pixel_outputs,
            cpu_task_token,
        )

        neuron_backbone_outputs = compiled_backbone(pixel_values)
        neuron_pixel_outputs = compiled_pixel_decoder(
            neuron_backbone_outputs,
            position_embeddings,
        )
        neuron_task_token = compiled_task_encoder(task_inputs)
        neuron_outputs = compiled_transformer(
            neuron_pixel_outputs,
            neuron_task_token,
        )

        backbone_latency = benchmark_callable(
            lambda: compiled_backbone(pixel_values),
            args.warmup,
            args.runs,
        )
        pixel_decoder_latency = benchmark_callable(
            lambda: compiled_pixel_decoder(
                neuron_backbone_outputs,
                position_embeddings,
            ),
            args.warmup,
            args.runs,
        )
        task_encoder_latency = benchmark_callable(
            lambda: compiled_task_encoder(task_inputs),
            args.warmup,
            args.runs,
        )
        transformer_latency = benchmark_callable(
            lambda: compiled_transformer(
                neuron_pixel_outputs,
                neuron_task_token,
            ),
            args.warmup,
            args.runs,
        )

        def full_pipeline():
            backbone_outputs = compiled_backbone(pixel_values)
            pixel_outputs = compiled_pixel_decoder(
                backbone_outputs,
                position_embeddings,
            )
            task_token = compiled_task_encoder(task_inputs)
            return compiled_transformer(pixel_outputs, task_token)

        full_latency = benchmark_callable(
            full_pipeline,
            args.warmup,
            args.runs,
        )

    class_logits, mask_logits = neuron_outputs
    expected_class, expected_mask = cpu_outputs
    class_metrics = tensor_metrics(class_logits, expected_class)
    mask_metrics = tensor_metrics(mask_logits, expected_mask)
    neuron_semantic = semantic_map(
        class_logits,
        mask_logits,
        target_size=(640, 640),
    )
    expected_semantic = semantic_map(
        expected_class,
        expected_mask,
        target_size=(640, 640),
    )
    semantic_pixel_agreement = (
        neuron_semantic == expected_semantic
    ).float().mean().item()

    report = {
        "model_id": args.model_id,
        "precision": "bf16-all",
        "instance_type": "trn2.3xlarge",
        "logical_neuroncore_config": 2,
        "batch_size": 1,
        "input_shape": list(pixel_values.shape),
        "component_count": {
            "backbone": 37,
            "pixel_decoder": 23,
            "task_encoder": 1,
            "transformer": 17,
            "total": 78,
        },
        "class_logits": class_metrics,
        "mask_logits": mask_metrics,
        "semantic_pixel_agreement": semantic_pixel_agreement,
        "latency_ms": {
            "backbone": backbone_latency,
            "pixel_decoder": pixel_decoder_latency,
            "task_encoder": task_encoder_latency,
            "transformer": transformer_latency,
            "full_pipeline": full_latency,
            "component_mean_sum": (
                backbone_latency["mean"]
                + pixel_decoder_latency["mean"]
                + task_encoder_latency["mean"]
                + transformer_latency["mean"]
            ),
        },
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
