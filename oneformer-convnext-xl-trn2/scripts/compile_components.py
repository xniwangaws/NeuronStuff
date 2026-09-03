#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch_neuronx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neuron_port.components import (
    BackboneCore,
    PixelDecoderCore,
    PixelLevelCore,
    TaskEncoderCore,
    TransformerCore,
)
from neuron_port.modeling import load_oneformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="shi-labs/oneformer_ade20k_swin_tiny",
    )
    parser.add_argument(
        "--inputs",
        default="agent_artifacts/data/reference/inputs.pt",
    )
    parser.add_argument("--cache-dir", default="agent_artifacts/data/hf_cache")
    parser.add_argument(
        "--output-dir",
        default="agent_artifacts/traces/components",
    )
    parser.add_argument("--custom-grid-sample", action="store_true")
    parser.add_argument("--fine-pixel-split", action="store_true")
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "bf16-all"),
        default="fp32",
    )
    return parser.parse_args()


def compiler_args(model_type: str, precision: str) -> str:
    if precision == "bf16-all":
        return (
            f"--model-type={model_type} -O1 "
            "--auto-cast=all --auto-cast-type=bf16"
        )
    if precision == "bf16":
        return (
            f"--model-type={model_type} -O1 "
            "--auto-cast=matmult --auto-cast-type=bf16"
        )
    return f"--model-type={model_type} -O1 --auto-cast=none"


def trace_and_save(
    module: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    compiler_args: str,
    output_path: Path,
) -> tuple[torch.jit.ScriptModule, float]:
    start = time.perf_counter()
    traced = torch_neuronx.trace(
        module,
        example_inputs,
        compiler_args=compiler_args,
        compiler_workdir=str(output_path.with_suffix("")) + "_workdir",
    )
    torch.jit.save(traced, str(output_path))
    return traced, time.perf_counter() - start


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    pixel_values = inputs["pixel_values"].contiguous()
    task_inputs = inputs["task_inputs"].contiguous()

    _, wrapper = load_oneformer(
        args.model_id,
        cache_dir=args.cache_dir,
        use_custom_grid_sample=args.custom_grid_sample,
        local_files_only=True,
    )
    pixel_core = PixelLevelCore(wrapper.core).eval()
    backbone_core = BackboneCore(wrapper.core).eval()
    pixel_decoder_core = PixelDecoderCore(wrapper.core).eval()
    task_core = TaskEncoderCore(wrapper.core).eval()
    transformer_core = TransformerCore(wrapper.core).eval()

    with torch.no_grad():
        if args.fine_pixel_split:
            backbone_outputs = backbone_core(pixel_values)
            pixel_outputs = pixel_decoder_core(*backbone_outputs)
        else:
            backbone_outputs = None
            pixel_outputs = pixel_core(pixel_values)
        task_token = task_core(task_inputs)
        cpu_outputs = transformer_core(*pixel_outputs, task_token)

    compile_seconds = {}
    if args.fine_pixel_split:
        compiled_backbone, compile_seconds["backbone"] = trace_and_save(
            backbone_core,
            (pixel_values,),
            compiler_args("unet-inference", args.precision),
            output_dir / "backbone.pt",
        )
        compiled_pixel_decoder, compile_seconds["pixel_decoder"] = (
            trace_and_save(
                pixel_decoder_core,
                backbone_outputs,
                compiler_args("transformer", args.precision),
                output_dir / "pixel_decoder.pt",
            )
        )
        compiled_pixel = None
    else:
        compiled_pixel, compile_seconds["pixel_level"] = trace_and_save(
            pixel_core,
            (pixel_values,),
            compiler_args("unet-inference", args.precision),
            output_dir / "pixel_level.pt",
        )
        compiled_backbone = None
        compiled_pixel_decoder = None

    compiled_task, compile_seconds["task_encoder"] = trace_and_save(
        task_core,
        (task_inputs,),
        compiler_args("transformer", args.precision),
        output_dir / "task_encoder.pt",
    )
    compiled_transformer, compile_seconds["transformer"] = trace_and_save(
        transformer_core,
        (*pixel_outputs, task_token),
        compiler_args("transformer", args.precision),
        output_dir / "transformer.pt",
    )

    with torch.no_grad():
        if args.fine_pixel_split:
            neuron_backbone_outputs = compiled_backbone(pixel_values)
            neuron_pixel_outputs = compiled_pixel_decoder(
                *neuron_backbone_outputs,
            )
        else:
            neuron_backbone_outputs = None
            neuron_pixel_outputs = compiled_pixel(pixel_values)
        neuron_task_token = compiled_task(task_inputs)
        neuron_outputs = compiled_transformer(
            *neuron_pixel_outputs,
            neuron_task_token,
        )

    metadata = {
        "custom_grid_sample": args.custom_grid_sample,
        "fine_pixel_split": args.fine_pixel_split,
        "precision": args.precision,
        "compile_seconds": compile_seconds,
        "backbone_output_shapes": (
            [list(value.shape) for value in backbone_outputs]
            if backbone_outputs is not None
            else None
        ),
        "pixel_output_shapes": [list(value.shape) for value in pixel_outputs],
        "task_token_shape": list(task_token.shape),
        "cpu_output_shapes": [list(value.shape) for value in cpu_outputs],
        "neuron_output_shapes": [list(value.shape) for value in neuron_outputs],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
