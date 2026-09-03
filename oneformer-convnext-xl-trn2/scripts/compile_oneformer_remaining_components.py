#!/usr/bin/env python3

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch_neuronx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neuron_port.components import (
    PixelDecoderCore,
    TaskEncoderCore,
    TransformerCore,
)
from neuron_port.modeling import load_oneformer
from scripts.run_validation import tensor_metrics


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


def compiler_args(model_type: str, precision: str) -> str:
    base = f"--model-type={model_type} -O1"
    if precision == "bf16-all":
        return f"{base} --auto-cast=all --auto-cast-type=bf16"
    if precision == "bf16":
        return f"{base} --auto-cast=matmult --auto-cast-type=bf16"
    return f"{base} --auto-cast=none"


def shape_tree(output) -> list[list[int]]:
    if isinstance(output, (tuple, list)):
        return [list(value.shape) for value in output]
    return [list(output.shape)]


def output_metrics(actual, expected):
    if isinstance(actual, (tuple, list)):
        return [
            tensor_metrics(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        ]
    return tensor_metrics(actual, expected)


def write_progress(path: Path, results: list[dict], current: str | None) -> None:
    path.write_text(
        json.dumps({"current": current, "results": results}, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"

    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    pixel_values = inputs["pixel_values"].contiguous()
    task_inputs = inputs["task_inputs"].contiguous()
    _, wrapper = load_oneformer(
        args.model_id,
        cache_dir=args.cache_dir,
        use_custom_grid_sample=args.custom_grid_sample,
        local_files_only=True,
    )

    pixel_decoder = PixelDecoderCore(wrapper.core).eval()
    task_encoder = TaskEncoderCore(wrapper.core).eval()
    transformer = TransformerCore(wrapper.core).eval()
    with torch.no_grad():
        backbone_outputs = tuple(
            wrapper.core.pixel_level_module.encoder(
                pixel_values
            ).feature_maps
        )
        pixel_outputs = pixel_decoder(*backbone_outputs)
        task_token = task_encoder(task_inputs)
        transformer_outputs = transformer(*pixel_outputs, task_token)

    specs = [
        {
            "name": "task_encoder",
            "module": task_encoder,
            "inputs": (task_inputs,),
            "expected": task_token,
            "model_type": "transformer",
        },
        {
            "name": "transformer",
            "module": transformer,
            "inputs": (*pixel_outputs, task_token),
            "expected": transformer_outputs,
            "model_type": "transformer",
        },
        {
            "name": "pixel_decoder",
            "module": pixel_decoder,
            "inputs": backbone_outputs,
            "expected": pixel_outputs,
            "model_type": "unet-inference",
        },
    ]

    results = []
    for spec in specs:
        name = spec["name"]
        artifact_path = output_dir / f"{name}.pt"
        workdir = output_dir / f"{name}_workdir"
        write_progress(progress_path, results, name)
        flags = compiler_args(spec["model_type"], args.precision)
        result = {
            "name": name,
            "input_shapes": [list(value.shape) for value in spec["inputs"]],
            "output_shapes": shape_tree(spec["expected"]),
            "compiler_args": flags,
        }
        print(json.dumps({"compiling": result}), flush=True)
        try:
            start = time.perf_counter()
            if artifact_path.exists():
                compiled = torch.jit.load(str(artifact_path))
                result["skipped_existing"] = True
            else:
                compiled = torch_neuronx.trace(
                    spec["module"],
                    spec["inputs"],
                    compiler_args=flags,
                    compiler_workdir=str(workdir),
                )
                torch.jit.save(compiled, str(artifact_path))
                result["compile_seconds"] = time.perf_counter() - start
            with torch.no_grad():
                actual = compiled(*spec["inputs"])
            result["artifact_bytes"] = artifact_path.stat().st_size
            result["metrics"] = output_metrics(actual, spec["expected"])
            result["status"] = "pass"
            del compiled
        except Exception as error:
            result["status"] = "failed"
            result["error_type"] = type(error).__name__
            result["error"] = str(error)
        results.append(result)
        gc.collect()
        write_progress(progress_path, results, None)

    metadata = {
        "model_id": args.model_id,
        "precision": args.precision,
        "results": results,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
