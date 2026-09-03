#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from transformers import OneFormerProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neuron_port.modeling import load_oneformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="shi-labs/oneformer_ade20k_swin_tiny",
    )
    parser.add_argument("--task", default="semantic")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--cache-dir", default="agent_artifacts/data/hf_cache")
    parser.add_argument("--output-dir", default="agent_artifacts/data/reference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = hf_hub_download(
        repo_id="hf-internal-testing/fixtures_ade20k",
        filename="ADE_val_00000001.jpg",
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(
        image,
        (args.size, args.size),
        method=Image.Resampling.BILINEAR,
    )
    image.save(output_dir / "input.png")

    processor = OneFormerProcessor.from_pretrained(
        args.model_id,
        cache_dir=str(cache_dir),
    )
    inputs = processor(
        image,
        [args.task],
        do_resize=False,
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"].contiguous()
    task_inputs = inputs["task_inputs"].contiguous()

    if tuple(pixel_values.shape) != (1, 3, args.size, args.size):
        raise ValueError(f"Unexpected pixel shape: {tuple(pixel_values.shape)}")

    model, wrapper = load_oneformer(
        args.model_id,
        cache_dir=cache_dir,
        use_custom_grid_sample=False,
    )
    with torch.no_grad():
        class_logits, mask_logits = wrapper(pixel_values, task_inputs)
        full_outputs = model(
            pixel_values=pixel_values,
            task_inputs=task_inputs,
        )
        semantic = processor.post_process_semantic_segmentation(
            full_outputs,
            target_sizes=[(args.size, args.size)],
        )[0]

    torch.save(
        {
            "pixel_values": pixel_values,
            "task_inputs": task_inputs,
        },
        output_dir / "inputs.pt",
    )
    torch.save(
        {
            "class_logits": class_logits,
            "mask_logits": mask_logits,
            "semantic": semantic,
        },
        output_dir / "golden.pt",
    )

    metadata = {
        "model_id": args.model_id,
        "task": args.task,
        "pixel_values_shape": list(pixel_values.shape),
        "task_inputs_shape": list(task_inputs.shape),
        "class_logits_shape": list(class_logits.shape),
        "mask_logits_shape": list(mask_logits.shape),
        "semantic_shape": list(semantic.shape),
        "task_inputs_sum": int(task_inputs.sum().item()),
        "pixel_values_mean": float(pixel_values.mean().item()),
        "pixel_values_std": float(pixel_values.std().item()),
        "class_logits_sum": float(class_logits.sum().item()),
        "mask_logits_mean": float(mask_logits.mean().item()),
        "semantic_sum": int(semantic.sum().item()),
        "torch_version": torch.__version__,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
