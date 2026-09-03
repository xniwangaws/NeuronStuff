#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.models.oneformer import modeling_oneformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from neuron_port.modeling import load_oneformer, patch_oneformer_for_neuron


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--gpu-reference", required=True)
    parser.add_argument("--inputs-output", required=True)
    parser.add_argument("--golden-output", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def metrics(actual: Tensor, expected: Tensor) -> dict[str, float]:
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


def semantic_argmax(
    class_logits: Tensor,
    mask_logits: Tensor,
    output_size: tuple[int, int],
) -> Tensor:
    masks = F.interpolate(
        mask_logits,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    class_probabilities = class_logits.softmax(dim=-1)[..., :-1]
    mask_probabilities = masks.sigmoid()
    semantic = torch.einsum(
        "bqc,bqhw->bchw",
        class_probabilities,
        mask_probabilities,
    )
    return semantic.argmax(dim=1)


def main() -> None:
    args = parse_args()
    report_path = Path(args.report)
    inputs_path = Path(args.inputs_output)
    golden_path = Path(args.golden_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path.parent.mkdir(parents=True, exist_ok=True)

    reference = torch.load(
        args.gpu_reference,
        map_location="cpu",
        weights_only=True,
    )
    pixel_values = reference["normalized_image"].contiguous()
    task_inputs = reference["task_tokens"].contiguous()

    _, wrapper = load_oneformer(
        args.model_dir,
        cache_dir=Path(args.model_dir).parent / "hf_cache",
        use_custom_grid_sample=False,
        local_files_only=True,
    )
    with torch.no_grad():
        hf_class_logits, hf_mask_logits = wrapper(
            pixel_values,
            task_inputs,
        )

    patch_oneformer_for_neuron(modeling_oneformer)
    with torch.no_grad():
        patched_class_logits, patched_mask_logits = wrapper(
            pixel_values,
            task_inputs,
        )

    gpu_semantic = reference["semantic_argmax"].to(torch.int64)
    hf_semantic = semantic_argmax(
        hf_class_logits,
        hf_mask_logits,
        tuple(gpu_semantic.shape[-2:]),
    )
    patched_semantic = semantic_argmax(
        patched_class_logits,
        patched_mask_logits,
        tuple(gpu_semantic.shape[-2:]),
    )

    report = {
        "gpu_vs_hf": {
            "class_logits": metrics(
                hf_class_logits,
                reference["class_logits"],
            ),
            "mask_logits": metrics(
                hf_mask_logits,
                reference["mask_logits"],
            ),
            "semantic_pixel_agreement": (
                hf_semantic == gpu_semantic
            ).float().mean().item(),
        },
        "hf_vs_neuron_patch": {
            "class_logits": metrics(
                patched_class_logits,
                hf_class_logits,
            ),
            "mask_logits": metrics(
                patched_mask_logits,
                hf_mask_logits,
            ),
            "semantic_pixel_agreement": (
                patched_semantic == hf_semantic
            ).float().mean().item(),
        },
    }
    report["passed"] = (
        report["gpu_vs_hf"]["class_logits"]["cosine_similarity"] >= 0.999
        and report["gpu_vs_hf"]["mask_logits"]["cosine_similarity"] >= 0.999
        and report["gpu_vs_hf"]["semantic_pixel_agreement"] >= 0.99
        and report["hf_vs_neuron_patch"]["class_logits"][
            "cosine_similarity"
        ]
        >= 0.999
        and report["hf_vs_neuron_patch"]["mask_logits"][
            "cosine_similarity"
        ]
        >= 0.999
        and report["hf_vs_neuron_patch"]["semantic_pixel_agreement"]
        >= 0.99
    )

    torch.save(
        {
            "pixel_values": pixel_values,
            "task_inputs": task_inputs,
        },
        inputs_path,
    )
    torch.save(
        {
            "class_logits": reference["class_logits"],
            "mask_logits": reference["mask_logits"],
            "semantic": gpu_semantic[0],
        },
        golden_path,
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise ValueError("Converted OneFormer did not match the GPU reference")


if __name__ == "__main__":
    main()
