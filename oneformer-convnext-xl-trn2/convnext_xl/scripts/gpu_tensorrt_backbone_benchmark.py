#!/usr/bin/env python3

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_tensorrt
from torch import Tensor, nn

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from gpu_reference_benchmark import (
    OneFormerCore,
    build_cfg,
    load_rgb_640,
)


class BackboneTuple(nn.Module):
    def __init__(self, backbone: nn.Module, feature_keys: tuple[str, ...]):
        super().__init__()
        self.backbone = backbone
        self.feature_keys = feature_keys

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        features = self.backbone(image)
        return tuple(features[key] for key in self.feature_keys)


class CompiledBackboneOneFormerCore(nn.Module):
    def __init__(
        self,
        compiled_backbone: nn.Module,
        feature_keys: tuple[str, ...],
        task_mlp: nn.Module,
        sem_seg_head: nn.Module,
    ):
        super().__init__()
        self.compiled_backbone = compiled_backbone
        self.feature_keys = feature_keys
        self.task_mlp = task_mlp
        self.sem_seg_head = sem_seg_head

    def forward(
        self,
        normalized_image: Tensor,
        task_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        feature_values = self.compiled_backbone(normalized_image)
        features = {
            key: value
            for key, value in zip(self.feature_keys, feature_values)
        }
        task_embedding = self.task_mlp(task_tokens.float())
        outputs = self.sem_seg_head(features, task_embedding)
        return outputs["pred_logits"], outputs["pred_masks"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu-reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--engine-cache-dir", required=True)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        required=True,
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return ordered[index]


def benchmark_cuda(fn, warmups: int, runs: int) -> dict[str, float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return {
        "runs": runs,
        "mean_ms": statistics.fmean(times),
        "p50_ms": statistics.median(times),
        "p90_ms": percentile(times, 0.90),
        "min_ms": min(times),
        "max_ms": max(times),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


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


def semantic_argmax(class_logits: Tensor, mask_logits: Tensor) -> Tensor:
    masks = F.interpolate(
        mask_logits,
        size=(640, 640),
        mode="bilinear",
        align_corners=False,
    )
    class_probabilities = class_logits.softmax(dim=-1)[..., :-1]
    mask_probabilities = masks.sigmoid()
    return torch.einsum(
        "bqc,bqhw->bchw",
        class_probabilities,
        mask_probabilities,
    ).argmax(dim=1)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    cache_dir = Path(args.engine_cache_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cfg = build_cfg(args)
    model = build_model(cfg).eval()
    DetectionCheckpointer(model).load(args.checkpoint)
    model.cuda()
    native_core = OneFormerCore(model).eval().cuda()

    raw_chw, _ = load_rgb_640(args.image)
    raw_chw = raw_chw.cuda()
    normalized = (
        raw_chw.unsqueeze(0) - model.pixel_mean.unsqueeze(0)
    ) / model.pixel_std.unsqueeze(0)
    task_tokens = (
        model.task_tokenizer("The task is semantic")
        .unsqueeze(0)
        .cuda()
        .float()
    )
    reference = torch.load(
        args.gpu_reference,
        map_location="cpu",
        weights_only=True,
    )

    with torch.inference_mode():
        native_features = model.backbone(normalized)
        native_outputs = native_core(normalized, task_tokens)
    feature_keys = tuple(native_features)
    native_backbone = BackboneTuple(
        model.backbone,
        feature_keys,
    ).eval().cuda()

    enabled_precision = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.precision]
    compile_options = {
        "enabled_precisions": {enabled_precision},
        "min_block_size": 1,
        "require_full_compilation": True,
        "truncate_double": True,
        "disable_tf32": args.precision == "fp32",
        "optimization_level": 5,
        "workspace_size": 4 << 30,
        "cache_built_engines": True,
        "reuse_cached_engines": True,
        "timing_cache_path": str(cache_dir / "timing_cache.bin"),
    }
    compile_start = time.perf_counter()
    compiled_backbone = torch_tensorrt.compile(
        native_backbone,
        ir="dynamo",
        inputs=[normalized],
        **compile_options,
    )

    with torch.inference_mode():
        compiled_features = compiled_backbone(normalized)
        torch.cuda.synchronize()
    compile_and_first_run_seconds = time.perf_counter() - compile_start

    hybrid_core = CompiledBackboneOneFormerCore(
        compiled_backbone,
        feature_keys,
        model.task_mlp,
        model.sem_seg_head,
    ).eval().cuda()

    with torch.inference_mode():
        native_backbone_benchmark = benchmark_cuda(
            lambda: native_backbone(normalized),
            args.warmups,
            args.runs,
        )
        tensorrt_backbone_benchmark = benchmark_cuda(
            lambda: compiled_backbone(normalized),
            args.warmups,
            args.runs,
        )
        native_core_benchmark = benchmark_cuda(
            lambda: native_core(normalized, task_tokens),
            args.warmups,
            args.runs,
        )
        hybrid_core_benchmark = benchmark_cuda(
            lambda: hybrid_core(normalized, task_tokens),
            args.warmups,
            args.runs,
        )
        hybrid_outputs = hybrid_core(normalized, task_tokens)
        torch.cuda.synchronize()

    hybrid_class, hybrid_mask = hybrid_outputs
    gpu_class = reference["class_logits"].cuda()
    gpu_mask = reference["mask_logits"].cuda()
    hybrid_semantic = semantic_argmax(hybrid_class, hybrid_mask)
    gpu_semantic = reference["semantic_argmax"].cuda().to(torch.int64)

    amp_report = None
    if args.precision != "fp32":
        def amp_backbone_call():
            with torch.autocast("cuda", dtype=enabled_precision):
                return native_backbone(normalized)

        def amp_core_call():
            with torch.autocast("cuda", dtype=enabled_precision):
                return native_core(normalized, task_tokens)

        with torch.inference_mode():
            amp_features = amp_backbone_call()
            amp_outputs = amp_core_call()
            amp_backbone_benchmark = benchmark_cuda(
                amp_backbone_call,
                args.warmups,
                args.runs,
            )
            amp_core_benchmark = benchmark_cuda(
                amp_core_call,
                args.warmups,
                args.runs,
            )
            torch.cuda.synchronize()
        amp_class, amp_mask = amp_outputs
        amp_semantic = semantic_argmax(amp_class, amp_mask)
        amp_report = {
            "backbone_benchmark": amp_backbone_benchmark,
            "core_benchmark": amp_core_benchmark,
            "native_vs_amp_backbone_features": {
                key: tensor_metrics(actual, native_features[key])
                for key, actual in zip(feature_keys, amp_features)
            },
            "native_vs_amp_core": {
                "class_logits": tensor_metrics(
                    amp_class,
                    native_outputs[0],
                ),
                "mask_logits": tensor_metrics(
                    amp_mask,
                    native_outputs[1],
                ),
            },
            "gpu_reference_vs_amp_core": {
                "class_logits": tensor_metrics(amp_class, gpu_class),
                "mask_logits": tensor_metrics(amp_mask, gpu_mask),
                "semantic_pixel_agreement": (
                    amp_semantic == gpu_semantic
                ).float().mean().item(),
            },
        }

    feature_metrics = {
        key: tensor_metrics(actual, native_features[key])
        for key, actual in zip(feature_keys, compiled_features)
    }

    report = {
        "precision": args.precision,
        "scope": "full TensorRT ConvNeXt-XL backbone plus native OneFormer head",
        "backbone_full_compilation_required": True,
        "compile_api": "torch_tensorrt.compile(ir=dynamo)",
        "torch_version": torch.__version__,
        "torch_tensorrt_version": torch_tensorrt.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "feature_keys": list(feature_keys),
        "feature_shapes": {
            key: list(native_features[key].shape)
            for key in feature_keys
        },
        "compile_options": {
            **compile_options,
            "enabled_precisions": [str(enabled_precision)],
        },
        "compile_and_first_run_seconds": compile_and_first_run_seconds,
        "native_backbone_benchmark": native_backbone_benchmark,
        "tensorrt_backbone_benchmark": tensorrt_backbone_benchmark,
        "backbone_speedup": (
            native_backbone_benchmark["mean_ms"]
            / tensorrt_backbone_benchmark["mean_ms"]
        ),
        "native_core_benchmark": native_core_benchmark,
        "tensorrt_backbone_native_head_benchmark": hybrid_core_benchmark,
        "end_to_end_speedup": (
            native_core_benchmark["mean_ms"]
            / hybrid_core_benchmark["mean_ms"]
        ),
        "pytorch_amp": amp_report,
        "pytorch_amp_fp16": (
            amp_report if args.precision == "fp16" else None
        ),
        "native_vs_tensorrt_backbone_features": feature_metrics,
        "native_vs_hybrid_core": {
            "class_logits": tensor_metrics(
                hybrid_class,
                native_outputs[0],
            ),
            "mask_logits": tensor_metrics(
                hybrid_mask,
                native_outputs[1],
            ),
        },
        "gpu_reference_vs_hybrid_core": {
            "class_logits": tensor_metrics(hybrid_class, gpu_class),
            "mask_logits": tensor_metrics(hybrid_mask, gpu_mask),
            "semantic_pixel_agreement": (
                hybrid_semantic == gpu_semantic
            ).float().mean().item(),
        },
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
