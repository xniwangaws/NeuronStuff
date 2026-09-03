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
        choices=("fp32", "fp16"),
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


def semantic_argmax(
    class_logits: Tensor,
    mask_logits: Tensor,
) -> Tensor:
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
    core = OneFormerCore(model).eval().cuda()

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
        native_outputs = core(normalized, task_tokens)
        native_benchmark = benchmark_cuda(
            lambda: core(normalized, task_tokens),
            args.warmups,
            args.runs,
        )

    enabled_precision = (
        torch.float32 if args.precision == "fp32" else torch.float16
    )
    compile_options = {
        "enabled_precisions": {enabled_precision},
        "min_block_size": 5,
        "require_full_compilation": False,
        "truncate_double": True,
        "disable_tf32": args.precision == "fp32",
        "optimization_level": 5,
        "workspace_size": 4 << 30,
        "cache_built_engines": True,
        "reuse_cached_engines": True,
        "timing_cache_path": str(cache_dir / "timing_cache.bin"),
    }
    torch._dynamo.reset()
    compiled = torch.compile(
        core,
        backend="tensorrt",
        options=compile_options,
        dynamic=False,
        fullgraph=False,
    )

    compile_start = time.perf_counter()
    with torch.inference_mode():
        compiled_outputs = compiled(normalized, task_tokens)
        torch.cuda.synchronize()
    compile_and_first_run_seconds = time.perf_counter() - compile_start

    with torch.inference_mode():
        trt_benchmark = benchmark_cuda(
            lambda: compiled(normalized, task_tokens),
            args.warmups,
            args.runs,
        )
        compiled_outputs = compiled(normalized, task_tokens)
        torch.cuda.synchronize()

    trt_class, trt_mask = compiled_outputs
    gpu_class = reference["class_logits"].cuda()
    gpu_mask = reference["mask_logits"].cuda()
    trt_semantic = semantic_argmax(trt_class, trt_mask)
    gpu_semantic = reference["semantic_argmax"].cuda().to(torch.int64)

    report = {
        "precision": args.precision,
        "torch_version": torch.__version__,
        "torch_tensorrt_version": torch_tensorrt.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "compile_options": {
            **compile_options,
            "enabled_precisions": [str(enabled_precision)],
        },
        "compile_and_first_run_seconds": compile_and_first_run_seconds,
        "native_benchmark": native_benchmark,
        "tensorrt_benchmark": trt_benchmark,
        "speedup": (
            native_benchmark["mean_ms"] / trt_benchmark["mean_ms"]
        ),
        "native_vs_tensorrt": {
            "class_logits": tensor_metrics(
                trt_class,
                native_outputs[0],
            ),
            "mask_logits": tensor_metrics(
                trt_mask,
                native_outputs[1],
            ),
        },
        "gpu_reference_vs_tensorrt": {
            "class_logits": tensor_metrics(trt_class, gpu_class),
            "mask_logits": tensor_metrics(trt_mask, gpu_mask),
            "semantic_pixel_agreement": (
                trt_semantic == gpu_semantic
            ).float().mean().item(),
        },
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
