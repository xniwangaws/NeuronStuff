#!/usr/bin/env python3

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from oneformer import (
    add_common_config,
    add_convnext_config,
    add_dinat_config,
    add_oneformer_config,
    add_swin_config,
)


class OneFormerCore(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.task_mlp = model.task_mlp
        self.sem_seg_head = model.sem_seg_head

    def forward(
        self,
        normalized_image: torch.Tensor,
        task_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(normalized_image)
        task_embedding = self.task_mlp(task_tokens.float())
        outputs = self.sem_seg_head(features, task_embedding)
        return outputs["pred_logits"], outputs["pred_masks"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--strict-fp32", action="store_true")
    return parser.parse_args()


def build_cfg(args: argparse.Namespace):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.defrost()
    cfg.MODEL.IS_TRAIN = False
    cfg.MODEL.IS_DEMO = True
    cfg.MODEL.WEIGHTS = args.checkpoint
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.TEST.SEMANTIC_ON = True
    cfg.MODEL.TEST.INSTANCE_ON = False
    cfg.MODEL.TEST.PANOPTIC_ON = False
    cfg.MODEL.TEST.DETECTION_ON = False
    cfg.TEST.AUG.ENABLED = False
    cfg.freeze()
    return cfg


def load_rgb_640(path: str) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(path).convert("RGB")
    image = image.resize((640, 640), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    chw = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()
    return chw, array.astype(np.uint8)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return ordered[index]


def benchmark_cuda(fn, warmups: int, runs: int) -> dict[str, float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    event_times = []
    wall_times = []
    for _ in range(runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        wall_times.append((time.perf_counter() - wall_start) * 1000.0)
        event_times.append(start_event.elapsed_time(end_event))

    return {
        "runs": runs,
        "event_mean_ms": statistics.fmean(event_times),
        "event_p50_ms": statistics.median(event_times),
        "event_p90_ms": percentile(event_times, 0.90),
        "event_min_ms": min(event_times),
        "event_max_ms": max(event_times),
        "wall_mean_ms": statistics.fmean(wall_times),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def semantic_logits(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    upsampled_masks = F.interpolate(
        mask_logits,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    class_probs = class_logits.softmax(dim=-1)[..., :-1]
    mask_probs = upsampled_masks.sigmoid()
    return torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.strict_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    initial_precision = {
        "strict_fp32_requested": args.strict_fp32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }

    cfg = build_cfg(args)
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(args.checkpoint)
    model.cuda()
    core = OneFormerCore(model).eval().cuda()

    raw_chw_cpu, input_rgb = load_rgb_640(args.image)
    Image.fromarray(input_rgb).save(output_dir / "input_640.png")
    raw_chw = raw_chw_cpu.cuda()
    normalized = (
        raw_chw.unsqueeze(0)
        - model.pixel_mean.unsqueeze(0)
    ) / model.pixel_std.unsqueeze(0)
    task_tokens = model.task_tokenizer("The task is semantic").unsqueeze(0).cuda()

    with torch.inference_mode():
        class_logits, mask_logits = core(normalized, task_tokens)
        core_semantic = semantic_logits(class_logits, mask_logits, (640, 640))
        full_output = model(
            [
                {
                    "image": raw_chw,
                    "height": 640,
                    "width": 640,
                    "task": "The task is semantic",
                }
            ]
        )[0]["sem_seg"].unsqueeze(0)
        max_full_core_error = (
            full_output.float() - core_semantic.float()
        ).abs().max().item()
        semantic_pixel_agreement = (
            full_output.argmax(dim=1) == core_semantic.argmax(dim=1)
        ).float().mean().item()

        core_benchmark = benchmark_cuda(
            lambda: core(normalized, task_tokens),
            args.warmups,
            args.runs,
        )
        full_benchmark = benchmark_cuda(
            lambda: model(
                [
                    {
                        "image": raw_chw,
                        "height": 640,
                        "width": 640,
                        "task": "The task is semantic",
                    }
                ]
            ),
            args.warmups,
            args.runs,
        )

    dtype_parameter_counts: dict[str, int] = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        dtype_parameter_counts[key] = (
            dtype_parameter_counts.get(key, 0) + parameter.numel()
        )

    reference = {
        "raw_image_chw": raw_chw_cpu,
        "normalized_image": normalized.cpu(),
        "task_tokens": task_tokens.cpu(),
        "class_logits": class_logits.cpu(),
        "mask_logits": mask_logits.cpu(),
        "semantic_argmax": core_semantic.argmax(dim=1).to(torch.int16).cpu(),
    }
    torch.save(reference, output_dir / "gpu_reference.pt")

    report = {
        "model": "OneFormer ConvNeXt-XL ADE20K 640x640",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": (
            "a022437a6cc16fd1485230670f2f7a3ed5e08ef9f08d3f67a42948e5a6a4d7ca"
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "input_shape": list(normalized.shape),
        "task_shape": list(task_tokens.shape),
        "class_logits_shape": list(class_logits.shape),
        "mask_logits_shape": list(mask_logits.shape),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "dtype_parameter_counts": dtype_parameter_counts,
        "precision": initial_precision,
        "full_vs_core_max_abs_error": max_full_core_error,
        "full_vs_core_semantic_pixel_agreement": semantic_pixel_agreement,
        "core_benchmark": core_benchmark,
        "full_semantic_benchmark": full_benchmark,
    }
    (output_dir / "gpu_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
