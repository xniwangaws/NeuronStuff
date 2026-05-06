"""
S3Diff benchmark harness.

Runs S3Diff super-resolution on a single LQ image across resolution buckets
{1K, 2K, 4K} in BF16, measuring:
  - model_load_time_s   : from_pretrained + LoRA apply + .to(device)
  - cold_start_s        : the very first inference (compile-like warmup)
  - steady_mean_s       : (total_N_runs - cold_start) / (N - 1)
  - p50_s, p95_s        : over the N-1 non-cold runs
  - peak_mem_gb         : torch.cuda.max_memory_allocated peak

Output: JSON per device+resolution, plus one SR image per resolution saved for
accuracy comparison.

Usage (GPU):
    PYTHONPATH=/home/ubuntu/s3diff/repo:/home/ubuntu/s3diff/repo/src \\
    python bench.py --device cuda --resolution 1K \\
        --input_image ~/s3diff/smoke_in/lq_00.png \\
        --sd_path ~/s3diff/models/sd-turbo \\
        --pretrained_path ~/s3diff/models/S3Diff/s3diff.pkl \\
        --de_net_path ~/s3diff/models/S3Diff/de_net.pth \\
        --output_dir ~/s3diff/bench_out \\
        --num_runs 10

The script is intentionally decoupled from inference_s3diff.py so timing is
not polluted by IQA metric initialization / dataset loaders.
"""

import argparse
import gc
import json
import math
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# S3Diff repo imports (PYTHONPATH must include repo root + src/)
from de_net import DEResNet
from s3diff_tile import S3Diff
from utils.wavelet_color import wavelet_color_fix


# ---------------------------------------------------------------------------
# Resolution buckets
# ---------------------------------------------------------------------------
# S3Diff is x4 SR.  Bucket name -> LQ side length (so SR = 4 * that).
BUCKETS = {
    "1K": 256,   # LQ 256 -> SR 1024
    "2K": 512,   # LQ 512 -> SR 2048
    "4K": 1024,  # LQ 1024 -> SR 4096, latent tiling kicks in
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cuda", "neuron"], default="cuda")
    p.add_argument("--resolution", choices=list(BUCKETS), required=True)
    p.add_argument("--input_image", required=True, type=str,
                   help="Path to one source image (any size). It will be "
                        "resized to the bucket's LQ side length (square).")
    p.add_argument("--sd_path", required=True, type=str)
    p.add_argument("--pretrained_path", required=True, type=str)
    p.add_argument("--de_net_path", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--num_runs", type=int, default=10,
                   help="N timed inferences; steady mean = (total-first)/(N-1).")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--lora_rank_unet", type=int, default=32)
    p.add_argument("--lora_rank_vae", type=int, default=16)
    p.add_argument("--latent_tiled_size", type=int, default=96)
    p.add_argument("--latent_tiled_overlap", type=int, default=32)
    p.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    p.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    p.add_argument("--padding_offset", type=int, default=32)
    p.add_argument("--pos_prompt", type=str,
                   default="A high-resolution, 8K, ultra-realistic image with "
                           "sharp focus, vibrant colors, and natural lighting.")
    p.add_argument("--neg_prompt", type=str,
                   default="oil painting, cartoon, blur, dirty, messy, low "
                           "quality, deformation, low resolution, oversmooth")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--align_method", default="wavelet",
                   choices=["wavelet", "adain", "nofix"])
    return p.parse_args()


def build_s3diff_args(cli):
    """S3Diff constructor reads many fields off of an argparse Namespace.

    We forward the benchmark CLI into a compatible Namespace so the S3Diff
    class sees what it expects (lora ranks, tiling knobs, prompts, etc.).
    """
    return argparse.Namespace(
        lora_rank_unet=cli.lora_rank_unet,
        lora_rank_vae=cli.lora_rank_vae,
        latent_tiled_size=cli.latent_tiled_size,
        latent_tiled_overlap=cli.latent_tiled_overlap,
        vae_encoder_tiled_size=cli.vae_encoder_tiled_size,
        vae_decoder_tiled_size=cli.vae_decoder_tiled_size,
        padding_offset=cli.padding_offset,
        pos_prompt=cli.pos_prompt,
        neg_prompt=cli.neg_prompt,
        sd_path=cli.sd_path,
        pretrained_path=cli.pretrained_path,
    )


def prepare_lq_tensor(path, lq_side, device, dtype):
    """Load image, resize to lq_side x lq_side square LQ, return tensor [1,3,H,W] in [0,1]."""
    im = Image.open(path).convert("RGB").resize((lq_side, lq_side), Image.BICUBIC)
    t = transforms.ToTensor()(im).unsqueeze(0).to(device=device)
    return t  # keep float32 here; inference code casts as needed


def run_inference(net_sr, net_de, im_lr, sf, pos_prompt, neg_prompt, padding_offset, amp_dtype):
    """Single forward pass under autocast(amp_dtype). Matches official `--mixed_precision bf16`.
    Model weights stay FP32, activations/matmuls run in amp_dtype.
    """
    im_lr = im_lr.to(memory_format=torch.contiguous_format).float()
    ori_h, ori_w = im_lr.shape[2:]
    im_lr_resize = F.interpolate(
        im_lr, size=(ori_h * sf, ori_w * sf),
        mode="bilinear", align_corners=False,
    ).contiguous()
    im_lr_resize_norm = torch.clamp(im_lr_resize * 2 - 1.0, -1.0, 1.0)
    resize_h, resize_w = im_lr_resize_norm.shape[2:]
    pad_h = (math.ceil(resize_h / 64)) * 64 - resize_h
    pad_w = (math.ceil(resize_w / 64)) * 64 - resize_w
    im_lr_resize_norm = F.pad(im_lr_resize_norm, (0, pad_w, 0, pad_h), mode="reflect")

    B = im_lr_resize.size(0)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
        deg_score = net_de(im_lr)
        x_pred = net_sr(
            im_lr_resize_norm, deg_score,
            pos_prompt=[pos_prompt] * B,
            neg_prompt=[neg_prompt] * B,
        )
        x_pred = x_pred[:, :, :resize_h, :resize_w]
    return x_pred, im_lr_resize


def save_sr(x_pred, im_lr_resize, align_method, out_path):
    out_img = (x_pred * 0.5 + 0.5).cpu().detach().float()
    out_pil = transforms.ToPILImage()(out_img[0].clamp(0, 1))
    if align_method != "nofix":
        lr_pil = transforms.ToPILImage()(im_lr_resize[0].cpu().detach().float().clamp(0, 1))
        if align_method == "wavelet":
            out_pil = wavelet_color_fix(out_pil, lr_pil)
        elif align_method == "adain":
            from utils.wavelet_color import adain_color_fix
            out_pil = adain_color_fix(out_pil, lr_pil)
    out_pil.save(out_path)


class NvidiaSmiSampler:
    """Poll nvidia-smi at ~0.25s intervals to catch true device-level peak VRAM.

    torch.cuda stats miss workspace/scratch from cuDNN/cuBLAS; nvidia-smi sees
    real device memory and is what OOM is measured against.
    """
    def __init__(self, interval_s=0.25, gpu_id=0):
        self.interval_s = interval_s
        self.gpu_id = gpu_id
        self.peak_mib = 0
        self.running = False
        self.t = None

    def _loop(self):
        while self.running:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                     "-i", str(self.gpu_id)],
                    timeout=2,
                ).decode().strip()
                mib = int(out.splitlines()[0])
                if mib > self.peak_mib:
                    self.peak_mib = mib
            except Exception:
                pass
            time.sleep(self.interval_s)

    def start(self):
        self.running = True
        self.peak_mib = 0
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def stop(self):
        self.running = False
        if self.t:
            self.t.join(timeout=2)
        return self.peak_mib / 1024.0  # GB


def peak_mem_gb(device):
    """Return (allocated_peak_gb, reserved_peak_gb).

    - allocated = what PyTorch tensors actively hold (undercounts kernel scratch)
    - reserved  = what the CUDA caching allocator has requested from the driver
                  (closer to real VRAM usage, but still excludes non-PyTorch).
    Use nvidia-smi sampling for true device peak if needed.
    """
    if device == "cuda":
        alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        return alloc, reserved
    # Neuron: filled in by caller via neuron-monitor sampling; return None here.
    return None, None


def main():
    args = parse_args()
    assert args.device == "cuda", "Neuron path not yet wired; use --device cuda."
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    amp_dtype = dtype_map[args.dtype]
    device = torch.device("cuda")

    lq_side = BUCKETS[args.resolution]
    sr_side = lq_side * 4

    # -----------------------------------------------------------------------
    # Phase 1: Model load time
    # -----------------------------------------------------------------------
    print(f"[{args.resolution}] loading models ({args.dtype})...", flush=True)
    t0 = time.perf_counter()

    s3diff_args = build_s3diff_args(args)
    net_sr = S3Diff(
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        sd_path=args.sd_path,
        pretrained_path=args.pretrained_path,
        args=s3diff_args,
    )
    net_sr.set_eval()
    # Model weights stay in FP32; autocast(bf16) wraps the forward below (matches
    # official `accelerate launch --mixed_precision bf16` semantics).
    net_sr = net_sr.to(device)

    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de = net_de.to(device)
    net_de.eval()

    torch.cuda.synchronize()
    load_time_s = time.perf_counter() - t0

    # -----------------------------------------------------------------------
    # Phase 2: Prepare input
    # -----------------------------------------------------------------------
    im_lr = prepare_lq_tensor(args.input_image, lq_side, device, amp_dtype)
    print(f"[{args.resolution}] LQ shape: {tuple(im_lr.shape)} -> SR target {sr_side}x{sr_side}", flush=True)

    # -----------------------------------------------------------------------
    # Phase 3: Timed runs
    # -----------------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    smi = NvidiaSmiSampler(interval_s=0.25) if args.device == "cuda" else None
    if smi:
        smi.start()
    per_run_s = []
    sr_saved_path = None

    for i in range(args.num_runs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        x_pred, im_lr_resize = run_inference(
            net_sr, net_de, im_lr, sf=4,
            pos_prompt=args.pos_prompt, neg_prompt=args.neg_prompt,
            padding_offset=args.padding_offset, amp_dtype=amp_dtype,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        per_run_s.append(dt)
        print(f"[{args.resolution}] run {i+1}/{args.num_runs}: {dt:.3f}s", flush=True)

        # Save the SR image from the second run (post-warmup, deterministic).
        if i == 1 and sr_saved_path is None:
            sr_saved_path = os.path.join(args.output_dir, f"sr_{args.resolution}.png")
            save_sr(x_pred, im_lr_resize, args.align_method, sr_saved_path)

    cold_s = per_run_s[0]
    rest = per_run_s[1:]
    total_s = sum(per_run_s)
    steady_mean_s = (total_s - cold_s) / max(1, len(rest))
    rest_sorted = sorted(rest)

    def q(sorted_list, p):
        if not sorted_list:
            return None
        k = max(0, min(len(sorted_list) - 1, int(round((p / 100) * (len(sorted_list) - 1)))))
        return sorted_list[k]

    peak_alloc_gb, peak_reserved_gb = peak_mem_gb("cuda")
    peak_device_gb = smi.stop() if smi else None

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    report = {
        "device": args.device,
        "gpu_name": gpu_name,
        "host": socket.gethostname(),
        "resolution": args.resolution,
        "lq_side": lq_side,
        "sr_side": sr_side,
        "dtype": args.dtype,
        "num_runs": args.num_runs,
        "load_time_s": round(load_time_s, 4),
        "cold_start_s": round(cold_s, 4),
        "steady_mean_s": round(steady_mean_s, 4),
        "p50_s": round(q(rest_sorted, 50), 4) if rest_sorted else None,
        "p95_s": round(q(rest_sorted, 95), 4) if rest_sorted else None,
        "min_s": round(min(rest), 4) if rest else None,
        "max_s": round(max(rest), 4) if rest else None,
        "peak_alloc_gb": round(peak_alloc_gb, 4) if peak_alloc_gb else None,
        "peak_reserved_gb": round(peak_reserved_gb, 4) if peak_reserved_gb else None,
        "peak_device_gb": round(peak_device_gb, 4) if peak_device_gb else None,
        "per_run_s": [round(x, 4) for x in per_run_s],
        "sr_image": sr_saved_path,
        "input_image": args.input_image,
        "seed": args.seed,
        "align_method": args.align_method,
        "torch_version": torch.__version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    json_path = os.path.join(
        args.output_dir,
        f"{gpu_name.replace(' ', '_').replace('/', '_')}_{args.resolution}_{args.dtype}.json",
    )
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== REPORT ===")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {json_path}")
    if sr_saved_path:
        print(f"SR image: {sr_saved_path}")

    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
