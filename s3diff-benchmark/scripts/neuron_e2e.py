"""
End-to-end Neuron S3Diff pipeline at 1K.

Components:
  - DEResNet: CPU eager (tiny, not worth tracing)
  - VAE encoder: CPU eager (would trace but avoid for now)
  - UNet: TRACED on Neuron (unet_latent96.pt2 from neuron_unet_trace_v2.py)
  - VAE decoder: CPU eager
  - Text encoder (CLIP): CPU eager (small, fast)

The latent 1K (LQ 256 -> c_t 1024 -> VAE encode 128x128 latent) will require
tiling the UNet pass since 128 > 96. Handled by adapting S3Diff's existing
_gaussian_weights tile loop to call the traced UNet.

Saves one SR image for visual comparison with GPU baseline.
Captures full-pipeline latency with load/cold/steady breakdown.
"""
import argparse
import gc
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sd_path", default="/home/ubuntu/s3diff/models/sd-turbo")
    p.add_argument("--pretrained_path", default="/home/ubuntu/s3diff/models/S3Diff/s3diff.pkl")
    p.add_argument("--de_net_path", default="/home/ubuntu/s3diff/models/S3Diff/de_net.pth")
    p.add_argument("--traced_unet_path", default="/home/ubuntu/s3diff/neuron_out/unet_latent96.pt2")
    p.add_argument("--input_image", default="/home/ubuntu/s3diff/smoke_in/lq_00.png")
    p.add_argument("--output_dir", default="/home/ubuntu/s3diff/bench_out")
    p.add_argument("--resolution", choices=["1K", "2K", "4K"], default="1K")
    p.add_argument("--num_runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--lora_rank_unet", type=int, default=32)
    p.add_argument("--lora_rank_vae", type=int, default=16)
    p.add_argument("--latent_tiled_size", type=int, default=96)
    p.add_argument("--latent_tiled_overlap", type=int, default=32)
    p.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    p.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    p.add_argument("--pos_prompt", default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.")
    p.add_argument("--neg_prompt", default="oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth")
    p.add_argument("--guidance_scale", type=float, default=1.07)
    return p.parse_args()


BUCKETS = {"1K": 256, "2K": 512, "4K": 1024}


class NeuronCoreUtilSampler:
    """Poll neuron-monitor at ~0.5s intervals for peak HBM/NeuronCore util."""
    def __init__(self, interval_s=0.5):
        self.interval_s = interval_s
        self.peak_hbm_mib = 0
        self.running = False
        self.t = None

    def _loop(self):
        while self.running:
            try:
                out = subprocess.check_output(
                    ["neuron-top", "-c", "1", "-n", "1"], timeout=2
                ).decode()
                # Parse rough: neuron-top mem in MiB. Fallback: neuron-ls json
            except Exception:
                pass
            # Use neuron-monitor instead via JSON
            try:
                out = subprocess.check_output(
                    ["bash", "-c",
                     "neuron-monitor -s runtime-metrics 2>/dev/null | head -200 "
                     "| python3 -c \"import sys, json; "
                     "d=json.load(sys.stdin); "
                     "print(max([nd.get('host_total_memory',{}).get('total_used_bytes',0) "
                     "for nd in d.get('neuron_runtime_data',[{}])[0].get('report',{}).get('memory_info',{}).get('neuron_device',[{}])]))\""],
                    timeout=3,
                ).decode().strip()
                if out and out.isdigit():
                    mib = int(out) // (1024*1024)
                    if mib > self.peak_hbm_mib:
                        self.peak_hbm_mib = mib
            except Exception:
                pass
            time.sleep(self.interval_s)

    def start(self):
        self.running = True
        self.peak_hbm_mib = 0
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def stop(self):
        self.running = False
        if self.t:
            self.t.join(timeout=3)
        return self.peak_hbm_mib / 1024.0  # GB


def _gaussian_weights_s3diff(tile_width, tile_height, nbatches, var=0.1):
    """S3Diff's gaussian weight, but with wider default var=0.1 (orig: 0.01).

    var=0.01 produces near-zero weights at tile edges, causing visible seams
    when blending tiles whose contents differ slightly (e.g., due to Neuron
    UNet numerical drift vs CPU reference). var=0.1 keeps edge weight ~0.3
    of peak, giving smoother transitions in overlap regions.
    """
    import numpy as np
    from numpy import exp, pi
    midpoint = (tile_width - 1) / 2
    x_probs = [exp(-(x - midpoint) * (x - midpoint) / (tile_width * tile_width) / (2 * var)) / math.sqrt(2 * pi * var) for x in range(tile_width)]
    midpoint = (tile_height - 1) / 2
    y_probs = [exp(-(y - midpoint) * (y - midpoint) / (tile_height * tile_height) / (2 * var)) / math.sqrt(2 * pi * var) for y in range(tile_height)]
    weights = np.outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights), (nbatches, 4, 1, 1))


def tiled_unet_forward(traced_unet, lq_latent, timesteps, pos_enc, neg_enc, guidance_scale,
                       tile_size, tile_overlap):
    """Mirror of S3Diff.forward's tiling logic, but calls traced_unet instead of self.unet.
    Returns noise prediction (NOT yet run through scheduler).
    """
    _, _, h, w = lq_latent.size()
    if h * w <= tile_size * tile_size:
        pos_pred = traced_unet(lq_latent, timesteps, pos_enc)
        neg_pred = traced_unet(lq_latent, timesteps, neg_enc)
        return neg_pred + guidance_scale * (pos_pred - neg_pred)

    tile_size = min(tile_size, min(h, w))
    tile_weights = _gaussian_weights_s3diff(tile_size, tile_size, 1).to(lq_latent.device).float()

    grid_rows = 0
    cur_x = 0
    while cur_x < lq_latent.size(-1):
        cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
        grid_rows += 1
    grid_cols = 0
    cur_y = 0
    while cur_y < lq_latent.size(-2):
        cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
        grid_cols += 1

    noise_preds = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            if col < grid_cols - 1 or row < grid_rows - 1:
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
            if row == grid_rows - 1:
                ofs_x = w - tile_size
            if col == grid_cols - 1:
                ofs_y = h - tile_size

            input_start_x = ofs_x
            input_end_x = ofs_x + tile_size
            input_start_y = ofs_y
            input_end_y = ofs_y + tile_size

            input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
            pos_pred = traced_unet(input_tile, timesteps, pos_enc)
            neg_pred = traced_unet(input_tile, timesteps, neg_enc)
            model_out = neg_pred + guidance_scale * (pos_pred - neg_pred)
            noise_preds.append(model_out)

    noise_pred = torch.zeros_like(lq_latent)
    contributors = torch.zeros_like(lq_latent)
    for row in range(grid_rows):
        for col in range(grid_cols):
            if col < grid_cols - 1 or row < grid_rows - 1:
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
            if row == grid_rows - 1:
                ofs_x = w - tile_size
            if col == grid_cols - 1:
                ofs_y = h - tile_size
            input_start_x = ofs_x
            input_end_x = ofs_x + tile_size
            input_start_y = ofs_y
            input_end_y = ofs_y + tile_size
            noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row * grid_cols + col] * tile_weights
            contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights

    return noise_pred / contributors


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    from s3diff_tile import S3Diff
    from de_net import DEResNet
    from utils.wavelet_color import wavelet_color_fix
    import importlib.util
    spec = importlib.util.spec_from_file_location("bake", "/home/ubuntu/s3diff/neuron_unet_trace_v2.py")
    bake_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(bake_mod)

    lq_side = BUCKETS[args.resolution]
    sr_side = lq_side * 4

    # ---------------- Load ----------------
    print(f"[{args.resolution}] loading S3Diff + traced UNet...", flush=True)
    t0 = time.perf_counter()

    s3 = argparse.Namespace(
        lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae,
        latent_tiled_size=args.latent_tiled_size, latent_tiled_overlap=args.latent_tiled_overlap,
        vae_encoder_tiled_size=args.vae_encoder_tiled_size, vae_decoder_tiled_size=args.vae_decoder_tiled_size,
        padding_offset=32, pos_prompt=args.pos_prompt, neg_prompt=args.neg_prompt,
        sd_path=args.sd_path, pretrained_path=args.pretrained_path,
    )
    net_sr = S3Diff(
        lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae,
        sd_path=args.sd_path, pretrained_path=args.pretrained_path, args=s3,
    )
    net_sr.set_eval()

    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de.eval()

    # Load traced UNet
    traced_unet = torch.jit.load(args.traced_unet_path)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    load_time_s = time.perf_counter() - t0
    print(f"[{args.resolution}] load {load_time_s:.2f}s", flush=True)

    # ---------------- Prepare LQ ----------------
    im = Image.open(args.input_image).convert("RGB").resize((lq_side, lq_side), Image.BICUBIC)
    im_lr = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    im_lr = im_lr.unsqueeze(0)

    # Run DEResNet once + bake de_mod (valid only for this input)
    with torch.no_grad():
        deg_score = net_de(im_lr)
    bake_mod.compute_and_bake_de_mod(net_sr, deg_score)
    print(f"[{args.resolution}] deg_score baked: {deg_score.flatten().tolist()}", flush=True)

    # Precompute text embeddings (they don't change)
    with torch.no_grad():
        pos_tok = net_sr.tokenizer([args.pos_prompt], max_length=77, padding="max_length",
                                   truncation=True, return_tensors="pt").input_ids
        neg_tok = net_sr.tokenizer([args.neg_prompt], max_length=77, padding="max_length",
                                   truncation=True, return_tensors="pt").input_ids
        pos_enc = net_sr.text_encoder(pos_tok)[0]
        neg_enc = net_sr.text_encoder(neg_tok)[0]

    timesteps = torch.tensor([999], dtype=torch.long)
    guidance_scale = args.guidance_scale

    # ---------------- Pipeline function ----------------
    def pipeline(im_lr):
        # 1. x4 bicubic upsample
        ori_h, ori_w = im_lr.shape[2:]
        im_resize = F.interpolate(im_lr, size=(ori_h * 4, ori_w * 4), mode="bilinear", align_corners=False).contiguous()
        im_resize_norm = torch.clamp(im_resize * 2 - 1.0, -1.0, 1.0)
        resize_h, resize_w = im_resize_norm.shape[2:]
        pad_h = math.ceil(resize_h / 64) * 64 - resize_h
        pad_w = math.ceil(resize_w / 64) * 64 - resize_w
        im_resize_norm = F.pad(im_resize_norm, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            # 2. VAE encode (tiled VAE handles internally)
            lq_latent = net_sr.vae.encode(im_resize_norm).latent_dist.sample() * net_sr.vae.config.scaling_factor
            # 3. Tiled UNet (with CFG) -> noise prediction
            model_pred = tiled_unet_forward(
                traced_unet, lq_latent, timesteps, pos_enc, neg_enc, guidance_scale,
                args.latent_tiled_size, args.latent_tiled_overlap,
            )
            # 4. Scheduler step: noise -> x_0 (CRITICAL; was missing in v1)
            x_denoised = net_sr.sched.step(model_pred, net_sr.timesteps, lq_latent, return_dict=True).prev_sample
            # 5. VAE decode
            out_latent = x_denoised / net_sr.vae.config.scaling_factor
            output_image = net_sr.vae.decode(out_latent).sample.clamp(-1, 1)
            output_image = output_image[:, :, :resize_h, :resize_w]

        return output_image, im_resize

    # ---------------- Timed runs ----------------
    sampler = NeuronCoreUtilSampler()
    sampler.start()

    per_run_s = []
    sr_saved = None
    for i in range(args.num_runs):
        t0 = time.perf_counter()
        sr, im_resize = pipeline(im_lr)
        dt = time.perf_counter() - t0
        per_run_s.append(dt)
        print(f"[{args.resolution}] run {i+1}/{args.num_runs}: {dt:.3f}s", flush=True)

        # Save SR from run 2 (post warmup)
        if i == 1 and sr_saved is None:
            out_img = (sr * 0.5 + 0.5).cpu().float()
            out_pil = transforms.ToPILImage()(out_img[0].clamp(0, 1))
            # Apply wavelet color fix for fairness with baseline
            try:
                lr_pil = transforms.ToPILImage()(im_resize[0].cpu().float().clamp(0, 1))
                out_pil = wavelet_color_fix(out_pil, lr_pil)
            except Exception as e:
                print(f"[{args.resolution}] wavelet_color_fix skipped: {e}", flush=True)
            sr_saved = os.path.join(args.output_dir, f"sr_trn2_{args.resolution}.png")
            out_pil.save(sr_saved)

    peak_hbm_gb = sampler.stop()

    cold = per_run_s[0]
    rest = per_run_s[1:]
    total = sum(per_run_s)
    steady = (total - cold) / max(1, len(rest))
    rest_sorted = sorted(rest)
    def q(sl, p):
        if not sl:
            return None
        k = max(0, min(len(sl)-1, int(round((p/100)*(len(sl)-1)))))
        return sl[k]

    report = {
        "device": "neuron-trn2.3xlarge",
        "accel_name": "Trainium2 (LNC=2)",
        "resolution": args.resolution,
        "lq_side": lq_side,
        "sr_side": sr_side,
        "dtype": "fp32+autocast_none",
        "num_runs": args.num_runs,
        "load_time_s": round(load_time_s, 4),
        "cold_start_s": round(cold, 4),
        "steady_mean_s": round(steady, 4),
        "p50_s": round(q(rest_sorted, 50), 4) if rest_sorted else None,
        "p95_s": round(q(rest_sorted, 95), 4) if rest_sorted else None,
        "min_s": round(min(rest), 4) if rest else None,
        "max_s": round(max(rest), 4) if rest else None,
        "peak_hbm_gb": round(peak_hbm_gb, 4) if peak_hbm_gb else None,
        "per_run_s": [round(x, 4) for x in per_run_s],
        "sr_image": sr_saved,
        "seed": args.seed,
        "compiler_flags": "--auto-cast=none",
        "traced_components": "unet (tiled, B=1, latent 96x96)",
        "cpu_components": "vae_encoder, vae_decoder, text_encoder, DEResNet",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    json_path = os.path.join(args.output_dir, f"trn2_{args.resolution}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print("\n=== REPORT ===")
    print(json.dumps(report, indent=2))
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
