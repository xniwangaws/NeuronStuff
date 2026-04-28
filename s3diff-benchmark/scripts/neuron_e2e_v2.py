"""
End-to-end S3Diff on Neuron v2: UNet + VAE encoder + VAE decoder all traced.

Components:
  - DEResNet: CPU eager (tiny, 9 MB, not worth tracing)
  - Text encoder (CLIP): CPU eager (not worth tracing, 1-call overhead)
  - VAE encoder: TRACED at pixel tile 256x256 -> latent 32x32
  - UNet: TRACED at latent 96x96, CFG done in Python with 2 calls
  - VAE decoder: TRACED at latent 64x64 -> pixel 512x512

Tiling:
  - At 1K (LQ 256, c_t 1024): VAE encode 1024 -> 128 latent; we tile pixel 256
    into 4x4=16 tiles of 32-latent. Then UNet runs on 128 latent with 96/32 tiles = 2x2=4 tiles.
    Then VAE decode 128 latent -> 1024 pixel with latent 64 tiles = 2x2=4 tiles.

For bench simplicity we compute CFG (pos/neg) at UNet stage; VAE calls don't need CFG.
"""
import argparse
import gc
import json
import math
import os
import subprocess
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


BUCKETS = {"1K": 256, "2K": 512, "4K": 1024}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sd_path", default="/home/ubuntu/s3diff/models/sd-turbo")
    p.add_argument("--pretrained_path", default="/home/ubuntu/s3diff/models/S3Diff/s3diff.pkl")
    p.add_argument("--de_net_path", default="/home/ubuntu/s3diff/models/S3Diff/de_net.pth")
    p.add_argument("--traced_unet", default="/home/ubuntu/s3diff/neuron_out/unet_latent96.pt2")
    p.add_argument("--traced_vae_encoder", default="/home/ubuntu/s3diff/neuron_out/vae_encoder_256.pt2")
    p.add_argument("--traced_vae_decoder", default="/home/ubuntu/s3diff/neuron_out/vae_decoder_64.pt2")
    p.add_argument("--input_image", required=True)
    p.add_argument("--output_dir", default="/home/ubuntu/s3diff/bench_out")
    p.add_argument("--resolution", choices=["1K", "2K", "4K"], required=True)
    p.add_argument("--num_runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--unet_tile", type=int, default=96)
    p.add_argument("--unet_overlap", type=int, default=32)
    p.add_argument("--vae_enc_tile", type=int, default=256)
    p.add_argument("--vae_dec_tile", type=int, default=64)
    p.add_argument("--guidance_scale", type=float, default=1.07)
    p.add_argument("--pos_prompt", default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.")
    p.add_argument("--neg_prompt", default="oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth")
    return p.parse_args()


def gaussian_weights(tile, var=0.1):
    """Same as neuron_e2e.py (wider gaussian to avoid tile seams)."""
    from numpy import exp, pi
    mid = (tile - 1) / 2
    xs = np.array([exp(-(x - mid) ** 2 / (tile * tile) / (2 * var)) / math.sqrt(2 * pi * var)
                   for x in range(tile)])
    return torch.from_numpy(np.outer(xs, xs)).float()


def tiled_call(traced, x, tile_size, overlap, channels_out):
    """Generic tiled Python-driven dispatch of a compiled per-tile module.
    x: [1, C_in, H, W]. Tile on last 2 dims at tile_size (input units).
    Handles arbitrary scale ratio (encoder 8x downsample, decoder 8x upsample).
    """
    B, C_in, H, W = x.shape
    if H == tile_size and W == tile_size:
        return traced(x)

    def compute_grid(dim):
        count, cur = 0, 0
        while cur < dim:
            cur = max(count * tile_size - overlap * count, 0) + tile_size
            count += 1
        return count

    grid_rows = compute_grid(H)
    grid_cols = compute_grid(W)

    # Probe output: figure out scale ratio (output_side / input_side); may be fractional < 1
    probe = traced(x[:, :, :tile_size, :tile_size].contiguous())
    out_tile_h = probe.shape[-2]
    out_tile_w = probe.shape[-1]
    # Ratios (can be >1 for decoder, <1 for encoder)
    from fractions import Fraction
    ratio_h = Fraction(out_tile_h, tile_size)
    ratio_w = Fraction(out_tile_w, tile_size)
    out_H = int(Fraction(H) * ratio_h)
    out_W = int(Fraction(W) * ratio_w)

    out = torch.zeros((B, channels_out, out_H, out_W), dtype=probe.dtype)
    weight_sum = torch.zeros_like(out)
    # gaussian at OUTPUT tile size
    weights = gaussian_weights(out_tile_h).unsqueeze(0).unsqueeze(0).to(probe.dtype)

    probe_used = False
    for row in range(grid_rows):
        for col in range(grid_cols):
            if col < grid_cols - 1 or row < grid_rows - 1:
                ofs_x = max(row * tile_size - overlap * row, 0)
                ofs_y = max(col * tile_size - overlap * col, 0)
            if row == grid_rows - 1:
                ofs_x = H - tile_size
            if col == grid_cols - 1:
                ofs_y = W - tile_size
            ix0, ix1 = ofs_x, ofs_x + tile_size
            iy0, iy1 = ofs_y, ofs_y + tile_size
            tile = x[:, :, iy0:iy1, ix0:ix1].contiguous()
            if not probe_used and ix0 == 0 and iy0 == 0:
                tile_out = probe
                probe_used = True
            else:
                tile_out = traced(tile)
            # Map input coords to output coords
            ox0 = int(Fraction(ix0) * ratio_h)
            ox1 = ox0 + out_tile_h
            oy0 = int(Fraction(iy0) * ratio_w)
            oy1 = oy0 + out_tile_w
            out[:, :, oy0:oy1, ox0:ox1] += tile_out * weights
            weight_sum[:, :, oy0:oy1, ox0:ox1] += weights

    return out / weight_sum


def unet_tiled_cfg(traced_unet, latent, timesteps, pos_enc, neg_enc, guidance, tile_size, overlap):
    """UNet tiling with CFG (pos + neg per tile)."""
    _, C, H, W = latent.shape
    if H == tile_size and W == tile_size:
        pos = traced_unet(latent, timesteps, pos_enc)
        neg = traced_unet(latent, timesteps, neg_enc)
        return neg + guidance * (pos - neg)

    def compute_grid(dim):
        count, cur = 0, 0
        while cur < dim:
            cur = max(count * tile_size - overlap * count, 0) + tile_size
            count += 1
        return count

    grid_rows = compute_grid(W)
    grid_cols = compute_grid(H)
    weights = gaussian_weights(tile_size).to(latent.dtype).unsqueeze(0).unsqueeze(0)

    noise_pred = torch.zeros_like(latent)
    contrib = torch.zeros_like(latent)

    for row in range(grid_rows):
        for col in range(grid_cols):
            if col < grid_cols - 1 or row < grid_rows - 1:
                ofs_x = max(row * tile_size - overlap * row, 0)
                ofs_y = max(col * tile_size - overlap * col, 0)
            if row == grid_rows - 1:
                ofs_x = W - tile_size
            if col == grid_cols - 1:
                ofs_y = H - tile_size
            tile = latent[:, :, ofs_y:ofs_y+tile_size, ofs_x:ofs_x+tile_size].contiguous()
            pos = traced_unet(tile, timesteps, pos_enc)
            neg = traced_unet(tile, timesteps, neg_enc)
            cfg = neg + guidance * (pos - neg)
            noise_pred[:, :, ofs_y:ofs_y+tile_size, ofs_x:ofs_x+tile_size] += cfg * weights
            contrib[:, :, ofs_y:ofs_y+tile_size, ofs_x:ofs_x+tile_size] += weights

    return noise_pred / contrib


class NvidiaSmiLikeSampler:
    """Best-effort Neuron HBM sampler via neuron-ls or 'neuron-top -b -c 1'.
    Returns peak MiB observed. Fails gracefully."""
    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak_mib = 0
        self.running = False
        self.t = None

    def _loop(self):
        while self.running:
            try:
                out = subprocess.check_output(
                    ["bash", "-c", "neuron-ls --json-output 2>/dev/null"],
                    timeout=2,
                ).decode()
                import json as _j
                d = _j.loads(out)
                # Find mem_used per neuron_device
                total = 0
                for dev in d if isinstance(d, list) else [d]:
                    for nd in dev.get("nd_info", []) if isinstance(dev, dict) else []:
                        total += nd.get("mem_used_bytes", 0)
                mib = total // (1024 * 1024)
                if mib > self.peak_mib:
                    self.peak_mib = mib
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.peak_mib = 0
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def stop(self):
        self.running = False
        if self.t:
            self.t.join(timeout=3)
        return self.peak_mib / 1024.0  # GB


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

    # ---- Load ----
    print(f"[{args.resolution}] loading...", flush=True)
    t0 = time.perf_counter()

    s3 = argparse.Namespace(
        lora_rank_unet=32, lora_rank_vae=16,
        latent_tiled_size=args.unet_tile, latent_tiled_overlap=args.unet_overlap,
        vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224,
        padding_offset=32,
        pos_prompt=args.pos_prompt, neg_prompt=args.neg_prompt,
        sd_path=args.sd_path, pretrained_path=args.pretrained_path,
    )
    net_sr = S3Diff(
        lora_rank_unet=32, lora_rank_vae=16,
        sd_path=args.sd_path, pretrained_path=args.pretrained_path, args=s3,
    )
    net_sr.set_eval()

    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de.eval()

    traced_unet = torch.jit.load(args.traced_unet)
    traced_enc = torch.jit.load(args.traced_vae_encoder)
    traced_dec = torch.jit.load(args.traced_vae_decoder)

    load_s = time.perf_counter() - t0
    print(f"[{args.resolution}] loaded in {load_s:.2f}s", flush=True)

    # ---- Prepare LQ + bake de_mod (must match UNet's baked de_mod) ----
    im = Image.open(args.input_image).convert("RGB").resize((lq_side, lq_side), Image.BICUBIC)
    im_lr = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    im_lr = im_lr.unsqueeze(0)

    with torch.no_grad():
        deg = net_de(im_lr)
    bake_mod.compute_and_bake_de_mod(net_sr, deg)
    print(f"[{args.resolution}] deg_score: {deg.flatten().tolist()}", flush=True)

    # Text embeds precomputed
    with torch.no_grad():
        pos_tok = net_sr.tokenizer([args.pos_prompt], max_length=77, padding="max_length",
                                   truncation=True, return_tensors="pt").input_ids
        neg_tok = net_sr.tokenizer([args.neg_prompt], max_length=77, padding="max_length",
                                   truncation=True, return_tensors="pt").input_ids
        pos_enc = net_sr.text_encoder(pos_tok)[0]
        neg_enc = net_sr.text_encoder(neg_tok)[0]
    timesteps = torch.tensor([999], dtype=torch.long)

    def pipeline(im_lr):
        with torch.no_grad():
            # 1. x4 bicubic upsample
            ori_h, ori_w = im_lr.shape[2:]
            im_resize = F.interpolate(im_lr, size=(ori_h * 4, ori_w * 4), mode="bilinear", align_corners=False).contiguous()
            im_resize_norm = torch.clamp(im_resize * 2 - 1.0, -1.0, 1.0)
            resize_h, resize_w = im_resize_norm.shape[2:]
            pad_h = math.ceil(resize_h / 64) * 64 - resize_h
            pad_w = math.ceil(resize_w / 64) * 64 - resize_w
            im_resize_norm = F.pad(im_resize_norm, (0, pad_w, 0, pad_h), mode="reflect")

            # 2. Traced tiled VAE encoder
            # encoder wrapper returns scaled latent; input pixel -> latent /8
            lq_latent = tiled_call(
                traced_enc, im_resize_norm, args.vae_enc_tile, overlap=0,
                channels_out=4,
            )
            # Note: encoder tile uses overlap=0 since S3Diff's vae encoder tile hook doesn't overlap either

            # 3. Traced tiled UNet with CFG
            model_pred = unet_tiled_cfg(
                traced_unet, lq_latent, timesteps, pos_enc, neg_enc,
                args.guidance_scale, args.unet_tile, args.unet_overlap,
            )

            # 4. Scheduler step (CPU, very fast)
            x_denoised = net_sr.sched.step(model_pred, net_sr.timesteps, lq_latent, return_dict=True).prev_sample

            # 5. Traced tiled VAE decoder.  Input is scaled latent; decoder wrapper divides by scaling internally.
            # But our decoder wrapper expects scaled latent (it does z/scaling internally).
            out = tiled_call(
                traced_dec, x_denoised, args.vae_dec_tile, overlap=0,
                channels_out=3,
            )
            out = out.clamp(-1, 1)
            out = out[:, :, :resize_h, :resize_w]
        return out, im_resize

    # ---- Timed runs ----
    sampler = NvidiaSmiLikeSampler()
    sampler.start()
    per_run = []
    sr_saved = None
    for i in range(args.num_runs):
        t0 = time.perf_counter()
        sr, im_resize = pipeline(im_lr)
        dt = time.perf_counter() - t0
        per_run.append(dt)
        print(f"[{args.resolution}] run {i+1}/{args.num_runs}: {dt:.3f}s", flush=True)

        if i == 1 and sr_saved is None:
            out_img = (sr * 0.5 + 0.5).cpu().float()
            pil = transforms.ToPILImage()(out_img[0].clamp(0, 1))
            try:
                lr_pil = transforms.ToPILImage()(im_resize[0].cpu().float().clamp(0, 1))
                pil = wavelet_color_fix(pil, lr_pil)
            except Exception:
                pass
            sr_saved = os.path.join(args.output_dir, f"sr_trn2v2_{args.resolution}.png")
            pil.save(sr_saved)

    peak_hbm_gb = sampler.stop()

    cold = per_run[0]
    rest = per_run[1:]
    steady = (sum(per_run) - cold) / max(1, len(rest))
    rest_s = sorted(rest)
    def q(s, p):
        if not s: return None
        k = max(0, min(len(s)-1, int(round((p/100)*(len(s)-1)))))
        return s[k]

    report = {
        "device": "neuron-trn2.3xlarge",
        "accel_name": "Trainium2 (LNC=2), fully-traced",
        "resolution": args.resolution,
        "lq_side": lq_side,
        "sr_side": sr_side,
        "dtype": "fp32+autocast_none",
        "num_runs": args.num_runs,
        "load_time_s": round(load_s, 4),
        "cold_start_s": round(cold, 4),
        "steady_mean_s": round(steady, 4),
        "p50_s": round(q(rest_s, 50), 4) if rest_s else None,
        "p95_s": round(q(rest_s, 95), 4) if rest_s else None,
        "min_s": round(min(rest), 4) if rest else None,
        "max_s": round(max(rest), 4) if rest else None,
        "peak_hbm_gb": round(peak_hbm_gb, 4) if peak_hbm_gb else None,
        "per_run_s": [round(x, 4) for x in per_run],
        "sr_image": sr_saved,
        "compiler_flags": "--auto-cast=none",
        "traced_components": "unet (tile 96x96) + vae_encoder (tile 256x256) + vae_decoder (tile 64x64)",
        "cpu_components": "text_encoder, DEResNet, scheduler_step",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json_path = os.path.join(args.output_dir, f"trn2v2_{args.resolution}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
