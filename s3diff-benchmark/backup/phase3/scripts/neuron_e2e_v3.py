"""Phase 3 e2e: 1K single-tile UNet + traced VAE enc/dec.

Uses Plan B attribute-routed UNet (NEFF accepts unet_de_mods as input).
No tile seams at 1K because the entire latent (128x128) is one UNet call.

Usage:
  python phase3/neuron_e2e_v3.py --resolution 1K --num_runs 10 \
    --input_image /home/ubuntu/s3diff/smoke_in/cat_LQ_256.png
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
# Latent side per bucket (c_t = 4*lq, latent = c_t/8)
LATENT_SIDE = {"1K": 128, "2K": 256, "4K": 512}
# Which traced UNet NEFF to use.
# Phase 3 finding: UNet at latent 256 exceeds 5M instruction limit (26M generated).
# So 2K/4K reuse the 128-latent NEFF and tile the latent grid.
# At 2K latent is 256x256 -> 2x2 tiles of 128x128 (no overlap — tile cleanly aligned).
# At 4K latent is 512x512 -> 4x4 tiles.
UNET_NEFF = {
    "1K": "/home/ubuntu/s3diff/neuron_out/unet_1K_v3b.pt2",
    "2K": "/home/ubuntu/s3diff/neuron_out/unet_1K_v3b.pt2",  # reuse; tile
    "4K": "/home/ubuntu/s3diff/neuron_out/unet_1K_v3b.pt2",  # reuse; tile
}
# UNet tile configuration per bucket
UNET_TILE = {"1K": (128, 0), "2K": (128, 32), "4K": (128, 32)}  # (tile, overlap)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sd_path", default="/home/ubuntu/s3diff/models/sd-turbo")
    p.add_argument("--pretrained_path", default="/home/ubuntu/s3diff/models/S3Diff/s3diff.pkl")
    p.add_argument("--de_net_path", default="/home/ubuntu/s3diff/models/S3Diff/de_net.pth")
    p.add_argument("--traced_vae_encoder", default="/home/ubuntu/s3diff/neuron_out/vae_encoder_256.pt2")
    p.add_argument("--traced_vae_decoder", default="/home/ubuntu/s3diff/neuron_out/vae_decoder_64.pt2")
    p.add_argument("--input_image", required=True)
    p.add_argument("--output_dir", default="/home/ubuntu/s3diff/bench_out")
    p.add_argument("--resolution", choices=["1K", "2K", "4K"], required=True)
    p.add_argument("--num_runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--vae_enc_tile", type=int, default=256)
    p.add_argument("--vae_dec_tile", type=int, default=64)
    p.add_argument("--guidance_scale", type=float, default=1.07)
    p.add_argument("--pos_prompt", default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.")
    p.add_argument("--neg_prompt", default="oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth")
    return p.parse_args()


def compute_de_mods(net_sr, deg_score):
    """Return (unet_stack, vae_stack)."""
    deg_proj = deg_score[..., None] * net_sr.W[None, None, :] * 2 * np.pi
    deg_proj = torch.cat([torch.sin(deg_proj), torch.cos(deg_proj)], dim=-1)
    deg_proj = torch.cat([deg_proj[:, 0], deg_proj[:, 1]], dim=-1)
    vae_de_c = net_sr.vae_de_mlp(deg_proj)
    unet_de_c = net_sr.unet_de_mlp(deg_proj)
    vae_block_c = net_sr.vae_block_mlp(net_sr.vae_block_embeddings.weight)
    unet_block_c = net_sr.unet_block_mlp(net_sr.unet_block_embeddings.weight)
    vae_embeds = net_sr.vae_fuse_mlp(torch.cat([
        vae_de_c.unsqueeze(1).repeat(1, vae_block_c.shape[0], 1),
        vae_block_c.unsqueeze(0).repeat(vae_de_c.shape[0], 1, 1),
    ], -1))
    unet_embeds = net_sr.unet_fuse_mlp(torch.cat([
        unet_de_c.unsqueeze(1).repeat(1, unet_block_c.shape[0], 1),
        unet_block_c.unsqueeze(0).repeat(unet_de_c.shape[0], 1, 1),
    ], -1))

    unet_list = []
    for name in net_sr.unet_lora_layers:
        split = name.split(".")
        if split[0] == "down_blocks":
            emb = unet_embeds[:, int(split[1])]
        elif split[0] == "mid_block":
            emb = unet_embeds[:, 4]
        elif split[0] == "up_blocks":
            emb = unet_embeds[:, int(split[1]) + 5]
        else:
            emb = unet_embeds[:, -1]
        unet_list.append(emb.reshape(-1, net_sr.lora_rank_unet, net_sr.lora_rank_unet))
    unet_stack = torch.stack(unet_list, dim=0)

    vae_list = []
    for name in net_sr.vae_lora_layers:
        split = name.split(".")
        if split[1] == "down_blocks":
            emb = vae_embeds[:, int(split[2])]
        elif split[1] == "mid_block":
            emb = vae_embeds[:, -2]
        else:
            emb = vae_embeds[:, -1]
        vae_list.append(emb.reshape(-1, net_sr.lora_rank_vae, net_sr.lora_rank_vae))
    vae_stack = torch.stack(vae_list, dim=0)
    return unet_stack, vae_stack


def tiled_call(traced, x, tile_size, channels_out, overlap=0, blend_var=0.3):
    """Tiled dispatch of a traced module that expects fixed tile shape.
    Supports encoder (pixel->latent, 8x down) and decoder (latent->pixel, 8x up).

    When overlap > 0, tiles overlap by `overlap` pixels and are blended with
    a 2D gaussian weight (var=blend_var) in output coordinates. This eliminates
    visible tile boundaries that appear from per-tile numerical drift.

    overlap=0 preserves the simple-stitch behavior (no overlap, no blend).
    """
    B, C, H, W = x.shape
    if H == tile_size and W == tile_size:
        return traced(x)

    # Probe to determine scale
    probe = traced(x[:, :, :tile_size, :tile_size].contiguous())
    scale_h = probe.shape[-2] / tile_size
    scale_w = probe.shape[-1] / tile_size
    out_tile_h = probe.shape[-2]
    out_tile_w = probe.shape[-1]
    out_H = int(H * scale_h)
    out_W = int(W * scale_w)

    if overlap == 0:
        # Simple-stitch path (original behavior)
        def grid(d, t):
            c, cur = 0, 0
            while cur < d:
                cur = c * t + t
                c += 1
            return c
        gr = grid(H, tile_size)
        gc = grid(W, tile_size)
        out = torch.zeros((B, channels_out, out_H, out_W), dtype=probe.dtype)
        probe_used = False
        for r in range(gr):
            for c in range(gc):
                iy0 = min(r * tile_size, H - tile_size)
                ix0 = min(c * tile_size, W - tile_size)
                tile = x[:, :, iy0:iy0+tile_size, ix0:ix0+tile_size].contiguous()
                if not probe_used and iy0 == 0 and ix0 == 0:
                    out_tile = probe
                    probe_used = True
                else:
                    out_tile = traced(tile)
                oy0 = int(iy0 * scale_h)
                ox0 = int(ix0 * scale_w)
                out[:, :, oy0:oy0+out_tile_h, ox0:ox0+out_tile_w] = out_tile
        return out

    # Overlapped gaussian-blend path (for seam elimination on VAE decoder)
    stride = tile_size - overlap

    def compute_grid(dim):
        # Number of tiles with given stride to cover `dim`
        count, cur = 0, 0
        while cur < dim:
            cur = max(count * tile_size - overlap * count, 0) + tile_size
            count += 1
        return count
    gr = compute_grid(H)
    gc = compute_grid(W)

    # Gaussian weight at OUTPUT tile size (weights applied in output coordinates)
    from numpy import exp, pi
    mid_h = (out_tile_h - 1) / 2
    mid_w = (out_tile_w - 1) / 2
    ys = np.array([exp(-(y - mid_h) ** 2 / (out_tile_h * out_tile_h) / (2 * blend_var)) / math.sqrt(2 * pi * blend_var) for y in range(out_tile_h)])
    xs = np.array([exp(-(xx - mid_w) ** 2 / (out_tile_w * out_tile_w) / (2 * blend_var)) / math.sqrt(2 * pi * blend_var) for xx in range(out_tile_w)])
    w2d = torch.from_numpy(np.outer(ys, xs)).to(probe.dtype).unsqueeze(0).unsqueeze(0)

    accum = torch.zeros((B, channels_out, out_H, out_W), dtype=probe.dtype)
    weight_sum = torch.zeros((B, 1, out_H, out_W), dtype=probe.dtype)

    probe_used = False
    for r in range(gr):
        for c in range(gc):
            # Input tile offset (same indexing as unet_tiled_cfg)
            if c < gc - 1 or r < gr - 1:
                iy0 = max(r * tile_size - overlap * r, 0)
                ix0 = max(c * tile_size - overlap * c, 0)
            if r == gr - 1:
                iy0 = H - tile_size
            if c == gc - 1:
                ix0 = W - tile_size
            tile = x[:, :, iy0:iy0+tile_size, ix0:ix0+tile_size].contiguous()
            if not probe_used and iy0 == 0 and ix0 == 0:
                out_tile = probe
                probe_used = True
            else:
                out_tile = traced(tile)
            oy0 = int(iy0 * scale_h)
            ox0 = int(ix0 * scale_w)
            oy1 = oy0 + out_tile_h
            ox1 = ox0 + out_tile_w
            accum[:, :, oy0:oy1, ox0:ox1] += out_tile * w2d
            weight_sum[:, :, oy0:oy1, ox0:ox1] += w2d

    return accum / weight_sum


def unet_1tile_cfg(traced, latent, t, enc_pos, enc_neg, de_mods, guidance):
    """Single-tile UNet with CFG: 2 traced calls (pos, neg), CFG combine."""
    pos = traced(latent, t, enc_pos, de_mods)
    neg = traced(latent, t, enc_neg, de_mods)
    return neg + guidance * (pos - neg)


def unet_tiled_cfg(traced, latent, t, enc_pos, enc_neg, de_mods, guidance, tile_size, overlap, var=0.1):
    """Tiled UNet CFG (for 4K). Gaussian blend with wider var=0.1."""
    _, _, H, W = latent.shape
    if H == tile_size and W == tile_size:
        return unet_1tile_cfg(traced, latent, t, enc_pos, enc_neg, de_mods, guidance)

    def compute_grid(dim):
        count, cur = 0, 0
        while cur < dim:
            cur = max(count * tile_size - overlap * count, 0) + tile_size
            count += 1
        return count

    gr = compute_grid(H)
    gc = compute_grid(W)

    # Wide gaussian
    from numpy import exp, pi
    mid = (tile_size - 1) / 2
    xs = np.array([exp(-(x - mid) ** 2 / (tile_size * tile_size) / (2 * var)) / math.sqrt(2 * pi * var) for x in range(tile_size)])
    w = torch.from_numpy(np.outer(xs, xs)).float().unsqueeze(0).unsqueeze(0)

    noise_pred = torch.zeros_like(latent)
    contrib = torch.zeros_like(latent)
    for r in range(gr):
        for c in range(gc):
            if c < gc - 1 or r < gr - 1:
                oy = max(r * tile_size - overlap * r, 0)
                ox = max(c * tile_size - overlap * c, 0)
            if r == gr - 1:
                oy = H - tile_size
            if c == gc - 1:
                ox = W - tile_size
            tile = latent[:, :, oy:oy+tile_size, ox:ox+tile_size].contiguous()
            cfg = unet_1tile_cfg(traced, tile, t, enc_pos, enc_neg, de_mods, guidance)
            noise_pred[:, :, oy:oy+tile_size, ox:ox+tile_size] += cfg * w
            contrib[:, :, oy:oy+tile_size, ox:ox+tile_size] += w
    return noise_pred / contrib


class HbmSampler:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak_mib = 0
        self.running = False
    def _loop(self):
        while self.running:
            try:
                out = subprocess.check_output(
                    ["bash", "-c", "neuron-ls --json-output 2>/dev/null"],
                    timeout=2,
                ).decode()
                import json as _j
                d = _j.loads(out)
                total = 0
                for dev in (d if isinstance(d, list) else [d]):
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
        if self.t: self.t.join(timeout=3)
        return self.peak_mib / 1024.0


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    from s3diff_tile import S3Diff
    from de_net import DEResNet
    from utils.wavelet_color import wavelet_color_fix

    lq_side = BUCKETS[args.resolution]
    latent_side = LATENT_SIDE[args.resolution]
    sr_side = lq_side * 4

    print(f"[{args.resolution}] loading...", flush=True)
    t0 = time.perf_counter()
    s3 = argparse.Namespace(
        lora_rank_unet=32, lora_rank_vae=16,
        latent_tiled_size=128, latent_tiled_overlap=32,
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

    traced_unet = torch.jit.load(UNET_NEFF[args.resolution])
    traced_enc = torch.jit.load(args.traced_vae_encoder)
    traced_dec = torch.jit.load(args.traced_vae_decoder)
    load_s = time.perf_counter() - t0
    print(f"[{args.resolution}] loaded in {load_s:.2f}s", flush=True)

    # Prepare input
    im = Image.open(args.input_image).convert("RGB").resize((lq_side, lq_side), Image.BICUBIC)
    im_lr = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    im_lr = im_lr.unsqueeze(0)
    with torch.no_grad():
        deg = net_de(im_lr)
    unet_de_mods, vae_de_mods = compute_de_mods(net_sr, deg)
    print(f"[{args.resolution}] deg_score: {deg.flatten().tolist()}", flush=True)

    with torch.no_grad():
        pos_tok = net_sr.tokenizer([args.pos_prompt], max_length=77, padding="max_length", truncation=True, return_tensors="pt").input_ids
        neg_tok = net_sr.tokenizer([args.neg_prompt], max_length=77, padding="max_length", truncation=True, return_tensors="pt").input_ids
        pos_enc = net_sr.text_encoder(pos_tok)[0]
        neg_enc = net_sr.text_encoder(neg_tok)[0]
    timesteps = torch.tensor([999], dtype=torch.long)

    def pipeline(im_lr):
        with torch.no_grad():
            ori_h, ori_w = im_lr.shape[2:]
            im_resize = F.interpolate(im_lr, size=(ori_h * 4, ori_w * 4), mode="bilinear", align_corners=False).contiguous()
            im_resize_norm = torch.clamp(im_resize * 2 - 1.0, -1.0, 1.0)
            resize_h, resize_w = im_resize_norm.shape[2:]
            pad_h = math.ceil(resize_h / 64) * 64 - resize_h
            pad_w = math.ceil(resize_w / 64) * 64 - resize_w
            im_resize_norm = F.pad(im_resize_norm, (0, pad_w, 0, pad_h), mode="reflect")

            # VAE encoder (tiled at 256 pixel tiles)
            lq_latent = tiled_call(traced_enc, im_resize_norm, args.vae_enc_tile, channels_out=4)

            # UNet
            tile, overlap = UNET_TILE[args.resolution]
            if args.resolution == "1K":
                model_pred = unet_1tile_cfg(traced_unet, lq_latent, timesteps, pos_enc, neg_enc, unet_de_mods, args.guidance_scale)
            else:
                model_pred = unet_tiled_cfg(traced_unet, lq_latent, timesteps, pos_enc, neg_enc, unet_de_mods, args.guidance_scale, tile_size=tile, overlap=overlap)

            # Scheduler step on CPU
            x_den = net_sr.sched.step(model_pred, net_sr.timesteps, lq_latent, return_dict=True).prev_sample

            # VAE decoder (tiled at 64 latent tiles)
            # P2-B outcome (documented 2026-04-29):
            # Gaussian blend with overlap=16 cuts row luma seam 73 → 39,
            # overlap=32 plateaus at 37.8 (PSNR gain: 24.24 → 24.56, essentially flat).
            # Root cause is UNet/VAE numerical drift at tile boundaries, not the blend.
            # We use overlap=16 as a balance: -46% seam for +1.6x latency.
            # Set overlap=0 to restore fastest path (12.5s at 1K) — seam visible.
            out = tiled_call(traced_dec, x_den, args.vae_dec_tile, channels_out=3,
                             overlap=16, blend_var=0.3)
            out = out.clamp(-1, 1)[:, :, :resize_h, :resize_w]
        return out, im_resize

    # Timed runs
    sampler = HbmSampler()
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
            sr_saved = os.path.join(args.output_dir, f"sr_trn2v3_{args.resolution}.png")
            pil.save(sr_saved)

    peak_hbm_gb = sampler.stop()

    cold = per_run[0]
    rest = per_run[1:]
    steady = (sum(per_run) - cold) / max(1, len(rest))
    rest_s = sorted(rest)
    q = lambda s, p: s[max(0, min(len(s)-1, int(round((p/100)*(len(s)-1)))))] if s else None

    report = {
        "device": "neuron-trn2",
        "accel_name": "Trainium2 (LNC=2), Phase 3 full-traced",
        "resolution": args.resolution,
        "lq_side": lq_side,
        "sr_side": sr_side,
        "latent_side": latent_side,
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
        "traced_components": f"unet (attr-routed, full latent {latent_side}), vae_encoder (tile 256), vae_decoder (tile 64)",
        "cpu_components": "text_encoder, DEResNet, scheduler_step",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json_path = os.path.join(args.output_dir, f"trn2v3_{args.resolution}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
