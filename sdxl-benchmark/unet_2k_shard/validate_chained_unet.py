"""Numerical validation: ChainedUNet (Neuron NEFFs) vs full SDXL UNet (CPU bf16).

Computes cosine similarity, max abs diff. Also measures cold + 5 warm latency.
Saves results to /home/ubuntu/work_e2e/UNET_2K_VALIDATION.json
"""
import os, sys, time, json, math, traceback
sys.path.insert(0, "/home/ubuntu/work_e2e/scripts")

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from chained_unet import ChainedUNet

DTYPE = torch.bfloat16
LATENT = 256
B = 1
NEFF_DIR = "/home/ubuntu/work_e2e/unet_neff"
OUT_JSON = "/home/ubuntu/work_e2e/UNET_2K_VALIDATION.json"


def cos_sim(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def main():
    torch.manual_seed(0)

    # Inputs
    sample = torch.randn(B, 4, LATENT, LATENT, dtype=DTYPE)
    timestep = torch.tensor([999.0])
    ehs = torch.randn(B, 77, 2048, dtype=DTYPE)
    text_embeds = torch.randn(B, 1280, dtype=DTYPE)
    time_ids = torch.randn(B, 6, dtype=DTYPE)

    # Reference: full UNet on CPU bf16
    print("[ref] loading full UNet...", flush=True)
    unet_cpu = UNet2DConditionModel.from_pretrained(
        "/home/ubuntu/sdxl-base", subfolder="unet", variant="fp16",
        torch_dtype=DTYPE, low_cpu_mem_usage=True,
    )
    unet_cpu.eval()
    print("[ref] running CPU forward...", flush=True)
    with torch.no_grad():
        ref_out = unet_cpu(
            sample, timestep, ehs,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )[0]
    print(f"[ref] out shape={tuple(ref_out.shape)}", flush=True)

    # Pre-compute t_emb (host-side, fp32) so it matches what stem expects
    t_emb_in = unet_cpu.time_proj(timestep).to(DTYPE)

    # Free CPU UNet
    del unet_cpu

    # Neuron chained
    print("[neuron] loading ChainedUNet...", flush=True)
    chained = ChainedUNet(NEFF_DIR)
    print("[neuron] cold run...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        neu_out = chained(sample, t_emb_in, text_embeds, time_ids, ehs)
    cold = time.time() - t0
    print(f"[neuron] cold {cold*1000:.1f} ms, out shape={tuple(neu_out.shape)}", flush=True)

    # Warm runs
    warm_times = []
    for i in range(5):
        t0 = time.time()
        with torch.no_grad():
            _ = chained(sample, t_emb_in, text_embeds, time_ids, ehs)
        warm_times.append(time.time() - t0)
    print(f"[neuron] warm {[f'{t*1000:.1f}' for t in warm_times]} ms", flush=True)

    # Metrics
    cs = cos_sim(ref_out, neu_out)
    mad = float((ref_out - neu_out).abs().max())
    mean_abs_ref = float(ref_out.abs().mean())
    rel = mad / (mean_abs_ref + 1e-12)
    print(f"[metrics] cos_sim={cs:.6f} max_abs_diff={mad:.4f} mean_abs_ref={mean_abs_ref:.4f} rel={rel:.4f}", flush=True)

    sizes = {}
    for n in [
        "p0_stem","p1_down0","p2_down1","p3_down2","p4_mid",
        "p5_up0","p6_up1","p7_up2","p8_head",
    ]:
        p = os.path.join(NEFF_DIR, f"traced_{n}.pt")
        if os.path.exists(p):
            sizes[n] = round(os.path.getsize(p) / 1e6, 1)
        else:
            sizes[n] = None

    result = {
        "approach": "per-block UNet trace, 9 sub-NEFFs, lnc=2 single-core",
        "latent_shape": list(sample.shape),
        "image_shape": [B, 3, LATENT * 8, LATENT * 8],
        "cold_ms": cold * 1000,
        "warm_ms": [t * 1000 for t in warm_times],
        "warm_mean_ms": sum(warm_times) / len(warm_times) * 1000,
        "warm_min_ms": min(warm_times) * 1000,
        "cos_sim": cs,
        "max_abs_diff": mad,
        "mean_abs_ref": mean_abs_ref,
        "rel_max_abs_diff": rel,
        "neff_sizes_mb": sizes,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] wrote {OUT_JSON}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
