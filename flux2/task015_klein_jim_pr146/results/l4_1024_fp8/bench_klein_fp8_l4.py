"""
Bench FLUX.2-klein-9b-fp8 on L4 using BFL official flux2 repo + custom FP8 shim.

Uses the guidance-distilled klein-9b FP8 checkpoint (4-step default).
NOTE: we override num_steps to 50 per bench spec (may produce distillation artifacts).
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image

sys.path.insert(0, "/home/ubuntu")
from klein_fp8_shim import convert_model_to_fp8, load_fp8_state_dict, FP8Linear

# BFL flux2 imports
from flux2.model import Flux2, Klein9BParams
from flux2.sampling import (
    batched_wrapper,
    denoise,
    get_schedule,
    prc_img,
    prc_txt,
    scatter_ids,
)
from flux2.util import load_ae, load_text_encoder


def peak_vram_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3


def reset_vram():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load_klein_9b_fp8(ckpt_path: str, device: str = "cuda") -> Flux2:
    # Build model on meta (no memory), swap Linears on meta (FP8Linear alloc on meta too),
    # then materialize the (much smaller) post-swap tree onto CUDA.
    print("[fp8] Constructing Flux2(Klein9BParams) on meta device")
    with torch.device("meta"):
        model = Flux2(Klein9BParams()).to(torch.bfloat16)

    print("[fp8] Swapping nn.Linear -> FP8Linear on meta")
    n_rep, n_kept = convert_model_to_fp8(model, device="meta")
    print(f"[fp8]   replaced={n_rep}  kept_as_bf16={n_kept}")

    print(f"[fp8] Materializing model on {device} (to_empty)")
    model = model.to_empty(device=device)
    with torch.no_grad():
        for m in model.modules():
            for p in m.parameters(recurse=False):
                p.data.zero_()
            for _, b in m.named_buffers(recurse=False):
                b.data.zero_()

    print(f"[fp8] Loading FP8 weights from {ckpt_path}")
    missing, unexpected = load_fp8_state_dict(model, ckpt_path, device=device)
    if missing:
        print(f"[fp8] WARN missing {len(missing)}: {missing[:10]}")
    if unexpected:
        print(f"[fp8] WARN unexpected {len(unexpected)}: {unexpected[:10]}")
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="A cat holding a sign that says hello world")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = "cuda"

    # ---- Load text encoder, encode the (shared) prompt, offload ----
    print("[stage] Loading Qwen3-8B text encoder...")
    reset_vram()
    t0 = time.perf_counter()
    text_encoder = load_text_encoder("flux.2-klein-9b", device=device)
    t_te_load = time.perf_counter() - t0
    print(f"[stage] text_encoder loaded in {t_te_load:.1f}s peak={peak_vram_gb():.2f}GB")

    print(f"[stage] Encoding prompt: {args.prompt!r}")
    with torch.inference_mode():
        ctx_cache = text_encoder([args.prompt]).to(torch.bfloat16).cpu()
    print(f"[stage] prompt encoded ctx={tuple(ctx_cache.shape)}")

    text_encoder = text_encoder.cpu()
    del text_encoder
    gc.collect(); torch.cuda.empty_cache()
    print(f"[stage] after TE offload peak={peak_vram_gb():.2f}GB")

    # ---- Load AE ----
    print("[stage] Loading autoencoder...")
    reset_vram()
    t0 = time.perf_counter()
    ae = load_ae("flux.2-klein-9b", device=device)
    # Cast AE to bf16 — default is fp32 which OOMs at 2K (intermediate [1,128,2048,2048] = 4GB)
    ae = ae.to(torch.bfloat16)
    t_ae_load = time.perf_counter() - t0
    ae.eval()
    print(f"[stage] ae loaded in {t_ae_load:.1f}s peak={peak_vram_gb():.2f}GB")

    # ---- Load FP8 transformer ----
    print("[stage] Loading FP8 transformer...")
    reset_vram()
    t0 = time.perf_counter()
    model = load_klein_9b_fp8(args.ckpt, device=device)
    t_fp8_load = time.perf_counter() - t0
    n_fp8 = sum(isinstance(m, FP8Linear) for m in model.modules())
    # Force final cleanup so the caching allocator releases transient load buffers
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    print(f"[stage] FP8 model loaded in {t_fp8_load:.1f}s "
          f"fp8_layers={n_fp8} alloc={torch.cuda.memory_allocated()/1024**3:.2f}GB "
          f"free={torch.cuda.mem_get_info()[0]/1024**3:.2f}GB")

    # ---- Bench loop ----
    batched_prc_img_l = batched_wrapper(prc_img)
    batched_prc_txt_l = batched_wrapper(prc_txt)

    results = []
    for seed in args.seeds:
        print(f"\n[bench] seed={seed} res={args.width}x{args.height} steps={args.steps}")
        reset_vram()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        ctx = ctx_cache.to(device)
        ctx, ctx_ids = batched_prc_txt_l(ctx)

        shape = (1, 128, args.height // 16, args.width // 16)
        generator = torch.Generator(device=device).manual_seed(seed)
        randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device=device)
        x, x_ids = batched_prc_img_l(randn)

        timesteps = get_schedule(args.steps, x.shape[1])
        with torch.inference_mode():
            x = denoise(model, x, x_ids, ctx, ctx_ids,
                        timesteps=timesteps, guidance=args.guidance)
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        # Free intermediate activations before AE decode — decode has its own working set.
        torch.cuda.empty_cache()
        with torch.inference_mode():
            x = ae.decode(x).float()
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak = peak_vram_gb()
        out_path = Path(args.out) / f"seed{seed}_cat.png"
        img.save(out_path, quality=95)

        arr = np.array(img)
        std = float(arr.std())
        pass_ok = 10.0 < std < 120.0
        print(f"[bench] seed={seed} elapsed={elapsed:.2f}s peak={peak:.2f}GB "
              f"std={std:.2f} pass={pass_ok} saved={out_path}")
        results.append({"seed": seed, "elapsed_s": elapsed, "peak_gb": peak,
                        "std": std, "pass": pass_ok, "file": str(out_path)})

    import statistics
    ok = [r for r in results if r["pass"]]
    all_t = [r["elapsed_s"] for r in results]
    ok_t = [r["elapsed_s"] for r in ok]
    summary = {
        "run": "klein-9b-fp8 (guidance-distilled) @ L4 BFL-flux2 + fp8-shim",
        "prompt": args.prompt, "res": f"{args.width}x{args.height}",
        "num_steps": args.steps, "guidance": args.guidance,
        "seeds": args.seeds,
        "n_total": len(results), "n_pass": len(ok),
        "mean_s_all": statistics.mean(all_t) if all_t else 0.0,
        "mean_s_pass": statistics.mean(ok_t) if ok_t else 0.0,
        "peak_gb_max": max(r["peak_gb"] for r in results) if results else 0.0,
        "stage_times": {"text_encoder_load_s": t_te_load,
                        "ae_load_s": t_ae_load,
                        "fp8_transformer_load_s": t_fp8_load},
        "runs": results,
    }
    json_path = Path(args.out) / "results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] {json_path}")
    print(f"[done] mean_all={summary['mean_s_all']:.2f}s "
          f"mean_pass={summary['mean_s_pass']:.2f}s "
          f"peak={summary['peak_gb_max']:.2f}GB "
          f"pass={len(ok)}/{len(results)}")


if __name__ == "__main__":
    main()
