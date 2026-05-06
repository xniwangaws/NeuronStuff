#!/usr/bin/env python3
"""FLUX.1-dev L4 FP8 bench @ 4096^2 using wangkanai/flux-dev-fp8 ckpt.

Sequential CPU offload required (17GB bf16 upcast > L4 22GB).
"""
import argparse, json, os, time
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

ap = argparse.ArgumentParser()
ap.add_argument("--resolution", type=int, default=4096)
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
ap.add_argument("--steps", type=int, default=28)
ap.add_argument("--guidance", type=float, default=3.5)
ap.add_argument("--out", default="/home/ubuntu/flux1_alien_l4_fp8_4096")
ap.add_argument("--model", default="wangkanai/flux-dev-fp8",
                help="ComfyUI-style single-file ckpt (FP8 E4M3 weights)")
ap.add_argument("--ckpt_file", default="flux1-dev-fp8-e4m3fn.safetensors")
ap.add_argument("--base_model", default="black-forest-labs/FLUX.1-dev",
                help="for text encoders + VAE + scheduler configs")
args = ap.parse_args()

PROMPT = "A close-up image of a green alien with fluorescent skin in the middle of a dark purple forest"
H = W = args.resolution
os.makedirs(args.out, exist_ok=True)

print(f"[load] FLUX.1-dev FP8 (wangkanai) @ {H}x{W}")
t0 = time.time()
transformer = FluxTransformer2DModel.from_single_file(
    f"https://huggingface.co/{args.model}/resolve/main/{args.ckpt_file}",
    torch_dtype=torch.bfloat16,
    config=args.base_model, subfolder="transformer",
)
pipe = FluxPipeline.from_pretrained(
    args.base_model, transformer=transformer, torch_dtype=torch.bfloat16,
)
pipe.enable_sequential_cpu_offload()
print(f"[load] done {time.time()-t0:.1f}s")

runs = []
peak = 0.0
for seed in args.seeds:
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    try:
        img = pipe(PROMPT, height=H, width=W, guidance_scale=args.guidance,
                   num_inference_steps=args.steps, max_sequence_length=512,
                   generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    except torch.cuda.OutOfMemoryError as e:
        msg = str(e)[:200]
        print(f"[seed {seed}] OOM: {msg}")
        runs.append({"seed": seed, "status": "OOM", "error": msg})
        torch.cuda.empty_cache(); continue
    except Exception as e:
        msg = str(e)[:200]
        print(f"[seed {seed}] ERROR: {msg}")
        runs.append({"seed": seed, "status": "error", "error": msg})
        torch.cuda.empty_cache(); continue
    e2e = time.time() - t
    pk = torch.cuda.max_memory_allocated() / 1024**3
    peak = max(peak, pk)
    import numpy as np
    arr = np.asarray(img).astype(np.float32)
    std_v = float(arr.std())
    png = os.path.join(args.out, f"seed{seed}_alien.png")
    img.save(png)
    runs.append({"seed": seed, "time_s": round(e2e, 2), "peak_vram_gb": round(pk, 2),
                 "std": round(std_v, 2), "png": png, "status": "ok"})
    print(f"[seed {seed}] {e2e:.2f}s peak={pk:.2f}GB std={std_v:.2f}")

ok = [r for r in runs if r.get("status") == "ok"]
summary = {
    "device": f"L4 g6.4xlarge FP8 (wangkanai, seq-offload) {args.resolution}x{args.resolution}",
    "model": "FLUX.1-dev",
    "prompt": PROMPT, "steps": args.steps, "guidance": args.guidance,
    "resolution": f"{H}x{W}",
    "peak_vram_gb": round(peak, 2),
    "mean_s": round(sum(r["time_s"] for r in ok)/max(1, len(ok)), 2) if ok else None,
    "n_ok": len(ok), "n_total": len(runs),
    "runs": runs,
}
with open(os.path.join(args.out, "results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"[done] mean={summary['mean_s']}s peak={peak:.2f}GB {summary['n_ok']}/{summary['n_total']} ok")
