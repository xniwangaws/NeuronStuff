#!/usr/bin/env python3
"""SDXL H100 FP8 + torch.compile (reduce-overhead + CUDA graphs) benchmark.

Tests whether torch.compile rescues torchao FP8 from eager-mode Python dispatch
overhead (FP8 eager was ~5x slower than BF16).
"""
import argparse, json, os, time
import torch
from diffusers import DiffusionPipeline
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--resolution", type=int, default=1024)
ap.add_argument("--device_label", default="h100")
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--mode", default="reduce-overhead",
                choices=["default", "reduce-overhead", "max-autotune"])
ap.add_argument("--fullgraph", action="store_true", default=False)
args = ap.parse_args()

PROMPT = "An astronaut riding a green horse"
STEPS = 50
GUIDANCE = 7.5
HEIGHT = WIDTH = args.resolution
os.makedirs(args.out, exist_ok=True)

print(f"[load] SDXL bf16 -> FP8 res={args.resolution}")
t0 = time.time()
pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                         use_safetensors=True).to("cuda")
quantize_(pipe.unet, Float8DynamicActivationFloat8WeightConfig())
if args.resolution >= 2048:
    pipe.enable_vae_tiling()
load_s = time.time() - t0

print(f"[compile] wrapping UNet with torch.compile(mode={args.mode}, "
      f"fullgraph={args.fullgraph})")
t1 = time.time()
pipe.unet = torch.compile(pipe.unet, mode=args.mode, fullgraph=args.fullgraph)
print(f"[compile] wrap took {time.time()-t1:.2f}s (actual compile on first warmup)")

print(f"[warmup] {args.warmup} rounds — first 1-2 trigger compile")
for i in range(args.warmup):
    t = time.time()
    _ = pipe(PROMPT, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS,
             guidance_scale=GUIDANCE,
             generator=torch.Generator("cpu").manual_seed(0)).images[0]
    print(f"  warmup {i+1}/{args.warmup}: {time.time()-t:.2f}s")

runs = []
peak = 0.0
for seed in args.seeds:
    try:
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        img = pipe(PROMPT, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS,
                   guidance_scale=GUIDANCE,
                   generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        e2e = time.time() - t
        pk = torch.cuda.max_memory_allocated() / 1024**3
        peak = max(peak, pk)
        import numpy as np
        arr = np.asarray(img).astype(np.float32)
        std_v = float(arr.std())
        png = os.path.join(args.out, f"seed{seed}_astro.png")
        img.save(png)
        runs.append({"seed": seed, "time_s": round(e2e, 2),
                     "peak_vram_gb": round(pk, 2), "std": round(std_v, 2),
                     "png": png, "status": "ok"})
        print(f"[seed {seed}] {e2e:.2f}s peak={pk:.2f}GB std={std_v:.2f}")
    except Exception as ex:
        msg = str(ex)[:200]
        runs.append({"seed": seed, "status": "error", "error": msg})
        print(f"[seed {seed}] ERROR: {msg}")
        torch.cuda.empty_cache()

ok = [r for r in runs if r.get("status") == "ok"]
summary = {
    "device": f"{args.device_label} res={args.resolution} fp8+compile({args.mode})",
    "model": "SDXL-base-1.0",
    "prompt": PROMPT, "steps": STEPS, "guidance": GUIDANCE,
    "resolution": f"{HEIGHT}x{WIDTH}",
    "compile_mode": args.mode, "fullgraph": args.fullgraph,
    "load_s": round(load_s, 1), "peak_vram_gb": round(peak, 2),
    "mean_s": round(sum(r["time_s"] for r in ok)/max(1, len(ok)), 2) if ok else None,
    "n_ok": len(ok), "n_total": len(runs),
    "runs": runs,
}
with open(os.path.join(args.out, "results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"[done] mean={summary['mean_s']}s peak={peak:.2f}GB "
      f"{summary['n_ok']}/{summary['n_total']} ok")
