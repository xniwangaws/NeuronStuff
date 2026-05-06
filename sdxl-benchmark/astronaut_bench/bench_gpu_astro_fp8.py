#!/usr/bin/env python3
"""SDXL GPU FP8 benchmark (torchao): astronaut prompt, 50 steps, 1K/2K/4K, seeds 42-51."""
import argparse, json, os, time
import torch
from diffusers import DiffusionPipeline
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--resolution", type=int, required=True, choices=[1024, 2048, 4096])
ap.add_argument("--device_label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
ap.add_argument("--warmup", type=int, default=2)
args = ap.parse_args()

PROMPT = "An astronaut riding a green horse"
STEPS = 50
GUIDANCE = 7.5
HEIGHT = WIDTH = args.resolution

os.makedirs(args.out, exist_ok=True)

print(f"[load] SDXL bf16->fp8 res={args.resolution}")
t0 = time.time()
pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, use_safetensors=True).to("cuda")
# Quantize UNet to FP8 (torchao dynamic activation + FP8 weight)
quantize_(pipe.unet, Float8DynamicActivationFloat8WeightConfig())
# NOTE: On torch 2.11 / CUDA 13, torchao FP8 without torch.compile runs slower
# than BF16 because each Linear dispatches a CPU-side quantize op per call.
# With torch.compile + max-autotune, FP8 gets ~1.5-2x speedup, but that compile
# takes >15 min per resolution which blows the benchmark budget. We therefore
# report FP8 without compile: the primary FP8 benefit here is the ~2x weight
# memory reduction; latency is a regression vs BF16.
# VAE tiling for 2K/4K to keep VAE decode memory bounded, matching BF16 config
if args.resolution >= 2048:
    pipe.enable_vae_tiling()
load_s = time.time() - t0
print(f"[load] done {load_s:.1f}s")

print(f"[warmup] {args.warmup} rounds")
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
        runs.append({"seed": seed, "time_s": round(e2e, 2), "peak_vram_gb": round(pk, 2),
                     "std": round(std_v, 2), "png": png, "status": "ok"})
        print(f"[seed {seed}] {e2e:.2f}s peak={pk:.2f}GB std={std_v:.2f}")
    except Exception as ex:
        msg = str(ex)[:200]
        runs.append({"seed": seed, "status": "error", "error": msg})
        print(f"[seed {seed}] ERROR: {msg}")
        torch.cuda.empty_cache()

ok_runs = [r for r in runs if r.get("status") == "ok"]
summary = {
    "device": f"{args.device_label} res={args.resolution} fp8",
    "model": "SDXL-base-1.0",
    "prompt": PROMPT, "steps": STEPS, "guidance": GUIDANCE,
    "resolution": f"{HEIGHT}x{WIDTH}",
    "load_s": round(load_s, 1), "peak_vram_gb": round(peak, 2),
    "mean_s": round(sum(r["time_s"] for r in ok_runs)/max(1, len(ok_runs)), 2) if ok_runs else None,
    "n_ok": len(ok_runs), "n_total": len(runs),
    "runs": runs,
}
with open(os.path.join(args.out, "results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"[done] mean={summary['mean_s']}s peak={peak:.2f}GB {summary['n_ok']}/{summary['n_total']} ok")
