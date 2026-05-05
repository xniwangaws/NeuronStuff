#!/usr/bin/env python3
"""FLUX.1-dev H100 benchmark: alien prompt, 28 steps, 10 seeds, 1024², BF16 + FP8."""
import argparse, json, os, time
import torch
from diffusers import FluxPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--precision", choices=["bf16", "fp8"], required=True)
ap.add_argument("--model", default="/opt/dlami/nvme/FLUX.1-dev")
ap.add_argument("--out", default=None)
args = ap.parse_args()

PROMPT = "A close-up image of a green alien with fluorescent skin in the middle of a dark purple forest"
STEPS = 28
SEEDS = list(range(42, 52))
GUIDANCE = 3.5
HEIGHT, WIDTH = 1024, 1024
WARMUP = 2
OUT = args.out or f"/opt/dlami/nvme/flux1_alien_out_h100_{args.precision}"
os.makedirs(OUT, exist_ok=True)

print(f"[load] FLUX.1-dev {args.precision.upper()}")
t0 = time.time()
pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

if args.precision == "fp8":
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_
    quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
    print("[fp8] transformer quantized via torchao")
load_s = time.time() - t0
print(f"[load] done {load_s:.1f}s")

print(f"[warmup] {WARMUP} rounds @ {STEPS} steps")
for i in range(WARMUP):
    t = time.time()
    _ = pipe(PROMPT, height=HEIGHT, width=WIDTH, guidance_scale=GUIDANCE,
             num_inference_steps=STEPS, max_sequence_length=512,
             generator=torch.Generator("cpu").manual_seed(0)).images[0]
    print(f"  warmup {i+1}/{WARMUP}: {time.time()-t:.2f}s")

runs = []
peak = 0.0
for seed in SEEDS:
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    img = pipe(PROMPT, height=HEIGHT, width=WIDTH, guidance_scale=GUIDANCE,
               num_inference_steps=STEPS, max_sequence_length=512,
               generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    e2e = time.time() - t
    pk = torch.cuda.max_memory_allocated() / 1024**3
    peak = max(peak, pk)
    import numpy as np
    arr = np.asarray(img).astype(np.float32)
    std_v = float(arr.std())
    png = os.path.join(OUT, f"seed{seed}_alien.png")
    img.save(png)
    runs.append({"seed": seed, "time_s": round(e2e, 2), "peak_vram_gb": round(pk, 2), "std": round(std_v, 2), "png": png})
    print(f"[seed {seed}] {e2e:.2f}s peak={pk:.2f}GB std={std_v:.2f}")

summary = {
    "device": f"H100 p5.4xlarge ({args.precision.upper()})",
    "model": "FLUX.1-dev",
    "prompt": PROMPT, "steps": STEPS, "guidance": GUIDANCE,
    "resolution": f"{HEIGHT}x{WIDTH}",
    "load_s": round(load_s, 1), "peak_vram_gb": round(peak, 2),
    "mean_s": round(sum(r["time_s"] for r in runs)/len(runs), 2),
    "runs": runs,
}
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"[done] mean={summary['mean_s']}s peak={peak:.2f}GB over {len(runs)} seeds")
