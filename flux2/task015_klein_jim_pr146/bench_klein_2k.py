#!/usr/bin/env python3
"""10-seed benchmark for FLUX.2-klein-base-9B at 2048x2048 on Neuron."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from application import NeuronFlux2KleinApplication, create_flux2_klein_config

MODEL_PATH = "/mnt/nvme/flux2_klein_weights"
COMPILE_DIR = "/mnt/nvme/klein_compile_workdir_2k"
OUTPUT_DIR = "/mnt/nvme/klein_bench_2k"
PROMPT = "A cat holding a sign that says hello world"
STEPS = 50
GUIDANCE = 4.0
TP = 4
SEEDS = list(range(42, 52))
RES = 2048

os.makedirs(COMPILE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Klein 10-seed bench @ {RES}x{RES}: TP={TP}, steps={STEPS}, seeds={SEEDS[0]}..{SEEDS[-1]}")

backbone_config = create_flux2_klein_config(
    model_path=MODEL_PATH, backbone_tp_degree=TP, dtype=torch.bfloat16,
    height=RES, width=RES,
)
app = NeuronFlux2KleinApplication(
    model_path=MODEL_PATH, backbone_config=backbone_config,
    height=RES, width=RES,
)

t0 = time.time()
app.compile(COMPILE_DIR)
compile_t = time.time()-t0
print(f"compile: {compile_t:.1f}s (may be 0 if cached)")

t0 = time.time()
app.load(COMPILE_DIR)
print(f"load: {time.time()-t0:.1f}s")

# Warmup
print("Warmup x1...")
app(prompt=PROMPT, height=RES, width=RES,
    num_inference_steps=STEPS, guidance_scale=GUIDANCE)

import numpy as np
from PIL import Image

results = []
for seed in SEEDS:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()
    result = app(prompt=PROMPT, height=RES, width=RES,
                 num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                 generator=generator)
    elapsed = time.time() - t0
    img = result.images[0]
    fn = os.path.join(OUTPUT_DIR, f"seed{seed}_cat.png")
    img.save(fn)
    arr = np.array(img)
    std = float(arr.std())
    mean = float(arr.mean())
    verdict = "IMAGE" if std > 50 else "NOISE"
    print(f"  seed={seed}: {elapsed:.2f}s std={std:.1f} mean={mean:.1f} {verdict} -> {fn}")
    results.append({"seed": seed, "time_s": elapsed, "std": std, "mean": mean, "verdict": verdict, "png": fn})

lats = np.array([r["time_s"] for r in results])
passes = sum(1 for r in results if r["verdict"] == "IMAGE")
summary = {
    "resolution": RES,
    "num_seeds": len(results),
    "mean_s": float(lats.mean()),
    "std_s": float(lats.std()),
    "min_s": float(lats.min()),
    "max_s": float(lats.max()),
    "passes": passes,
    "steps": STEPS,
    "tp": TP,
    "prompt": PROMPT,
    "compile_s": compile_t,
    "results": results,
}
ms = summary["mean_s"]; ss = summary["std_s"]; print(f"\nSummary: mean={ms:.2f}s std={ss:.2f}s passes={passes}/{len(results)}")
with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved {OUTPUT_DIR}/results.json")
