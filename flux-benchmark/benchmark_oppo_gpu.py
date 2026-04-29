#!/usr/bin/env python3
"""FLUX.1-dev benchmark on a single GPU (H100) per OPPO spec."""
import csv
import json
import os
import subprocess
import time

import torch
from diffusers import FluxPipeline

MODEL_PATH = "/home/ubuntu/models/FLUX.1-dev/"
OUTPUT_DIR = "/home/ubuntu/outputs/h100"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT = (
    "A close-up image of a green alien with fluorescent skin in the middle "
    "of a dark purple forest"
)
STEPS = 28
SEED = 0
HEIGHT, WIDTH = 1024, 1024
GUIDANCE_SCALE = 3.5
MAX_SEQ_LEN = 512
N_RUNS = 6

print("Phase B: load (from_pretrained + to(cuda))")
t0 = time.time()
pipe = FluxPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
pipe.to("cuda:0")
load_s = time.time() - t0
print(f"  load_s = {load_s:.2f}")

gpu_info = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
    text=True,
).strip()
print(f"GPU: {gpu_info}")

torch.cuda.reset_peak_memory_stats()

runs = []
for i in range(N_RUNS):
    torch.cuda.synchronize()
    t0 = time.time()
    image = pipe(
        PROMPT,
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=STEPS,
        max_sequence_length=MAX_SEQ_LEN,
        generator=torch.Generator("cpu").manual_seed(SEED),
    ).images[0]
    torch.cuda.synchronize()
    dt = time.time() - t0
    runs.append(dt)
    print(f"  run[{i}] = {dt:.2f}s")
    if i == 0:
        image.save(os.path.join(OUTPUT_DIR, f"flux1_step{STEPS}_seed{SEED}_first.png"))
    elif i == 1:
        image.save(os.path.join(OUTPUT_DIR, f"flux1_step{STEPS}_seed{SEED}.png"))

peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
first_inference_s = runs[0]
steady_runs = runs[1:]
steady_mean_s = sum(steady_runs) / len(steady_runs)

summary = {
    "instance": "p5.48xlarge",
    "accelerator": "1x H100 80GB",
    "dtype": "bfloat16",
    "prompt": PROMPT,
    "steps": STEPS,
    "seed": SEED,
    "resolution": f"{HEIGHT}x{WIDTH}",
    "guidance_scale": GUIDANCE_SCALE,
    "n_runs": N_RUNS,
    "compile_s": None,
    "load_s": round(load_s, 2),
    "first_inference_s": round(first_inference_s, 2),
    "steady_mean_s": round(steady_mean_s, 2),
    "steady_runs_s": [round(x, 2) for x in steady_runs],
    "peak_mem_gb": round(peak_mem_gb, 2),
    "gpu_info": gpu_info,
}

with open(os.path.join(OUTPUT_DIR, "run.json"), "w") as f:
    json.dump(summary, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "run.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["phase", "value_s"])
    w.writerow(["load", f"{load_s:.2f}"])
    w.writerow(["first_inference", f"{first_inference_s:.2f}"])
    w.writerow(["steady_mean", f"{steady_mean_s:.2f}"])
    for i, v in enumerate(runs):
        w.writerow([f"run_{i}", f"{v:.2f}"])
    w.writerow(["peak_mem_gb", f"{peak_mem_gb:.2f}"])

print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
