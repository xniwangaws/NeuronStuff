#!/usr/bin/env python3
"""Unified FLUX.1-dev benchmark on GPU - steps 15, 25, 50."""
import time
import torch
from diffusers import FluxPipeline

MODEL_PATH = "/home/ubuntu/models/FLUX.1-dev/"
PROMPT = "A cat holding a sign that says hello world"
HEIGHT, WIDTH = 1024, 1024
WARMUP_ROUNDS = 5
STEPS_LIST = [15, 25, 50]

print("Loading FLUX.1-dev (bfloat16)...")
load_start = time.time()
pipe = FluxPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
pipe.to("cuda:0")
load_time = time.time() - load_start
print(f"Load time: {load_time:.2f}s")

import subprocess
gpu_info = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
    text=True
).strip()
print(f"GPU: {gpu_info}")

results = {}
for steps in STEPS_LIST:
    print(f"\nsteps={steps}: warming up ({WARMUP_ROUNDS} rounds)...")
    for i in range(WARMUP_ROUNDS):
        _ = pipe(
            PROMPT, height=HEIGHT, width=WIDTH,
            guidance_scale=3.5, num_inference_steps=steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]
        print(f"  Warmup {i+1}/{WARMUP_ROUNDS}")

    start = time.time()
    image = pipe(
        PROMPT, height=HEIGHT, width=WIDTH,
        guidance_scale=3.5, num_inference_steps=steps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    elapsed = time.time() - start
    results[steps] = elapsed
    image.save(f"/home/ubuntu/flux_unified_steps{steps}.png")
    print(f"  steps={steps}: {elapsed:.2f}s")

gpu_name = gpu_info.split(",")[0].strip()
print(f"\n{'='*50}")
print(f"FLUX.1-dev Unified Benchmark - {gpu_name}")
print(f"{'='*50}")
print(f"dtype=bfloat16, warmup={WARMUP_ROUNDS}, seed=0")
for steps in STEPS_LIST:
    print(f"  steps={steps:>2}: {results[steps]:.2f}s")
print(f"{'='*50}")
