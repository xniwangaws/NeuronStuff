#!/usr/bin/env python3
"""FLUX.1-dev on single L4 - FP8 via torchao + model_cpu_offload."""
import time
import torch

MODEL_PATH = "/home/ubuntu/models/FLUX.1-dev/"
PROMPT = "A cat holding a sign that says hello world"
HEIGHT, WIDTH = 1024, 1024
WARMUP_ROUNDS = 5
STEPS_LIST = [15, 25, 50]

print("Loading pipeline...")
load_start = time.time()

from diffusers import FluxPipeline, FluxTransformer2DModel
from torchao.quantization import quantize_, Float8WeightOnlyConfig

# Load transformer in bf16, quantize to FP8
transformer = FluxTransformer2DModel.from_pretrained(
    MODEL_PATH, subfolder="transformer", torch_dtype=torch.bfloat16,
)
quantize_(transformer, Float8WeightOnlyConfig())

pipe = FluxPipeline.from_pretrained(
    MODEL_PATH,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

load_time = time.time() - load_start
print(f"Load + quantize time: {load_time:.2f}s")

import subprocess
gpu_info = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
    text=True
).strip()
print(f"GPU after load: {gpu_info}")

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
    image.save(f"/home/ubuntu/flux_fp8ao_steps{steps}.png")
    print(f"  steps={steps}: {elapsed:.2f}s")

print(f"\n{'='*50}")
print(f"FLUX.1-dev FP8 (torchao) + model_cpu_offload - Single L4")
print(f"{'='*50}")
print(f"GPU: {gpu_info}")
for steps in STEPS_LIST:
    print(f"  steps={steps:>2}: {results[steps]:.2f}s")
print(f"{'='*50}")
