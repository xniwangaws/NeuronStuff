#!/usr/bin/env python3
"""FLUX.1-dev on g6e.2xlarge L40S - multiple optimization levels, float16."""
import time
import gc
import torch
import subprocess

MODEL_PATH = "/home/ubuntu/models/FLUX.1-dev/"
PROMPT = "A cat holding a sign that says hello world"
HEIGHT, WIDTH = 1024, 1024
WARMUP_ROUNDS = 5
STEPS = 25

def get_gpu_info():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
        text=True
    ).strip()

def run_benchmark(pipe, label, steps=STEPS):
    """5 warmup + 1 timed run."""
    print(f"\n--- {label} (steps={steps}) ---")
    print(f"GPU before run: {get_gpu_info()}")
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
    print(f"  => {label}: {elapsed:.2f}s")
    return elapsed

results = {}

# ============================================================
# Test 1: Baseline float16
# ============================================================
from diffusers import FluxPipeline

print("=" * 60)
print("Test 1: Baseline float16")
print("=" * 60)
pipe = FluxPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
pipe.to("cuda:0")
print(f"GPU after load: {get_gpu_info()}")

results["fp16_baseline"] = {}
for steps in [15, 25, 50]:
    results["fp16_baseline"][steps] = run_benchmark(pipe, f"fp16 baseline", steps=steps)

# ============================================================
# Test 2: float16 + torch.compile
# ============================================================
print("\n" + "=" * 60)
print("Test 2: float16 + torch.compile")
print("=" * 60)

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False

print("Compiling transformer...")
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)

results["fp16_compile"] = {}
for steps in [15, 25, 50]:
    results["fp16_compile"][steps] = run_benchmark(pipe, f"fp16+compile", steps=steps)

# ============================================================
# Test 3: float16 + torch.compile + FirstBlockCache
# ============================================================
print("\n" + "=" * 60)
print("Test 3: float16 + torch.compile + FirstBlockCache")
print("=" * 60)

try:
    from diffusers import FirstBlockCacheConfig
    cache_config = FirstBlockCacheConfig(threshold=0.18)
    pipe.transformer.enable_cache(cache_config)
    print("FirstBlockCache enabled")

    results["fp16_compile_fbcache"] = {}
    for steps in [15, 25, 50]:
        results["fp16_compile_fbcache"][steps] = run_benchmark(pipe, f"fp16+compile+fbcache", steps=steps)
except Exception as e:
    print(f"FirstBlockCache not available: {e}")
    results["fp16_compile_fbcache"] = None

del pipe
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# Summary
# ============================================================
gpu_name = get_gpu_info().split(",")[0].strip()
print(f"\n{'='*60}")
print(f"FLUX.1-dev Optimization Benchmark - {gpu_name} (float16)")
print(f"{'='*60}")
print(f"{'Config':<30} {'15 steps':>10} {'25 steps':>10} {'50 steps':>10}")
print("-" * 62)
for label, res in results.items():
    if res is None:
        continue
    print(f"{label:<30} {res.get(15, 0):>10.2f} {res.get(25, 0):>10.2f} {res.get(50, 0):>10.2f}")
print(f"{'='*60}")
