#!/usr/bin/env python3
"""FLUX.1-dev on single L4 - compare all CPU offload strategies."""
import time
import gc
import torch
import subprocess
import traceback

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

def run_test(pipe, label):
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    print(f"GPU: {get_gpu_info()}")
    try:
        # Warmup
        for i in range(WARMUP_ROUNDS):
            _ = pipe(
                PROMPT, height=HEIGHT, width=WIDTH,
                guidance_scale=3.5, num_inference_steps=STEPS,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0),
            ).images[0]
            print(f"  Warmup {i+1}/{WARMUP_ROUNDS}")

        # Timed run
        start = time.time()
        image = pipe(
            PROMPT, height=HEIGHT, width=WIDTH,
            guidance_scale=3.5, num_inference_steps=STEPS,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]
        elapsed = time.time() - start
        image.save(f"/home/ubuntu/flux_{label.replace(' ', '_').replace('+', '_')}.png")
        print(f"  => {label}: {elapsed:.2f}s")
        print(f"  GPU peak: {get_gpu_info()}")
        return elapsed
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return None
    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

results = {}

# ============================================================
# Test 1: bf16 + enable_model_cpu_offload
# ============================================================
from diffusers import FluxPipeline
print("\nLoading bf16 pipeline...")
pipe = FluxPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
results["bf16 model_offload"] = run_test(pipe, "bf16 model_offload")

# ============================================================
# Test 2: bf16 + enable_sequential_cpu_offload
# ============================================================
print("\nLoading bf16 pipeline...")
pipe = FluxPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
results["bf16 seq_offload"] = run_test(pipe, "bf16 seq_offload")

# ============================================================
# Test 3: bf16 + group_offload (leaf_level + stream)
# ============================================================
print("\nLoading bf16 pipeline...")
pipe = FluxPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
try:
    pipe.enable_group_offload(offload_type="leaf_level", use_stream=True)
    results["bf16 group_offload_stream"] = run_test(pipe, "bf16 group_offload_stream")
except Exception as e:
    print(f"group_offload not available: {e}")
    del pipe; gc.collect(); torch.cuda.empty_cache()
    results["bf16 group_offload_stream"] = None

# ============================================================
# Test 4: NF4 + no offload (vae tiling)
# ============================================================
from diffusers import FluxTransformer2DModel
from transformers import BitsAndBytesConfig

print("\nLoading NF4 transformer...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
transformer = FluxTransformer2DModel.from_pretrained(
    MODEL_PATH, subfolder="transformer",
    quantization_config=quant_config, torch_dtype=torch.bfloat16,
)
pipe = FluxPipeline.from_pretrained(
    MODEL_PATH, transformer=transformer, torch_dtype=torch.bfloat16,
)
pipe.to("cuda:0")
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
results["NF4 no_offload"] = run_test(pipe, "NF4 no_offload")

# ============================================================
# Test 5: NF4 + enable_model_cpu_offload
# ============================================================
print("\nLoading NF4 transformer...")
transformer = FluxTransformer2DModel.from_pretrained(
    MODEL_PATH, subfolder="transformer",
    quantization_config=quant_config, torch_dtype=torch.bfloat16,
)
pipe = FluxPipeline.from_pretrained(
    MODEL_PATH, transformer=transformer, torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
results["NF4 model_offload"] = run_test(pipe, "NF4 model_offload")

# ============================================================
# Test 6: NF4 + group_offload (leaf_level + stream)
# ============================================================
print("\nLoading NF4 transformer...")
transformer = FluxTransformer2DModel.from_pretrained(
    MODEL_PATH, subfolder="transformer",
    quantization_config=quant_config, torch_dtype=torch.bfloat16,
)
pipe = FluxPipeline.from_pretrained(
    MODEL_PATH, transformer=transformer, torch_dtype=torch.bfloat16,
)
try:
    pipe.enable_group_offload(offload_type="leaf_level", use_stream=True)
    results["NF4 group_offload_stream"] = run_test(pipe, "NF4 group_offload_stream")
except Exception as e:
    print(f"group_offload not available: {e}")
    del pipe; gc.collect(); torch.cuda.empty_cache()
    results["NF4 group_offload_stream"] = None

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"FLUX.1-dev CPU Offload Comparison - Single L4 (steps={STEPS})")
print(f"{'='*60}")
print(f"{'Method':<30} {'Time (s)':>10} {'Status':>10}")
print("-" * 52)
for label, t in results.items():
    if t is None:
        print(f"{label:<30} {'N/A':>10} {'FAILED':>10}")
    else:
        print(f"{label:<30} {t:>10.2f} {'OK':>10}")
print(f"{'='*60}")
