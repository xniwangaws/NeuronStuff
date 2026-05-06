#!/usr/bin/env python3
"""
FLUX.1-dev benchmark — ComfyUI-style loading (component-by-component).

Two modes:
  1. Full pipeline:  T5-XXL + CLIP-L text encoding → DiT denoising → VAE decode
  2. Skip T5:        CLIP-L only (T5 output zeroed) → DiT denoising → VAE decode

This mirrors how ComfyUI loads FLUX: each component (text encoders, transformer,
VAE) is a separate safetensor loaded independently, NOT through diffusers Pipeline.

Usage:
  python benchmark_comfyui_style.py                    # run both modes
  python benchmark_comfyui_style.py --mode full         # T5 + CLIP-L
  python benchmark_comfyui_style.py --mode skip_t5      # CLIP-L only
"""
import argparse
import gc
import subprocess
import time

import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

MODEL_PATH = "/home/ubuntu/models/FLUX.1-dev/"
PROMPT = "A cat holding a sign that says hello world"
HEIGHT, WIDTH = 1024, 1024
WARMUP_ROUNDS = 3
STEPS_LIST = [15, 25, 50]
SEED = 0


def get_gpu_info():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
         "--format=csv,noheader"], text=True
    ).strip()


def load_component(name, load_fn, **kwargs):
    print(f"  Loading {name}...")
    start = time.time()
    obj = load_fn(**kwargs)
    elapsed = time.time() - start
    print(f"  {name} loaded in {elapsed:.1f}s | GPU: {get_gpu_info()}")
    return obj


def build_pipeline_full():
    """Load all components separately (ComfyUI style) — full T5 + CLIP-L."""
    print("\n=== Loading FULL pipeline (T5-XXL + CLIP-L) ===")
    print(f"GPU before load: {get_gpu_info()}")

    tokenizer = load_component("CLIP-L tokenizer",
        CLIPTokenizer.from_pretrained, pretrained_model_name_or_path=MODEL_PATH, subfolder="tokenizer")
    tokenizer_2 = load_component("T5 tokenizer",
        T5TokenizerFast.from_pretrained, pretrained_model_name_or_path=MODEL_PATH, subfolder="tokenizer_2")

    text_encoder = load_component("CLIP-L text encoder",
        CLIPTextModel.from_pretrained, pretrained_model_name_or_path=MODEL_PATH,
        subfolder="text_encoder", torch_dtype=torch.bfloat16)

    text_encoder_2 = load_component("T5-XXL text encoder",
        T5EncoderModel.from_pretrained, pretrained_model_name_or_path=MODEL_PATH,
        subfolder="text_encoder_2", torch_dtype=torch.bfloat16)

    transformer = load_component("Flux transformer (DiT)",
        FluxTransformer2DModel.from_pretrained, pretrained_model_name_or_path=MODEL_PATH,
        subfolder="transformer", torch_dtype=torch.bfloat16)

    vae = load_component("VAE",
        AutoencoderKL.from_pretrained, pretrained_model_name_or_path=MODEL_PATH,
        subfolder="vae", torch_dtype=torch.bfloat16)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

    pipe = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        transformer=transformer,
        vae=vae,
    )
    pipe.enable_model_cpu_offload()
    print(f"GPU after pipeline assembly + offload: {get_gpu_info()}")
    return pipe


def build_pipeline_skip_t5():
    """Load without T5-XXL (ComfyUI style skip text encoder)."""
    print("\n=== Loading SKIP-T5 pipeline (CLIP-L only) ===")
    print(f"GPU before load: {get_gpu_info()}")

    tokenizer = load_component("CLIP-L tokenizer",
        CLIPTokenizer.from_pretrained, pretrained_model_name_or_path=MODEL_PATH, subfolder="tokenizer")
    tokenizer_2 = load_component("T5 tokenizer",
        T5TokenizerFast.from_pretrained, pretrained_model_name_or_path=MODEL_PATH, subfolder="tokenizer_2")

    text_encoder = load_component("CLIP-L text encoder",
        CLIPTextModel.from_pretrained, pretrained_model_name_or_path=MODEL_PATH,
        subfolder="text_encoder", torch_dtype=torch.bfloat16)

    # Skip T5-XXL — set to None
    text_encoder_2 = None

    transformer = load_component("Flux transformer (DiT)",
        FluxTransformer2DModel.from_pretrained, pretrained_model_name_or_path=MODEL_PATH,
        subfolder="transformer", torch_dtype=torch.bfloat16)

    vae = load_component("VAE",
        AutoencoderKL.from_pretrained, pretrained_model_name_or_path=MODEL_PATH,
        subfolder="vae", torch_dtype=torch.bfloat16)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

    pipe = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        transformer=transformer,
        vae=vae,
    )
    pipe.enable_model_cpu_offload()
    print(f"GPU after pipeline assembly + offload: {get_gpu_info()}")
    return pipe


def benchmark(pipe, label, steps_list=STEPS_LIST):
    results = {}
    for steps in steps_list:
        print(f"\n  [{label}] steps={steps}: warming up ({WARMUP_ROUNDS} rounds)...")
        for i in range(WARMUP_ROUNDS):
            _ = pipe(
                PROMPT, height=HEIGHT, width=WIDTH,
                guidance_scale=3.5, num_inference_steps=steps,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(SEED),
            ).images[0]
            print(f"    Warmup {i+1}/{WARMUP_ROUNDS} | GPU: {get_gpu_info()}")

        start = time.time()
        image = pipe(
            PROMPT, height=HEIGHT, width=WIDTH,
            guidance_scale=3.5, num_inference_steps=steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(SEED),
        ).images[0]
        elapsed = time.time() - start
        results[steps] = elapsed
        fname = f"/home/ubuntu/flux_comfy_{label.replace(' ', '_')}_steps{steps}.png"
        image.save(fname)
        print(f"    [{label}] steps={steps}: {elapsed:.2f}s | saved {fname}")

    return results


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def print_results(all_results):
    print(f"\n{'='*70}")
    print(f"FLUX.1-dev ComfyUI-Style Benchmark — Single L4")
    print(f"Resolution: {HEIGHT}x{WIDTH}, seed={SEED}, warmup={WARMUP_ROUNDS}")
    print(f"GPU: {get_gpu_info()}")
    print(f"{'='*70}")
    print(f"{'Mode':<35} {'15 steps':>10} {'25 steps':>10} {'50 steps':>10}")
    print("-" * 70)
    for label, results in all_results.items():
        cols = []
        for s in STEPS_LIST:
            if s in results:
                cols.append(f"{results[s]:.2f}s")
            else:
                cols.append("—")
        print(f"{label:<35} {cols[0]:>10} {cols[1]:>10} {cols[2]:>10}")
    print(f"{'='*70}")

    if len(all_results) == 2:
        labels = list(all_results.keys())
        print(f"\nT5 encoder overhead (full - skip_t5):")
        for s in STEPS_LIST:
            if s in all_results[labels[0]] and s in all_results[labels[1]]:
                diff = all_results[labels[0]][s] - all_results[labels[1]][s]
                pct = diff / all_results[labels[0]][s] * 100
                print(f"  steps={s}: {diff:+.2f}s ({pct:+.1f}% of total)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "skip_t5", "both"], default="both")
    args = parser.parse_args()

    all_results = {}

    if args.mode in ("full", "both"):
        pipe = build_pipeline_full()
        all_results["bf16 full (T5+CLIP-L) offload"] = benchmark(pipe, "full")
        del pipe
        cleanup()

    if args.mode in ("skip_t5", "both"):
        pipe = build_pipeline_skip_t5()
        all_results["bf16 skip-T5 (CLIP-L only) offload"] = benchmark(pipe, "skip_t5")
        del pipe
        cleanup()

    print_results(all_results)


if __name__ == "__main__":
    main()
