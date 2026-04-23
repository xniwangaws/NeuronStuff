#!/usr/bin/env python3
"""
FLUX.1-dev benchmark — ComfyUI-style loading with Kijai pre-quantized FP8 transformer.

Loads Kijai/flux-fp8 `flux1-dev-fp8.safetensors` via `from_single_file` and enables
`enable_layerwise_casting(storage=float8_e4m3fn, compute=bfloat16)` to match
ComfyUI's default FP8 behavior (FP8 storage, bf16 compute).

Two modes:
  1. Full:     T5-XXL + CLIP-L → FP8 DiT → VAE  (model_cpu_offload)
  2. Skip T5:  CLIP-L only     → FP8 DiT → VAE  (fits in 24GB without offload)

Usage:
  python benchmark_comfyui_style_fp8.py                # run both modes
  python benchmark_comfyui_style_fp8.py --mode full
  python benchmark_comfyui_style_fp8.py --mode skip_t5
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
FP8_CKPT = "/home/ubuntu/flux1-dev-fp8.safetensors"  # Kijai/flux-fp8 single-file
PROMPT = "A cat holding a sign that says hello world"
HEIGHT, WIDTH = 1024, 1024
WARMUP_ROUNDS = 3
STEPS_LIST = [25]
SEED = 0


def get_gpu_info():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
         "--format=csv,noheader"], text=True
    ).strip()


def load_fp8_transformer():
    """Load Kijai's pre-quantized single-file FP8 FLUX and enable layerwise casting
    (ComfyUI-default: FP8 E4M3FN storage, bf16 compute)."""
    print("  Loading pre-quantized FP8 transformer (Kijai single-file)...")
    start = time.time()
    transformer = FluxTransformer2DModel.from_single_file(
        FP8_CKPT, torch_dtype=torch.bfloat16, config=MODEL_PATH, subfolder="transformer",
    )
    transformer.enable_layerwise_casting(
        storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16,
    )
    elapsed = time.time() - start
    print(f"  FP8 transformer loaded in {elapsed:.1f}s | GPU: {get_gpu_info()}")
    return transformer


def build_pipeline_full_fp8():
    """FP8 transformer + T5-XXL + CLIP-L with model_cpu_offload."""
    print("\n=== Loading FULL FP8 pipeline (T5-XXL + CLIP-L) ===")
    print(f"GPU before load: {get_gpu_info()}")

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(MODEL_PATH, subfolder="tokenizer_2")

    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_PATH, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        MODEL_PATH, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)

    transformer = load_fp8_transformer()

    vae = AutoencoderKL.from_pretrained(
        MODEL_PATH, subfolder="vae", torch_dtype=torch.bfloat16)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

    pipe = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder, tokenizer=tokenizer,
        text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2,
        transformer=transformer, vae=vae,
    )
    pipe.enable_model_cpu_offload()
    print(f"GPU after assembly + offload: {get_gpu_info()}")
    return pipe


def build_pipeline_skip_t5_fp8():
    """FP8 transformer + CLIP-L only — no T5, try to fit on GPU without offload."""
    print("\n=== Loading SKIP-T5 FP8 pipeline (CLIP-L only) ===")
    print(f"GPU before load: {get_gpu_info()}")

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(MODEL_PATH, subfolder="tokenizer_2")

    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_PATH, subfolder="text_encoder", torch_dtype=torch.bfloat16)

    transformer = load_fp8_transformer()

    vae = AutoencoderKL.from_pretrained(
        MODEL_PATH, subfolder="vae", torch_dtype=torch.bfloat16)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

    pipe = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder, tokenizer=tokenizer,
        text_encoder_2=None, tokenizer_2=tokenizer_2,
        transformer=transformer, vae=vae,
    )

    # FP8-storage DiT (~12GB) + CLIP-L (~0.5GB) + VAE (~0.2GB) ≈ 12.7GB
    # Should fit on L4 24GB without offload. Try direct GPU first.
    try:
        pipe.to("cuda:0")
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        print(f"  Loaded to GPU without offload | GPU: {get_gpu_info()}")
    except Exception as e:
        print(f"  Direct GPU failed ({e}), falling back to model_cpu_offload")
        pipe.enable_model_cpu_offload()

    print(f"GPU after assembly: {get_gpu_info()}")
    return pipe


def benchmark(pipe, label):
    results = {}
    skip_t5 = (label == "skip_t5")

    # In skip_t5 mode, pre-compute CLIP-L pooled embed once and use zeros for T5
    # (mirrors ComfyUI's skip-T5 behavior: CLIP-L still runs, T5 output is zeroed).
    pre_kwargs = {}
    if skip_t5:
        device = pipe._execution_device if hasattr(pipe, "_execution_device") else "cuda:0"
        clip_ids = pipe.tokenizer(
            PROMPT, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        with torch.no_grad():
            pooled = pipe.text_encoder(clip_ids, output_hidden_states=False).pooler_output
        pooled = pooled.to(dtype=torch.bfloat16)
        t5_embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=device)
        pre_kwargs = {
            "prompt": None,
            "prompt_embeds": t5_embeds,
            "pooled_prompt_embeds": pooled,
        }
        print(f"    [skip_t5] pre-computed pooled={tuple(pooled.shape)}, t5_embeds={tuple(t5_embeds.shape)} (zeros)")

    for steps in STEPS_LIST:
        print(f"\n  [{label}] steps={steps}: warming up ({WARMUP_ROUNDS} rounds)...")
        for i in range(WARMUP_ROUNDS):
            if skip_t5:
                _ = pipe(
                    height=HEIGHT, width=WIDTH,
                    guidance_scale=3.5, num_inference_steps=steps,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(SEED),
                    **pre_kwargs,
                ).images[0]
            else:
                _ = pipe(
                    PROMPT, height=HEIGHT, width=WIDTH,
                    guidance_scale=3.5, num_inference_steps=steps,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(SEED),
                ).images[0]
            print(f"    Warmup {i+1}/{WARMUP_ROUNDS} | GPU: {get_gpu_info()}")

        start = time.time()
        if skip_t5:
            image = pipe(
                height=HEIGHT, width=WIDTH,
                guidance_scale=3.5, num_inference_steps=steps,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(SEED),
                **pre_kwargs,
            ).images[0]
        else:
            image = pipe(
                PROMPT, height=HEIGHT, width=WIDTH,
                guidance_scale=3.5, num_inference_steps=steps,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(SEED),
            ).images[0]
        elapsed = time.time() - start
        results[steps] = elapsed
        fname = f"/home/ubuntu/flux_comfy_fp8_{label.replace(' ', '_')}_steps{steps}.png"
        image.save(fname)
        print(f"    [{label}] steps={steps}: {elapsed:.2f}s | saved {fname}")

    return results


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def print_results(all_results):
    print(f"\n{'='*70}")
    print(f"FLUX.1-dev ComfyUI-Style FP8 Benchmark — Single L4")
    print(f"Resolution: {HEIGHT}x{WIDTH}, seed={SEED}, warmup={WARMUP_ROUNDS}")
    print(f"GPU: {get_gpu_info()}")
    print(f"{'='*70}")
    header = f"{'Mode':<40}"
    for s in STEPS_LIST:
        header += f" {f'{s} steps':>10}"
    print(header)
    print("-" * 70)
    for label, results in all_results.items():
        row = f"{label:<40}"
        for s in STEPS_LIST:
            if s in results:
                row += f" {f'{results[s]:.2f}s':>10}"
            else:
                row += f" {'—':>10}"
        print(row)
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
        pipe = build_pipeline_full_fp8()
        all_results["FP8 full (T5+CLIP-L) offload"] = benchmark(pipe, "full")
        del pipe
        cleanup()

    if args.mode in ("skip_t5", "both"):
        pipe = build_pipeline_skip_t5_fp8()
        all_results["FP8 skip-T5 (CLIP-L only) no-offload"] = benchmark(pipe, "skip_t5")
        del pipe
        cleanup()

    print_results(all_results)


if __name__ == "__main__":
    main()
