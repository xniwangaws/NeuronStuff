"""FLUX.2-dev FP8 benchmark on H100 80GB.

Loads the base BF16 FLUX.2-dev model and casts the DiT transformer to FP8 e4m3 via torchao.
Text encoder (Mistral-3-24B) and VAE stay BF16. Runs 10 fixed-seed prompts at one resolution.

Usage: python bench_h100.py --resolution {1024,2048} [--steps 28]
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from diffusers import Flux2Pipeline
from transformers import AutoModel


PROMPTS = [
    "a high-resolution photograph of a red panda sitting on a tree branch in a misty forest, volumetric lighting",
    "an oil painting of a medieval castle at sunset, dramatic clouds, style of Caspar David Friedrich",
    "a futuristic cyberpunk city street at night, neon signs in Japanese, rain-slicked pavement, cinematic",
    "a watercolor illustration of a hummingbird drinking nectar from a hibiscus flower, soft pastel colors",
    "a studio product photograph of a luxury mechanical wristwatch on black velvet, shallow depth of field",
    "an astronaut riding a horse on the surface of Mars, photorealistic, wide-angle shot",
    "a cozy coffee shop interior on a rainy afternoon, warm lighting, people reading books, bokeh",
    "a macro photograph of a dewdrop on a spider web at sunrise, iridescent refraction",
    "a detailed pencil sketch of a Victorian mansion with ivy climbing the walls, moonlit",
    "a fantasy landscape with floating islands, waterfalls cascading into clouds, dual moons in the sky",
]
SEED = 42


def percentile(values, p):
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--model", default="/home/ubuntu/flux2_weights")
    parser.add_argument("--out-dir", default="out_h100")
    parser.add_argument("--precision", choices=["bf16", "fp8"], default="fp8")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Use enable_model_cpu_offload (needed for BF16 on 80GB H100)")
    args = parser.parse_args()

    out = Path(args.out_dir) / f"{args.precision}_{args.resolution}"
    out.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print(f"[load] starting: {args.model} at {args.precision} (cpu_offload={args.cpu_offload})")
    t0 = time.perf_counter()
    pipe = Flux2Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    if args.precision == "fp8":
        from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
        quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
        print(f"[load] DiT quantized to FP8 e4m3")

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    load_time = time.perf_counter() - t0
    print(f"[load] load_time={load_time:.2f}s  VRAM peak so far={torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    per_run = []
    for i, prompt in enumerate(PROMPTS):
        gc.collect()
        torch.cuda.empty_cache()
        gen = torch.Generator(device="cuda").manual_seed(SEED + i)
        t = time.perf_counter()
        image = pipe(
            prompt=prompt,
            height=args.resolution, width=args.resolution,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=gen,
        ).images[0]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        per_run.append(dt)
        image.save(out / f"seed{SEED + i:04d}_p{i:02d}.png")
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[run {i}] {dt:.2f}s  peak_vram={peak_gb:.2f}GB  prompt[:60]={prompt[:60]}")

    first = per_run[0]
    rest = per_run[1:]
    results = {
        "device": "H100-80GB",
        "instance": "p5.4xlarge",
        "region": "sa-east-1",
        "model": args.model,
        "precision": args.precision,
        "resolution": args.resolution,
        "steps": args.steps,
        "guidance": args.guidance,
        "seed_base": SEED,
        "load_time_s": load_time,
        "first_image_s": first,
        "mean_steady_s": sum(rest) / len(rest) if rest else 0.0,
        "p50_s": percentile(per_run, 0.5),
        "p95_s": percentile(per_run, 0.95),
        "per_run_s": per_run,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[done] load={load_time:.2f}s  first={first:.2f}s  mean_rest={results['mean_steady_s']:.2f}s  p95={results['p95_s']:.2f}s  peak={results['peak_vram_gb']:.2f}GB")


if __name__ == "__main__":
    main()
