"""Neuron end-to-end FLUX.2-dev benchmark at 1024² BF16.

Reuses the traced components:
  - Text encoder: /home/ubuntu/text_encoder_traced/text_encoder.pt (TP=8)
  - DiT: /home/ubuntu/dit_traced/dit_tp8_1024.pt (TP=8)  <-- from compile_dit_tp8.py
  - VAE decoder: /home/ubuntu/vae_traced/vae_decoder_512.pt (tiled 4x at 1024²)

Runs 10 fixed prompts, seed=42..51, 28 steps, guidance=4.0 — same as GPU baselines.

Usage: python bench_neuron.py --resolution 1024
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch_neuronx  # noqa: F401

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
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--out-dir", default="/home/ubuntu/bench_neuron_out")
    parser.add_argument("--dit-neff", default="/home/ubuntu/dit_traced/dit_tp8_1024.pt")
    parser.add_argument("--te-neff", default="/home/ubuntu/text_encoder_traced/text_encoder.pt")
    parser.add_argument("--vae-neff", default="/home/ubuntu/vae_traced/vae_decoder_512.pt")
    args = parser.parse_args()

    out = Path(args.out_dir) / f"neuron_bf16_{args.resolution}"
    out.mkdir(parents=True, exist_ok=True)

    # Import the pipeline scaffold
    import sys
    sys.path.insert(0, "/home/ubuntu")
    from neuron_flux2_pipeline import NeuronFlux2Pipeline

    print(f"[load] pipeline components (DiT={args.dit_neff})")
    t0 = time.perf_counter()
    pipe = NeuronFlux2Pipeline.from_traced(
        dit_neff=args.dit_neff,
        te_neff=args.te_neff,
        vae_neff=args.vae_neff,
        weights_dir="/home/ubuntu/flux2_weights",
    )
    load_time = time.perf_counter() - t0
    print(f"[load] {load_time:.2f}s")

    per_run = []
    for i, prompt in enumerate(PROMPTS):
        gen = torch.Generator(device="cpu").manual_seed(SEED + i)
        t = time.perf_counter()
        image = pipe(
            prompt=prompt,
            height=args.resolution, width=args.resolution,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=gen,
        )
        dt = time.perf_counter() - t
        per_run.append(dt)
        image.save(out / f"seed{SEED + i:04d}_p{i:02d}.png")
        print(f"[run {i}] {dt:.2f}s  prompt[:60]={prompt[:60]}")

    first = per_run[0]
    rest = per_run[1:]
    results = {
        "device": "Neuron-trn2.48xlarge",
        "precision": "BF16",
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
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[done] load={load_time:.2f}s  first={first:.2f}s  mean_rest={results['mean_steady_s']:.2f}s  p95={results['p95_s']:.2f}s")
    print(f"[done] wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
