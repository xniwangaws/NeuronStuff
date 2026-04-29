"""FLUX.2-dev NF4 cat-prompt 50-step benchmark on L4 24GB (OPPO customer spec)."""
import argparse, gc, json, time
from pathlib import Path
import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from transformers import AutoModel

CAT_PROMPT = "A cat holding a sign that says hello world"
SEED_BASE = 42
N_SEEDS = 10


def percentile(values, p):
    s = sorted(values); k = (len(s) - 1) * p; f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--model", default="diffusers/FLUX.2-dev-bnb-4bit")
    ap.add_argument("--out-dir", default="out_cat")
    args = ap.parse_args()

    out = Path(args.out_dir) / f"{args.resolution}_cat_{args.steps}step"
    out.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    print(f"[load] {args.model}")
    t0 = time.perf_counter()
    transformer = Flux2Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cpu")
    text_encoder = AutoModel.from_pretrained(
        args.model, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cpu")
    pipe = Flux2Pipeline.from_pretrained(
        args.model, transformer=transformer, text_encoder=text_encoder,
        torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    load_time = time.perf_counter() - t0
    print(f"[load] {load_time:.2f}s")

    per_run = []
    for i in range(N_SEEDS):
        gc.collect(); torch.cuda.empty_cache()
        gen = torch.Generator(device="cuda").manual_seed(SEED_BASE + i)
        t = time.perf_counter()
        image = pipe(
            prompt=CAT_PROMPT, height=args.resolution, width=args.resolution,
            num_inference_steps=args.steps, guidance_scale=args.guidance, generator=gen,
        ).images[0]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        per_run.append(dt)
        image.save(out / f"seed{SEED_BASE + i:04d}_cat.png")
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[run {i}] {dt:.2f}s  peak_vram={peak_gb:.2f}GB")

    first = per_run[0]; rest = per_run[1:]
    mean_rest = sum(rest) / len(rest) if rest else 0.0
    results = {
        "device": "L4-24GB", "instance": "g6.4xlarge", "region": "sa-east-1",
        "model": args.model, "precision": "NF4 (bnb 4-bit)",
        "prompt": CAT_PROMPT, "resolution": args.resolution,
        "steps": args.steps, "guidance": args.guidance, "seed_base": SEED_BASE,
        "load_time_s": load_time, "first_image_s": first,
        "mean_steady_s": mean_rest,
        "p50_s": percentile(per_run, 0.5), "p95_s": percentile(per_run, 0.95),
        "per_run_s": per_run,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[done] load={load_time:.2f}s first={first:.2f}s mean_rest={mean_rest:.2f}s p95={results['p95_s']:.2f}s peak={results['peak_vram_gb']:.2f}GB")


if __name__ == "__main__":
    main()
