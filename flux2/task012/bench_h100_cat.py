"""H100 cat-prompt 50-step benchmark (OPPO spec): BF16 + FP8 variants."""
import argparse, gc, json, time
from pathlib import Path
import torch
from diffusers import Flux2Pipeline

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
    ap.add_argument("--out-dir", default="out_cat")
    ap.add_argument("--mode", choices=["bf16", "fp8"], default="bf16")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--model", default="black-forest-labs/FLUX.2-dev")
    args = ap.parse_args()

    out = Path(args.out_dir) / f"h100_{args.mode}_{args.resolution}_cat_{args.steps}step"
    out.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    print(f"[load] mode={args.mode} model={args.model}")
    t0 = time.perf_counter()
    pipe = Flux2Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    

    if args.mode == "fp8":
        from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
        print("[fp8] torchao FP8 dynamic quant on transformer+te")
        q = float8_dynamic_activation_float8_weight()
        quantize_(pipe.transformer, q)
        quantize_(pipe.text_encoder, q)
    if args.offload:
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
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[run {i}] {dt:.2f}s  peak={peak:.2f}GB")

    first = per_run[0]; rest = per_run[1:]
    results = {
        "device": "H100-80GB", "instance": "p5.4xlarge", "region": "ap-south-1",
        "model": args.model, "precision": args.mode.upper(),
        "prompt": CAT_PROMPT, "resolution": args.resolution,
        "steps": args.steps, "guidance": args.guidance, "seed_base": SEED_BASE,
        "load_time_s": load_time, "first_image_s": first,
        "mean_steady_s": sum(rest) / len(rest) if rest else 0.0,
        "p50_s": percentile(per_run, 0.5), "p95_s": percentile(per_run, 0.95),
        "per_run_s": per_run,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[done] load={load_time:.2f}s first={first:.2f}s mean_rest={results['mean_steady_s']:.2f}s p95={results['p95_s']:.2f}s peak={results['peak_vram_gb']:.2f}GB")


if __name__ == "__main__":
    main()
