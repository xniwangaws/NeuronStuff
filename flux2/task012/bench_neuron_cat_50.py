"""Neuron cat-prompt 50-step benchmark (OPPO customer spec)."""
import argparse, json, time
from pathlib import Path
import torch
import torch_neuronx  # noqa

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
    ap.add_argument("--out-dir", default="/home/ubuntu/bench_neuron_cat_50")
    ap.add_argument("--dit-neff", default="/home/ubuntu/dit_traced/dit_tp8_1024.pt")
    ap.add_argument("--te-neff", default="/home/ubuntu/text_encoder_traced/text_encoder.pt")
    ap.add_argument("--vae-neff", default="/home/ubuntu/vae_traced/vae_decoder_512.pt")
    args = ap.parse_args()

    out = Path(args.out_dir) / f"neuron_bf16_{args.resolution}_cat_{args.steps}step"
    out.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, "/home/ubuntu")
    from neuron_flux2_pipeline import NeuronFlux2Pipeline

    print(f"[load] DiT={args.dit_neff}")
    t0 = time.perf_counter()
    pipe = NeuronFlux2Pipeline.from_traced(
        dit_neff=args.dit_neff, te_neff=args.te_neff, vae_neff=args.vae_neff,
        weights_dir="/home/ubuntu/flux2_weights")
    load_time = time.perf_counter() - t0
    print(f"[load] {load_time:.2f}s")

    per_run = []
    for i in range(N_SEEDS):
        gen = torch.Generator(device="cpu").manual_seed(SEED_BASE + i)
        t = time.perf_counter()
        image = pipe(prompt=CAT_PROMPT, height=args.resolution, width=args.resolution,
                     num_inference_steps=args.steps, guidance_scale=args.guidance, generator=gen)
        dt = time.perf_counter() - t
        per_run.append(dt)
        image.save(out / f"seed{SEED_BASE + i:04d}_cat.png")
        print(f"[run {i}] {dt:.2f}s")

    first = per_run[0]; rest = per_run[1:]
    results = {
        "device": "Neuron-trn2.48xlarge", "precision": "BF16",
        "prompt": CAT_PROMPT, "resolution": args.resolution,
        "steps": args.steps, "guidance": args.guidance, "seed_base": SEED_BASE,
        "load_time_s": load_time, "first_image_s": first,
        "mean_steady_s": sum(rest) / len(rest) if rest else 0.0,
        "p50_s": percentile(per_run, 0.5), "p95_s": percentile(per_run, 0.95),
        "per_run_s": per_run,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[done] load={load_time:.2f}s first={first:.2f}s mean_rest={results['mean_steady_s']:.2f}s p95={results['p95_s']:.2f}s")


if __name__ == "__main__":
    main()
