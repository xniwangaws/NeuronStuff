"""10-seed benchmark for FLUX.2-dev v3 NEFF. Generates cat-with-hello-world image
across 10 seeds and reports (mean_s, std, pass_count, individual stds).

Pass criterion: per-image std > 40 AND neighbor-pixel correlation > 0.6
(both signals of a structured image, more robust than std alone).
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch_neuronx  # noqa

sys.path.insert(0, "/home/ubuntu")
from neuron_flux2_pipeline import NeuronFlux2Pipeline

CAT_PROMPT = "A cat holding a sign that says hello world"
SEEDS = list(range(42, 52))  # 10 seeds

def main():
    dit_neff = "/mnt/nvme/neff/dit_tp8_1024_v3.pt"
    te_neff = "/mnt/nvme/neff/text_encoder_tp8.pt"
    vae_neff = "/mnt/nvme/neff/vae_decoder_512.pt"
    weights_dir = "/home/ubuntu/flux2_weights"

    print(f"[load] DiT={dit_neff}", flush=True)
    t0 = time.perf_counter()
    pipe = NeuronFlux2Pipeline.from_traced(
        dit_neff=dit_neff, te_neff=te_neff, vae_neff=vae_neff,
        weights_dir=weights_dir)
    print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

    results = []
    for i, seed in enumerate(SEEDS):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        t = time.perf_counter()
        image = pipe(prompt=CAT_PROMPT, height=1024, width=1024,
                     num_inference_steps=50, guidance_scale=4.0, generator=gen)
        gen_time = time.perf_counter() - t
        out_png = f"/tmp/bench_v3_seed{seed}.png"
        image.save(out_png)
        arr = np.asarray(image).astype(np.float32)
        std = arr.std()
        mean = arr.mean()
        # Neighbor pixel correlation (structural signal)
        c1 = arr[:, :-1, :].flatten()
        c2 = arr[:, 1:, :].flatten()
        corr = float(np.corrcoef(c1, c2)[0, 1])
        passed = (std > 40) and (corr > 0.6)
        results.append(dict(seed=seed, std=std, mean=mean, corr=corr, time=gen_time, passed=passed))
        print(f"[seed {seed}] t={gen_time:.2f}s std={std:.2f} mean={mean:.1f} corr={corr:.3f} "
              f"verdict={'PASS' if passed else 'FAIL'}", flush=True)

    times = [r["time"] for r in results]
    n_pass = sum(1 for r in results if r["passed"])

    print("\n=== SUMMARY ===", flush=True)
    print(f"Samples : {len(results)}", flush=True)
    print(f"mean_s  : {np.mean(times):.2f}", flush=True)
    print(f"std_s   : {np.std(times):.2f}", flush=True)
    print(f"min_s   : {np.min(times):.2f}", flush=True)
    print(f"max_s   : {np.max(times):.2f}", flush=True)
    print(f"pass    : {n_pass}/{len(results)}", flush=True)
    print(f"stds    : {[round(r['std'],1) for r in results]}", flush=True)
    print(f"corrs   : {[round(r['corr'],3) for r in results]}", flush=True)


if __name__ == "__main__":
    main()
