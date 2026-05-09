# Image Generation Models on Trainium2 — Benchmark Summary

* FLUX.1-dev (https://github.com/xniwangaws/NeuronStuff/tree/main/flux-benchmark): 12B DiT model deployed on trn2.3xlarge with NxDI NeuronFluxApplication (BF16 TP=4). At 1K resolution, Neuron achieves 8.03s/image which is 5.2× faster than L4 FP8 (41.4s), and 67% cheaper per image ($0.005 vs $0.015). At 2K, Neuron achieves 93.3s compared to L4 231.6s — 2.5× faster and 32% cheaper. There is further optimization potential: VAE segmented decode (7 NEFFs pre-loaded to HBM) can reduce VAE from 50s to 2.93s, bringing total to ~47s.

* FLUX.2-klein (https://github.com/xniwangaws/NeuronStuff/tree/main/flux2): 9B guidance-distilled DiT deployed on trn2.3xlarge with NxDI PR #146 (BF16 TP=4). Neuron is 1.8× faster at 1K (42.9s vs 76.5s) with cost comparable ($0.027 vs $0.028), and 2.0× faster at 2K (191.4s vs 382.7s) with Neuron 16% cheaper ($0.119 vs $0.141). L4 cannot use torch.compile due to BFL custom code graph breaks. 4K is blocked by model spec (max_area=4MP), not hardware — all devices produce gray noise at 4K.

* SDXL-base-1.0 (https://github.com/xniwangaws/NeuronStuff/tree/main/sdxl-benchmark): 2.6B UNet deployed on trn2.3xlarge with torch_neuronx.trace + DataParallel(DP=2) and NKI flash-attention. At 1K, Neuron is comparable speed (11.14s vs 12.68s) and 26% cheaper ($0.0035 vs $0.0047). However, at 2K L4 wins decisively (74.85s vs 213.9s) — SDXL UNet is too small (2.6B) to benefit from TP=4 parallelism, communication overhead dominates. TP=1 at 2K is not feasible (49GB activation > 24GB HBM per core pair).

## Summary Table

| Model | Params | 1K Neuron | 1K L4 | Neuron vs L4 (1K) | 2K Neuron | 2K L4 | Neuron vs L4 (2K) |
|---|---|---|---|---|---|---|---|
| FLUX.1-dev | 12B | 8.03s / $0.005 | 41.4s / $0.015 | **5.2× faster, 67% cheaper** | 93.3s / $0.058 | 231.6s / $0.085 | **2.5× faster, 32% cheaper** |
| FLUX.2-klein | 9B | 42.9s / $0.027 | 76.5s / $0.028 | **1.8× faster, cost 持平** | 191.4s / $0.119 | 382.7s / $0.141 | **2.0× faster, 16% cheaper** |
| SDXL | 2.6B | 11.14s / $0.0035 | 12.68s / $0.0047 | **1.1× faster, 26% cheaper** | 213.9s / $0.133 | 74.85s / $0.028 | L4 2.9× faster, 79% cheaper |

## Key Insight

Neuron advantage scales with model size: large models (9-12B FLUX) show 1.8-5.2× speedup and significant cost savings; small models (2.6B SDXL) at high resolution lose to L4 due to TP communication overhead exceeding parallelism benefit.
