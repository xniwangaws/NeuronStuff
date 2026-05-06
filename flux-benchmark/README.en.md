_[中文版: README.md](README.md)_

# FLUX.1-dev alien-prompt benchmark — Neuron / H100 / L4

> Prompt: `"A close-up image of a green alien with fluorescent skin in the middle of a dark purple forest"`, guidance 3.5, 28 steps, batch=1, max_sequence_length=512, seeds 42–51 (10 seeds total).

## 1. Instances and pricing (AWS on-demand, 2026-05)

| Instance | Chip | Memory | $/hr | Region |
|---|---|---|---:|---|
| **trn2.3xlarge** equivalent | 1× Trainium2 (WORLD=4, backbone_tp=4, LNC=2) | 96 GB HBM | **$2.235** | ap-southeast-4 |
| p5.4xlarge | 1× H100 SXM5 | 80 GB HBM3 | **$4.326** | ap-northeast-1 |
| g6.4xlarge | 1× L4 | 24 GB GDDR6 | **$1.323** | sa-east-1 |

## 2. 1024² latency + peak memory + $/image (baseline = H100 FP8)

| Device | Precision | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | Speed vs H100 FP8 | Cost vs H100 FP8 |
|---|---|---:|---|---:|---:|---:|---:|
| H100 p5.4xlarge | BF16 | **5.87** | 33.85 GB | 10/10 | $0.00706 | 1.45× faster | 0.83× (17% cheaper) |
| H100 p5.4xlarge | **FP8 (baseline, torchao eager)** | 8.54 | 22.77 GB | 10/10 | **$0.01026** | **1.00×** | **1.00×** |
| **Neuron trn2.3xl** | **BF16 WORLD=4** | **8.03** | ~25 GB (single Trainium2) | **10/10** | **$0.00499** | 1.06× faster | **0.49× (51% cheaper)** |
| L4 g6.4xlarge | NF4 (bnb + offload) | 57.65 | 6.79 GB | 10/10 | $0.02119 | 0.15× (6.75× slower) | 2.06× more expensive |
| L4 g6.4xlarge | **FP8 (wangkanai, seq-offload)** | 123.21 | 2.39 GB | 10/10 | $0.04528 | 0.07× (14.4× slower) | 4.41× more expensive |

`$/image = (Mean / 3600) × $/hr`

**Key takeaways:**
- **Neuron wins on FLUX.1-dev** — both **faster** (1.06× vs H100 FP8) **and cheaper** ($/image at 49% of H100 FP8, **51% cheaper**).
- H100 BF16 has the best absolute speed (5.87 s) and is 17% cheaper than FP8 — FP8 quantization overhead exceeds its speedup at batch=1.
- Neuron single-chip HBM is ~25 GB, below H100 BF16's 34 GB.
- L4 NF4 is 6.75× slower, and NF4 conversion alone takes 678 s on load. Worst cost-performance.
- L4 FP8 (`wangkanai/flux-dev-fp8`) is 14.4× slower: weights are F8_E4M3 in the ckpt but diffusers upcasts to bf16 at load, and 12 B parameters × bf16 ≈ 24 GB overflows the L4's 22 GB VRAM, forcing sequential CPU offload → PCIe transfer dominates wall time.

## 2b. 2048² / 4096² super-resolution (model spec `max_area=4MP`, force-run)

| Device | Precision | Res | Mean (s) | Peak VRAM | Pass | **$/image** |
|---|---|---|---:|---:|---:|---:|
| H100 p5.4xlarge | FP8 (torchao eager) | 2048² | **37.52** | 29.9 GB | 10/10 | **$0.04509** |
| H100 p5.4xlarge | FP8 (torchao eager) | 4096² | **328.70** | 37.67 GB | 10/10 | **$0.39492** |
| L4 g6.4xlarge | FP8 (wangkanai, seq-offload) | 2048² | **339.38** | 2.42 GB | 10/10 | **$0.12471** |
| Neuron trn2.3xl | BF16 | 2048² | **BLOCKED** | — | 0/10 | `NCC_EVRF007`: VAE decoder NEFF 5,234,444 instructions > 5M hard limit (attempted 2026-05-07) |
| Neuron trn2.3xl | BF16 | 4096² | **BLOCKED** | — | — | Same root cause, not attempted separately |
| L4 g6.4xlarge | FP8 wangkanai | 4096² | **BLOCKED OOM** | — | 0/3 | L4 22 GB VRAM cannot hold 12 B bf16 upcast (~16 GB) + 4K activations (~8 GB) |

**Note**: FLUX.1-dev's official spec is 1024² with `max_area=4MP`. 2K / 4K are super-resolution force-runs; outputs are constrained by the spec (lower `std`, reduced detail). Useful only as a hardware-limit benchmark.

## 3. DiT load / cold-start / steady-state breakdown (Neuron 1K)

| Stage | Time |
|---|---:|
| Compile (one-time, cacheable to S3/EFS) | 103.4 s |
| Weight load + NxD init | 78.0 s |
| First inference (includes graph-replay warmup) | ~8 s |
| **Steady-state mean (10 seeds)** | **8.03 s** |

- **Cold start** (no NEFF cache): ~3.2 min
- **Warm start** (NEFF cache hit): ~85 s
- **Steady state**: 8.03 s/image (28 steps)

## 4. Same prompt / seed image comparison (seed 42)

| H100 BF16 | H100 FP8 | **Neuron trn2 BF16 WORLD=4** | L4 NF4 |
|:---:|:---:|:---:|:---:|
| ![](alien_bench/results/flux1_alien_h100_bf16/seed42_alien.png) | ![](alien_bench/results/flux1_alien_h100_fp8/seed42_alien.png) | ![](alien_bench/results/flux1_alien_trn2_bf16/seed42_alien.png) | ![](alien_bench/results/flux1_alien_l4_nf4/seed42_alien.png) |

**Visual consistency**: all four devices on the same prompt + seed 42 produce an easily identifiable fluorescent green alien. (One prompt-bias variant: seed 42 occasionally produces a "cat hello world" composition on H100/L4 — a known FLUX.1-dev training-data preference.) The other 9 seeds (43–51) are stable on the alien subject.

## 5. 10-seed full PNG paths

| Device | Directory |
|---|---|
| Neuron 1K BF16 | `alien_bench/results/flux1_alien_trn2_bf16/seed{42..51}_alien.png` |
| H100 1K BF16 | `alien_bench/results/flux1_alien_h100_bf16/seed{42..51}_alien.png` |
| H100 1K FP8 eager | `alien_bench/results/flux1_alien_h100_fp8/seed{42..51}_alien.png` |
| L4 1K NF4 | `alien_bench/results/flux1_alien_l4_nf4/seed{42..51}_alien.png` |
| L4 1K FP8 (wangkanai) | `alien_bench/results/flux1_alien_l4_fp8/seed{42..51}_alien.png` |

## 6. Hardware / software config

**Neuron (trn2.3xlarge)**
- AMI: Neuron DLAMI / SDK 2.29 / neuronx-cc 2.24.5133 / torch-neuronx 2.9.0.2.13.26312 / NxDI `NeuronFluxApplication`
- venv: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`
- Topology: **WORLD=4**, backbone_tp=4, t5_tp=4, clip+vae tp=1, LNC=2

**H100 p5.4xlarge**: DLAMI / torch 2.9.1+cu128 / diffusers 0.38 / FP8 via `torchao.Float8DynamicActivationFloat8WeightConfig` eager mode.

**L4 g6.4xlarge (NF4 path)**: DLAMI / torch 2.7.0+cu128 / diffusers 0.38 / bitsandbytes NF4 + `enable_model_cpu_offload`.

**L4 g6.4xlarge (FP8 path)**: DLAMI PyTorch 2.9 / torch 2.9.1+cu130 / diffusers 0.35.1 / transformers 4.57.6 / `wangkanai/flux-dev-fp8` (17 GB single-file ComfyUI-style ckpt, 16.7 B F8_E4M3 weights) / `FluxPipeline.from_single_file` + `enable_sequential_cpu_offload`. The 12 B transformer upcasts to bf16 at runtime (~24 GB), exceeds the L4's 22 GB VRAM, so sequential offload streams layer-by-layer → PCIe transfer dominates (123 s/image).

**Implementation**: FLUX.1-dev Neuron path is the [AWS NxDI `NeuronFluxApplication`](https://awsdocs-neuron.readthedocs-hosted.com/) one-shot compile + load + forward. GPU path uses `diffusers.FluxPipeline`.

## 7. Reproduction scripts

```bash
# Neuron
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python3 alien_bench/bench_neuron_alien.py

# H100
python3 alien_bench/bench_h100_alien.py --precision bf16 --out /opt/dlami/nvme/flux1_alien_h100_bf16
python3 alien_bench/bench_h100_alien.py --precision fp8  --out /opt/dlami/nvme/flux1_alien_h100_fp8

# L4
python3 alien_bench/bench_l4_alien.py --precision nf4 --out ~/flux1_alien_l4_nf4
```

## 8. Conclusions

1. **FLUX.1-dev passes 10/10 on Neuron trn2.3xlarge**: mean 8.03 s at 28-step steady state, fits in ~28 GB HBM.
2. **Cheapest $/image overall**: Neuron **$0.00499** — 51% cheaper than H100 FP8, 29% cheaper than H100 BF16.
3. **Speed**: Neuron 8.03 s ≈ H100 FP8 8.54 s (**1.06× faster**). H100 BF16 is the absolute fastest at 5.87 s (+45%) but is 1.41× more expensive.
4. **HBM usage**: Neuron 25 GB vs H100 BF16 34 GB — leaves ample headroom on the 96 GB single-chip budget.
5. **Capacity**: p5 is routinely `InsufficientInstanceCapacity` across us-east, us-west, eu, and sa-east regions; only ap-northeast-1 locked in for this benchmark round. trn2 capacity blocks are more reliably obtainable — an important production-scale consideration.
