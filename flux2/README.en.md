_[中文版: README.md](README.md)_

# FLUX.2-klein-base-9B multi-device benchmark

> Prompt: `"A cat holding a sign that says hello world"`, guidance 4.0, 50 steps, batch=1, seeds 42–51 (10 seeds total).

## 1. Instances and pricing (AWS on-demand, 2026-05)

| Instance | Chip | Memory | $/hr | Region |
|---|---|---|---:|---|
| **trn2.3xlarge** equivalent | 1× Trainium2 (TP=4, LNC=2) | 96 GB HBM | **$2.235** | ap-southeast-4 (Melbourne) |
| p5.4xlarge | 1× H100 SXM5 | 80 GB HBM3 | **$4.326** | us-east-1 |
| g6.4xlarge | 1× L4 | 24 GB GDDR6 | **$1.323** | sa-east-1 |

> Neuron physically runs on trn2.48xlarge; klein TP=4 only uses a single Trainium2 via `NEURON_RT_VISIBLE_CORES=16-19` (8 physical cores → 4 logical cores, LNC=2). Priced at the trn2.3xlarge single-chip equivalent rate.

## 2. 1024² latency + peak memory + $/image (baseline = H100 FP8)

| Device | Precision | Mean (s) | P95 | Peak VRAM/HBM | Pass | **$/image** | Speed vs H100 FP8 | Cost vs H100 FP8 |
|---|---|---:|---:|---|---:|---:|---:|---:|
| H100 p5.4xlarge | **FP8 (baseline, torchao eager)** | **21.18** | — | 28.25 GB | 10/10 | **$0.02545** | **1.00×** | **1.00×** |
| H100 p5.4xlarge | BF16 | 24.10 | — | 37.33 GB | 10/10 | $0.02896 | 0.88× (1.14× slower) | 1.14× more expensive |
| **Neuron trn2.3xl** | **BF16 TP=4** | **42.90** | 43.28 | **~24 GB** | **10/10** | **$0.02664** | 0.49× (2.03× slower) | 1.05× (5% more expensive) |
| L4 g6.4xlarge | **FP8 (BFL official ckpt)** | 77.25 | — | 13.12 GB | 10/10 | $0.02839 | 0.27× (3.65× slower) | 1.12× (12% more expensive) |

`$/image = (Mean / 3600) × $/hr`

**Key takeaways:**
- **Neuron 1K $/image = $0.0266** — cheaper than H100 BF16 ($0.0290, **8% cheaper**), 5% more expensive than H100 FP8 ($0.0254).
- Absolute speed: Neuron is 2.03× slower than H100 FP8, but the trn2.3xlarge hourly rate is 52% of H100's, so $/image is still only 5% higher than H100 FP8 and beats H100 BF16.
- Neuron single-chip HBM ~24 GB, lower than H100 FP8's 28 GB.

## 3. 2048² latency + peak memory + $/image (baseline = H100 FP8)

| Device | Precision | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | Speed vs H100 FP8 | Cost vs H100 FP8 |
|---|---|---:|---|---:|---:|---:|---:|
| H100 p5.4xlarge | **FP8 (baseline)** | **106.20** | 35.92 GB | 10/10 | **$0.1276** | **1.00×** | **1.00×** |
| H100 p5.4xlarge | BF16 | 107.05 | 45.00 GB | 10/10 | $0.1286 | 0.99× | 1.01× |
| **Neuron trn2.3xl** | **BF16 TP=4** | **191.44** | ~40 GB | **10/10** | **$0.1189** | 0.55× (1.80× slower) | **0.93× (7% cheaper)** |
| L4 g6.4xlarge | **FP8 (BFL official ckpt)** | 388.66 | 17.48 GB | 10/10 | $0.1428 | 0.27× (3.66× slower) | 1.12× (12% more expensive) |

**Key takeaways:**
- At 2K, H100 FP8 / H100 BF16 / Neuron are all within 8% $/image of each other, and **Neuron has the lowest $/image ($0.1189)**.
- FP8 is compute-bound at 2K; its advantage over BF16 shrinks to 1%.
- L4 takes ~15 min/image and VRAM is tight — only feasible for sampling, **not production-ready**.
- After the fix, 2K steady state moved from 196.06 s → 191.44 s (2.4% faster) due to the scalar-modulation fix changing compiler scheduling.

## 4. 4096² feasibility — model spec limit

klein's official `max_area = 4 MP (≈ 2048²)`; 4K = 16 MP is out of spec. **All devices produce `std≈20` noise images (GRAY) — this is a model limitation, not a hardware limit.**

| Device | Result |
|---|---|
| H100 FP8 | 1019 s/image, GRAY noise |
| H100 BF16 | 1024 s/image, GRAY noise |
| Neuron BF16 TP=4 | Compile did not complete (HLO gen timeout, NUM_PATCHES=65536 too large) |
| L4 NF4 / BF16 | OOM |

If 4K is a hard requirement for the customer, they need BFL to release a larger-spec checkpoint.

## 5. DiT load / cold-start / steady-state breakdown (Neuron 1K / 2K, fixed version)

| Stage | 1K | 2K |
|---|---:|---:|
| Compile (one-time, cacheable to S3/EFS) | 156.9 s | 795.2 s |
| Weight load + NxD init | 19.9 s | 136.4 s |
| **Steady-state mean (10 seeds)** | **42.90 s** | **191.44 s** |

- **1K cold start** (no NEFF cache): ~3.4 min; **warm start**: ~65 s; steady state 42.90 s/image.
- **2K cold start**: ~15.5 min; **warm start**: ~5.5 min; steady state 191.44 s/image.

## 6. Same prompt / seed image comparison

### 6.1 1024² seed 42

| H100 BF16 | H100 FP8 | **Neuron trn2 BF16 TP=4** | L4 FP8 |
|:---:|:---:|:---:|:---:|
| ![](task015_klein_jim_pr146/results/h100_1024_bf16/seed42_cat.png) | ![](task015_klein_jim_pr146/results/h100_1024_fp8/seed42_cat.png) | ![](task015_klein_jim_pr146/results/klein_fixed_1k_50step/seed42_cat.png) | ![](task015_klein_jim_pr146/results/l4_1024_fp8/seed42_cat.png) |

### 6.2 2048² seed 42

| H100 BF16 | H100 FP8 | **Neuron trn2 BF16 TP=4** | L4 FP8 |
|:---:|:---:|:---:|:---:|
| ![](task015_klein_jim_pr146/results/h100_2048_bf16/seed42_cat.png) | ![](task015_klein_jim_pr146/results/h100_2048_fp8/seed42_cat.png) | ![](task015_klein_jim_pr146/results/klein_fixed_2k_50step/seed42_cat.png) | ![](task015_klein_jim_pr146/results/l4_2048_fp8/seed42_cat.png) |

**Visual consistency**: at 1K / 2K, every device produces a clear cat holding a "HELLO WORLD" sign. 10/10 seeds pass on every device (no more "decorative painting" variant at seed 42 after the modulation fix); only seed-noise level differences remain.

### 6.3 10-seed full PNG directories

| Device | Directory |
|---|---|
| **Neuron 1K BF16** | `task015_klein_jim_pr146/results/klein_fixed_1k_50step/seed{42..51}_cat.png` |
| **Neuron 2K BF16** | `task015_klein_jim_pr146/results/klein_fixed_2k_50step/seed{42..51}_cat.png` |
| H100 1K BF16 / FP8 | `task015_klein_jim_pr146/results/h100_1024_{bf16,fp8}/seed{42..51}_cat.png` |
| H100 2K BF16 / FP8 | `task015_klein_jim_pr146/results/h100_2048_{bf16,fp8}/seed{42..51}_cat.png` |
| L4 1K / 2K FP8 | `task015_klein_jim_pr146/results/l4_{1024,2048}_fp8/seed{42..51}_cat.png` |

### 6.4 Pixel-level similarity vs H100 BF16 (10-seed average)

| Resolution | Mean PSNR (dB) | Mean SSIM (skimage) |
|---|---:|---:|
| **1K** | **7.91** | **0.368** |
| **2K** | **7.23** | **0.381** |

PSNR of 7–9 dB is the normal range for same-prompt cross-stack diffusion (different bf16 numeric paths + solver step-to-step drift accumulate at the pixel level). It does NOT indicate low output quality.

## 7. Hardware / software config

**Neuron (trn2.3xlarge equivalent)**
- AMI: `ami-042fbe428a1a7a882` (Neuron DLAMI 20260410, Ubuntu 22.04)
- SDK: **2.29** / neuronx-cc 2.24.5133 / torch-neuronx 2.9.0.2.13 / NxDI 0.9.17334 (PR #146 `contrib/flux2-klein`)
- venv: `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/`
- TP: **TP=4**, LNC=2, `NEURON_RT_VISIBLE_CORES=16-19`

**H100 p5.4xlarge**: DLAMI PyTorch 2.9 / CUDA 12.9 / torch 2.9.1+cu128 / diffusers 0.38.0 / FP8 via **torchao** eager mode.

**L4 g6.4xlarge**: DLAMI PyTorch 2.7 / torch 2.8.0+cu128 / **BFL official flux2 repo** (commit `50fe51627778`) + custom FP8 Linear shim (`torch._scaled_mm`, per-tensor E4M3). Checkpoint = `black-forest-labs/FLUX.2-klein-9b-fp8` (guidance-distilled, natively 4-step, forced to 50-step here). Qwen3-8B-FP8 text encoder is loaded in stages to avoid OOM; AE is cast to BF16 for 2K decode (FP32 AE's intermediate feature map is 4 GB at 2K and OOMs).

**klein implementation source**: AWS NxDI [PR #146](https://github.com/aws-neuron/neuronx-distributed-inference/pull/146) by Jim Burtoft (AWS), branch `contrib/flux2-klein`. Re-run at the customer spec (50 steps / 10 seeds) with multi-device GPU alignment.

## 8. Reproduction scripts

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_RT_VISIBLE_CORES=16-19          # one Trainium2's 4 logical cores
export NEURON_COMPILED_ARTIFACTS=$KLEIN_CACHE

python task015_klein_jim_pr146/bench_klein_1k_50step.py \
    --model /mnt/nvme/flux2_klein \
    --out   /mnt/nvme/klein_bench_1k_50step \
    --seeds 42 43 44 45 46 47 48 49 50 51 \
    --prompt "A cat holding a sign that says hello world" \
    --guidance 4.0 --steps 50 --tp 4
```

2K / 4K equivalents: `bench_klein_2k.py` / `bench_klein_4k.py`.

## 9. HBM virtualization (1/2, 1/4) — N/A

Trainium2 does **not** have NVIDIA MIG-style hardware HBM partitioning. Neuron uses **LNC (Logical Neuron Core)** to partition **compute cores** (`NEURON_LOGICAL_NC_CONFIG=2` fuses 8 physical cores into 4 logical cores). The 96 GB HBM is shared across a single-device workload, not hardware-partitioned. Multi-tenant concurrency requires **separate Python processes with non-overlapping `NEURON_RT_VISIBLE_CORES` subsets** — application-level partitioning.

In this benchmark, klein TP=4 fully occupies one Trainium2 (HBM usage ~24 GB @ 1K / ~40 GB @ 2K). The remaining HBM could in principle host a second instance, but that requires LNC=1 + a separate Python process — out of scope for this round.

## 10. Conclusions

1. **Neuron klein passes 1K / 2K × 50 steps × 10 seeds**: `std` = 55–95, visually clear cat + "HELLO WORLD" sign. Seed 42 is also normalized after the modulation fix (no longer the "decorative painting" variant seen in earlier versions).
2. **$/image is 8% cheaper at 1K and 7% cheaper at 2K vs H100 BF16**; 5% more expensive at 1K but 7% cheaper at 2K vs H100 FP8.
3. **Speed**: Neuron is 2.03× slower than H100 FP8 at 1K and 1.80× slower at 2K, but the trn2.3xlarge hourly rate is 52% of H100's, so overall $/image stays competitive.
4. **HBM usage**: Neuron uses 24 GB at 1K, below H100 FP8's 28 GB — good for small-VRAM / high-concurrency deployment.
5. **4K not viable** (all devices, including H100, produce noise images — this is the model-spec `max_area=4MP` limit, not a hardware issue).
6. **L4 FP8 is feasible but not cost-advantaged**: BFL official `klein-9b-fp8` + `torch._scaled_mm` shim, 1K 77.25 s / 2K 388.66 s, 10/10 pass; $/image = $0.0284 / $0.1428 — 7% more expensive than Neuron at 1K, 20% more expensive at 2K; 12% more expensive than H100 FP8 at both. The checkpoint is guidance-distilled (natively 4-step) but no obvious artifacts appear in this prompt at 50 steps.
7. **Capacity availability**: p5 is `InsufficientInstanceCapacity` across us-east / us-west / eu / sa-east; only ap-northeast-1c locked in. trn2 capacity blocks are easier to obtain — combined with the cost-performance story, **Neuron trn2 is the more production-scalable choice for FLUX.2-klein**.
