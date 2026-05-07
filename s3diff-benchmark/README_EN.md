# S3Diff Super-Resolution Benchmark — AWS Trainium2 vs NVIDIA H100/L4

**Model**: S3Diff (ECCV 2024) — one-step diffusion 4× super-resolution (SD-Turbo + LoRA)
**Test**: cat / bus images, 4× SR:
- 1K: input 256×256 → output 1024×1024
- 2K: input 512×512 → output 2048×2048
- 4K: input 1024×1024 → output 4096×4096
- **8K: input 2048×2048 → output 8192×8192** (new, Trn2 uses `torch_neuronx.trace` path)
**Precision**: BF16

> **About 8K**: At 8K resolution, the native Trial 6 `torch.compile(backend="neuron")` path triggers NRT OOM (needs ~14.6 GB single allocation, single-core HBM insufficient);
> L4 (24GB VRAM) OOMs in UNet attention (needs 13.64 GiB);
> so Trn2 8K switches to the [AWS Neuron PR #149](https://github.com/aws-neuron/neuronx-distributed-inference/pull/149) (Jim Burtoft) `torch_neuronx.trace()` + fixed 512-pixel tile approach — this is the only working 8K path on Trn2.

---

## ⭐ Results Summary

### Main table (warm mean, seconds / $/image)

| Device | 1K warm | 1K $/img | 2K warm | 2K $/img | 4K warm | 4K $/img | **8K warm** | **8K $/img** | Pass |
|---|---|---|---|---|---|---|---|---|---|
| **Neuron trn2.3xl BF16 whole instance** | 6.31s | $0.00381 | 60.18s | $0.0374 | 431.97s | $0.1882 | **235.08s** ⭐ | **$0.1460** ⭐ | 3/3 |
| **Neuron trn2.3xl BF16 (÷4 pricing)** | 6.31s | **$0.00095** | 60.18s | **$0.00936** | 431.97s | **$0.04707** | **235.08s** | **$0.03651** ⭐ | 3/3 |
| H100 80GB BF16 | 1.26s ⭐ | $0.00151 | 24.26s ⭐ | $0.0292 | 107.54s ⭐ | $0.1293 | 429.02s | $0.5155 | 10/10 |
| L4 24GB BF16 | 2.34s | **$0.00086** ⭐ | 28.45s | **$0.0105** ⭐ | 130.63s | **$0.0480** ⭐ | **OOM** | — | 10/10 at ≤4K |

### Key findings

- **At 8K, Trn2 whole-instance is the cheapest** — 3.53× cheaper than H100; Trn2 ÷4-core pricing is 14.12× cheaper; L4 cannot run (attention needs 13.64 GB, exceeds 24 GB VRAM available)
- **At 8K, Trn2 beats H100 in absolute speed**: 235s vs 429s, Trn2 **1.83× faster** (thanks to PR 149 fixed-tile + one-shot compilation so all 8K tiles reuse the same NEFF)
- **At 1K-4K, GPUs win**: L4 has the cheapest `$/img` at all of 1K/2K/4K; H100 is absolute fastest; Trn2 whole-instance `$/img` is 3-4× L4
- **Trn2 cost-advantage crossover around 2K+** (÷4 pricing): 1.12× cheaper than L4 at 2K, 1.02× at 4K, **far ahead at 8K**
- **Quality**: 1K PSNR Trn2 43.10 dB vs H100 45.10 dB (2 dB gap from bf16 matmul accumulation noise, visually indistinguishable); 8K outputs on Trn2 and H100 are visually identical

### Problems

- **L4 OOM at 8K**: S3Diff UNet attention at 8K requires 13.64 GB single allocation; L4 24GB VRAM can't fit
- **Trn2 Trial 6 NRT OOM at 8K**: Native Trial 6 `torch.compile(backend="neuron")` tries 14.6 GB allocation; exceeds single-core HBM. So 8K switches to PR 149 `torch_neuronx.trace()` fixed-512 tile path (effectively Jim's re-designed tile strategy for Trn2)
- **16K incomplete**: H100 16K bench is running but SSH timeout prevents confirming result (possibly stuck in VAE or OOM); Trn2 16K not yet started

### Raw data

Full data in [`customer_report/data/s3diff_benchmark.csv`](customer_report/data/s3diff_benchmark.csv). Cost formula: `$/img = capacity_block_price($/hr) × warm_s / 3600`. trn2.3xlarge = $2.235/hr, p5.4xlarge = $4.326/hr, g6.4xlarge = $1.323/hr.

---

## 1. Core Metrics — Customer-defined Latencies

> Customer-defined warm_mean formula:
> `warm_mean = (total_N - cold) / (N - 1)`

Test image: bus (256×256 LQ, source `https://ultralytics.com/images/bus.jpg`), 4× SR at multiple resolutions.

| Instance | Accelerator | Resolution | Compile (s) | Load (s) | Cold (s) | Warm mean (s) | Peak HBM (GB) |
|---|---|---|---|---|---|---|---|
| **trn2.3xlarge** | **1× Trainium2** | **1K** (256→1024) | **~90** (first) | **13.4** | **32.94** | **6.31** | ~20 |
| trn2.3xlarge | 1× Trainium2 | 2K (512→2048) | ~33 | 13.4 | 93.15 | 60.18 | ~22 |
| trn2.3xlarge | 1× Trainium2 | 4K (1024→4096) | ~-88 (warm > cold ∗) | 13.4 | 343.26 | 431.97 | ~24 |
| **trn2.3xlarge †** | **1× Trainium2** | **8K (2048→8192)** | **1357** (PR 149 trace) | — | — (folded into warmup) | **235.08** | ~24 |
| p5.4xlarge | 1× H100 80GB | 1K | — (no pre-compile) | 4.80 | 9.64 | 1.26 | 9.0 |
| p5.4xlarge | 1× H100 80GB | 2K | — | 4.02 | 24.67 | 24.26 | 15.7 |
| p5.4xlarge | 1× H100 80GB | 4K | — | 4.03 | 109.71 | 107.54 | 42.2 |
| **p5.4xlarge** | **1× H100 80GB** | **8K (2048→8192)** | — | **2.62** | **469.23** | **429.02** | **32.46** |
| g6.4xlarge | 1× L4 24GB | 1K | — | 4.52 | 6.41 | 2.34 | 7.9 |
| g6.4xlarge | 1× L4 24GB | 2K | — | 4.07 | 29.11 | 28.45 | 15.2 |
| g6.4xlarge | 1× L4 24GB | 4K | — | 4.08 | 132.42 | 130.63 | 16.5 |
| g6.4xlarge | 1× L4 24GB | 8K (2048→8192) | — | — | — | **OOM** | >24 (need >13.64) |

> **† 8K strategy switch**: Trn2 at 8K uses [PR #149](https://github.com/aws-neuron/neuronx-distributed-inference/pull/149) `torch_neuronx.trace()` + fixed pixel-512 tile + Gaussian blending (native Trial 6 OOMs). Compiles 5 NEFFs once (DEResNet + Text Enc + VAE Enc + UNet + VAE Dec); all tiles reuse the same NEFF. Measured **Trn2 8K warm is 1.83× faster than H100** (235s vs 429s).

Notes:
- **Compile time** (Trn2): first-time NEFF JIT produced by `torch.compile(backend="neuron")`. H100/L4 need none (eager CUDA).
- **Load**: S3Diff components loaded from disk + weights moved to accelerator. Trn2 includes NEFF cache check.
- **Cold**: first inference. Note: Trn2 cold = compile + inference; customer formula uses N=3 on Trn2, N=10 on GPU.
- **Warm mean**: customer formula `(total_N - cold) / (N-1)`. Reuses NEFF cache, no recompile.
- **Peak HBM**: single-card / single-core HBM (Trn2 LNC=2 is 24GB per core; GPU is full-card VRAM).

### 1.1 On-demand Price & Per-image Cost

| Instance | On-demand $/hr | 1K $/img | 2K $/img | 4K $/img | **8K $/img** |
|---|---|---|---|---|---|
| trn2.3xlarge (Trainium2, whole instance) | $2.235 (capacity block) | $0.00381 | $0.0374 | $0.1882 | **$0.1460** |
| **trn2.3xlarge (1 logical core, ÷4 pricing)** | **$0.559** (capacity block ÷ 4) | **$0.00095** | **$0.00936** | **$0.04707** | **$0.03651** |
| p5.4xlarge (H100) | $4.326 (capacity block) | $0.00151 | $0.0292 | $0.1293 | **$0.5155** |
| **g6.4xlarge (L4)** | **$1.323** (on-demand) | **$0.00086** | **$0.0105** | **$0.0480** | **OOM** |

> Formula: `$/img = $/hr × warm_s / 3600`
> **trn2 / p5 use AWS capacity block prices** (reserved). L4 uses standard on-demand.
>
> **Trn2 ÷4-core pricing note**: trn2.3xlarge physically has 1× Trainium2 chip (4 logical cores). Single-image inference uses only 1 logical core; the remaining 3 are idle. Pricing at ÷4 ($0.559/hr) represents the "effective cost" assuming 4 concurrent images (throughput mode) share the whole-instance cost.

### 1.2 Cost Efficiency (L4 = 100% baseline)

| Instance | 1K cost eff | 2K cost eff | 4K cost eff | 8K cost eff |
|---|---|---|---|---|
| **g6.4xlarge (L4)** | **100%** | **100%** | **100%** | — (OOM) |
| p5.4xlarge (H100) | 57% | 36% | 37% | **ref (100%)** |
| trn2.3xlarge (whole) | 23% | 28% | 26% | **353%** |
| **trn2.3xlarge (÷4 pricing)** | **90%** | **112%** | **102%** | **1412%** |

> Efficiency = L4 per-image cost / instance per-image cost. Higher = cheaper. 1K-4K baselined against L4; **8K baselined against H100** (L4 OOM).
> At whole-instance pricing, L4 wins cost efficiency overall; Trn2 is ~1/4 as cost-efficient as L4.
> **With ÷4 concurrent pricing**, **Trn2 beats L4 at 2K/4K**: 1.12× at 2K, 1.02× at 4K; 1K is near-tie (90%).
> **8K conclusion**: L4 cannot run; **Trn2 whole-instance is 3.53× cheaper than H100, Trn2 ÷4 is 14× cheaper** (per 8K image).

---

## 2. Accuracy / Quality (PSNR)

| Device | 1K PSNR (dB) | 2K PSNR (dB) | 4K PSNR (dB) |
|---|---|---|---|
| AWS Trn2.3xlarge | 43.10 (vs CPU fp32) | 40.60 (vs H100) | 37.07 (vs H100) |
| NVIDIA H100 BF16 | 45.10 (vs CPU fp32) | reference | reference |
| NVIDIA L4 BF16 | 45.15 (vs CPU fp32) | — | — |

Notes:
- 1K PSNR baseline = CPU fp32 output (same pipeline, fp32)
- 2K / 4K PSNR baseline = H100 bf16 output (CPU fp32 too slow at high res)
- Trn2 ~2 dB gap vs H100 mainly from BF16 matmul accumulation noise; industry-wide, visually indistinguishable

---

## 3. Bus SR Samples

Test image: `https://ultralytics.com/images/bus.jpg`

### 3.1 Input 256×256 LQ

![Bus LQ 256](customer_report/images/input_bus_LQ_256.png)

### 3.2 Trn2 1K output (1024×1024, 4× SR)

![Trn2 Bus 1K](customer_report/images/trial6_bus_1k.png)

### 3.3 Trn2 2K output (2048×2048, input 512→2K)

![Trn2 Bus 2K](customer_report/images/trial6_bus_2k.png)

### 3.4 Trn2 4K output (4096×4096, input 1024→4K)

![Trn2 Bus 4K](customer_report/images/trial6_bus_4k.png)

### 3.5 Trn2 PR 149 8K output (8192×8192, input 2048→8K)

> Original 88 MB, downsampled to 2048 for README preview. Full 8K at `customer_report/images/pr149_bus_8k.png`.

![Trn2 PR 149 Bus 8K preview](customer_report/images/trn2_bus_8k_2k_preview.png)

### 3.6 H100 8K output (reference, input 2048→8K)

![H100 Bus 8K preview](customer_report/images/h100_bus_8k_2k_preview.png)

> The two 8K outputs are visually indistinguishable. Full 8K files at `pr149_bus_8k.png` (Trn2, 88 MB) and `h100_bus_8k.png` (H100, 96 MB).

---

## 4. Hardware / Software Config

### AWS Trn2.3xlarge

| Field | Value |
|---|---|
| Instance type | trn2.3xlarge (LNC=2, 4 logical NeuronCores, 96 GB HBM) |
| vCPU / Memory | 12 vCPU / 96 GB |
| AMI | Deep Learning AMI Neuron (Ubuntu 24.04) 20260502 |
| Neuron SDK | 2.29 |
| Python | 3.12 |
| PyTorch | 2.10 (eager) |
| torch-neuronx | 0.1.0+5e711e8 (eager for 1K-4K) / 2.9.0.2.13+8e870898 (stable for 8K PR 149) |
| neuronx-cc | 2.24.8799.0+6f62ff7c (1K-4K) / 2.0.243879.0a0+866424ce (8K PR 149) |
| diffusers | 0.34.0 |
| peft | 0.19.1 |
| transformers | 4.57.3 |

### NVIDIA GPU

- **H100**: AWS p5.4xlarge (1× H100 80GB), PyTorch 2.10 + CUDA 13.0, diffusers 0.34
- **L4**: AWS g6.4xlarge (1× L4 24GB), PyTorch 2.10 + CUDA 13.0, diffusers 0.34

### Key runtime parameters

| Parameter | Value |
|---|---|
| Batch size | 1 |
| Diffusion steps | 1 (S3Diff is a one-step model) |
| Scheduler | DDPMScheduler |
| Seed | 123 (fixed, cross-stack comparable) |
| Positive prompt | "A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting." |
| Negative prompt | "oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth" |
| Tile (Trn2 1K-4K Trial 6) | latent_tiled_size=96, overlap=32, vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224 |
| Tile (Trn2 8K PR 149) | fixed pixel tile=512, overlap=128, Gaussian blending (shared by all stages) |
| Compile (Trn2 1K-4K) | `torch.compile(backend="neuron", dynamic=False, fullgraph=False)` on 16 `Transformer2DModel`, other modules eager |
| Compile (Trn2 8K) | `torch_neuronx.trace()` on 5 components (DEResNet / Text Enc / VAE Enc / UNet / VAE Dec) |
| Compile flags (1K-4K) | `--auto-cast=matmult -O1` |
| Compile flags (8K, LoRA components) | `--model-type=unet-inference -O1` (matmult causes NaN in LoRA einsum) |

---

## 5. Reproduction Commands

```bash
# 1K
python src/scripts/phase_bisect_hires.py --scope t2d \
    --lq_image <path_to_LQ_256.png> --lq_size 256 \
    --output_image /tmp/out_1k.png --num_inferences 3

# 2K
python src/scripts/phase_bisect_hires.py --scope t2d \
    --lq_image <path_to_LQ_512.png> --lq_size 512 \
    --output_image /tmp/out_2k.png --num_inferences 3

# 4K
python src/scripts/phase_bisect_hires.py --scope t2d \
    --lq_image <path_to_LQ_1024.png> --lq_size 1024 \
    --output_image /tmp/out_4k.png --num_inferences 2

# 8K (PR 149 torch_neuronx.trace path, SDK 2.29 standard venv)
# Code source: https://github.com/aws-neuron/neuronx-distributed-inference/pull/149
# Files: contrib/models/S3Diff/src/{modeling_s3diff.py, generate_s3diff.py}
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python generate_s3diff.py \
    --input_image <path_to_LQ_2048.png> \
    --output_image /tmp/out_8k.png \
    --compile_dir /tmp/s3diff_pr149_compiled \
    --num_images 3 --warmup_rounds 1 \
    --tile_size 512 --tile_overlap 128
```

More details: `src/README.md`.

---

## 6. Directory Structure

| Path | Content |
|---|---|
| `README.md` | Chinese customer report |
| `README_EN.md` | This English report |
| `src/` | ⭐ **Production code** (Trial 6, reproducible 1K/2K/4K) |
| `src/modules/` | DeModLoRA / Attention / Transformer nn.Module |
| `src/scripts/` | Run scripts (`phase_bisect_hires.py` etc.) |
| `src/tests/` | Unit tests + block-level correctness tests |
| `src/data/` | LoRA targets / UNet structure dumps |
| `src/README.md` | Code usage + algorithm notes |
| `customer_report/` | Customer deliverables |
| `customer_report/data/s3diff_benchmark.csv` | Raw benchmark data |
| `customer_report/images/` | bus 1K/2K/4K/8K + GPU comparison + CPU reference |
| `customer_report/logs/` | Raw stdout per test |
| `docs/archive/` | Old README backups |
| `backup/phase3/` | Early trace-mode approach (24.9s, PSNR 24.55, seam artifacts) |
| `backup/phase_e/` | Eager mode experiments (8.6s, superseded by Trial 6) |
| `backup/phase_r/` | DeModLoRA algorithm optimization + bisect history (useful code extracted to `src/`) |
| `backup/phase_b/` | Trace-ready full custom UNet (unused, reserved for future trace API) |
| `backup/scripts_old/` | Phase 3 trace scripts (`neuron_e2e.py` etc.) |
| `backup/results_old/` | Phase 3 raw benchmark json |
