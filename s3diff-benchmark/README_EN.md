# S3Diff Super-Resolution Benchmark — AWS Trainium2 vs NVIDIA L4

**Model**: S3Diff (ECCV 2024) — one-step diffusion 4× super-resolution (SD-Turbo + LoRA)
**Test**: bus image, 4× SR:
- **0.5K**: input 128×128 → output 512×512 (single tile)
- **1K**: input 256×256 → output 1024×1024 (9 tiles)
**Precision**: BF16

> Trn2 uses [AWS Neuron PR #149](https://github.com/aws-neuron/neuronx-distributed-inference/pull/149) (Jim Burtoft) `torch_neuronx.trace()` + fixed 512-pixel tile + Gaussian blending.

---

## ⭐ Results Summary

### Main table (warm mean / cost per image)

| Device | 128→512 | 256→1024 | Pass |
|---|---|---|---|
| **Neuron trn2.3xl PR 149 (BF16, whole instance)** | 0.545s / $0.00034 | 4.81s / $0.00299 | 5/5 |
| **Neuron trn2.3xl PR 149 (BF16, ÷4 pricing)** | 0.545s / **$0.00008** | 4.81s / **$0.00075** | 5/5 |
| L4 24GB BF16 | 0.914s / $0.00034 | 2.34s / $0.00086 | 5/5 |

### Cost efficiency (L4 = 100% baseline)

| Device | 128→512 cost eff | 256→1024 cost eff |
|---|---|---|
| **Neuron trn2.3xl whole instance** | 100% | 29% |
| **Neuron trn2.3xl (÷4 pricing)** | **397%** ⭐ | **115%** ⭐ |
| **L4 24GB** | **100%** (ref) | **100%** (ref) |

> Efficiency = L4 $/img ÷ device $/img. Higher is cheaper.

### Key findings

- **128→512 (single tile)**: Trn2 is **1.68× faster** than L4 (0.545s vs 0.914s). Whole-instance cost is on par with L4 ($0.00034 each); with ÷4 pricing Trn2 is **4× cheaper**
- **256→1024 (9 tiles)**: Trn2 is 2× slower than L4 (4.81s vs 2.34s) because the fixed 512 tile produces 9 tiles for 1K HR, while GPU runs in a single pass; whole-instance is 3.5× more expensive, but with ÷4 pricing Trn2 is still **15% cheaper**
- **Crossover point**: Trn2 wins both speed and cost at small resolutions (single-tile path); past the multi-tile threshold Trn2 falls behind in absolute speed, requiring 4-way concurrent pricing to beat L4 on cost
- **Tile counts**: 128→512 = 1 tile (HR=512 ≤ 512 tile_size), 256→1024 = 9 tiles (HR=1024, stride=384, 3×3 grid)
- **Quality**: Trn2 1K PSNR 43.10 dB vs CPU fp32 (BF16 matmul accumulation noise ~2 dB, visually indistinguishable)

### Raw data

Cost formula: `$/img = $/hr × warm_s / 3600`
- trn2.3xlarge whole = $2.235/hr (capacity block); ÷4 pricing = $0.559/hr (4-way concurrent)
- g6.4xlarge (L4) = $1.323/hr (on-demand)

---

## 1. Detailed Test Parameters

| Parameter | Value |
|---|---|
| Batch size | 1 |
| Diffusion steps | 1 (S3Diff is a one-step model) |
| Scheduler | DDPMScheduler |
| Seed | 123 (fixed, cross-stack comparable) |
| Positive prompt | "high quality, highly detailed, clean" |
| Negative prompt | "blurry, dotted, noise, raster lines, unclear, lowres, over-smoothed" |
| Guidance scale | 1.07 |
| Tile (Trn2 PR 149) | fixed pixel tile=512, overlap=128, Gaussian blending |
| Compile (Trn2) | `torch_neuronx.trace()` on 5 components (DEResNet / Text Enc / VAE Enc / UNet / VAE Dec) |
| Compile flags (LoRA components) | `--model-type=unet-inference -O1` (matmult causes NaN in LoRA einsum) |
| Total compile time | ~1357s (~22.6 min, one-time) |
| Test image | bus (source `https://ultralytics.com/images/bus.jpg`) |

---

## 2. Bus SR Samples

Test image: `https://ultralytics.com/images/bus.jpg`

### 2.1 Input LQ

256×256 LQ input (for 1K output test):

![Bus LQ 256](customer_report/images/input_bus_LQ_256.png)

### 2.2 Trn2 PR 149 1K output (1024×1024, 4× SR)

![Trn2 Bus 1K](customer_report/images/trial6_bus_1k.png)

---

## 3. Hardware / Software Config

### AWS Trn2.3xlarge

| Field | Value |
|---|---|
| Instance type | trn2.3xlarge (LNC=2, 4 logical NeuronCores, 96 GB HBM) |
| vCPU / Memory | 12 vCPU / 96 GB |
| AMI | Deep Learning AMI Neuron (Ubuntu 24.04) 20260502 |
| Neuron SDK | 2.29 |
| Python | 3.12 |
| PyTorch | 2.9.1 |
| torch-neuronx | 2.9.0.2.13.26312+8e870898 |
| neuronx-cc | 2.0.243879.0a0+866424ce |
| diffusers | 0.34.0 |
| peft | 0.19.1 |
| transformers | 4.57.3 |

### NVIDIA L4

- AWS g6.4xlarge (1× L4 24GB)
- PyTorch 2.10 + CUDA 13.0
- diffusers 0.34

---

## 4. Reproduction Commands

### Trn2 (PR 149 path)

```bash
# Code source: https://github.com/aws-neuron/neuronx-distributed-inference/pull/149
# Files: contrib/models/S3Diff/src/{modeling_s3diff.py, generate_s3diff.py}

source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# 0.5K (128 → 512, single tile)
python generate_s3diff.py \
    --input_image <path_to_LQ_128.png> \
    --output_image /tmp/out_512.png \
    --compile_dir /tmp/s3diff_pr149_compiled \
    --num_images 5 --warmup_rounds 2 \
    --tile_size 512 --tile_overlap 128

# 1K (256 → 1024, 9 tiles)
python generate_s3diff.py \
    --input_image <path_to_LQ_256.png> \
    --output_image /tmp/out_1k.png \
    --compile_dir /tmp/s3diff_pr149_compiled \
    --num_images 5 --warmup_rounds 2 \
    --tile_size 512 --tile_overlap 128
```

### L4 (native diffusers)

S3Diff repo default `inference_s3diff.py` + `accelerate launch --mixed_precision=bf16`.

---

## 5. Directory Structure

| Path | Content |
|---|---|
| `README.md` | Chinese report |
| `README_EN.md` | English report (this file) |
| `customer_report/data/s3diff_benchmark.csv` | Raw benchmark data |
| `customer_report/images/` | bus LQ input + output images |
| `backup/jim_port_attempt/` | Jim's early notebook port (had NaN with `--auto-cast=matmult`, superseded by PR 149 latest) |
| `backup/phase3/`, `backup/phase_e/`, `backup/phase_r/`, `backup/phase_b/` | Earlier Trial 6 + experimental backups |
