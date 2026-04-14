---
title: "AWS CV Model Inference Benchmark Report"
subtitle: "Trainium2 vs H100 vs Trainium1 vs L4"
date: "2026-04-14"
---

# AWS CV Model Inference Benchmark Report

## 1. Overview

This report presents inference performance benchmarks for computer vision models on AWS accelerators, including Trainium2, Trainium1, NVIDIA H100, and NVIDIA L4. The primary focus is on diffusion-based image generation models (FLUX.1-dev, SDXL, FLUX.2-dev), with benchmarks covering end-to-end latency, cost efficiency, and quantization strategies.

## 2. Test Environment

### 2.1 Instance Specifications

| Instance | Accelerator | VRAM | On-Demand $/hr |
|----------|------------|------|----------------|
| trn2.48xlarge | 2x Trainium2 chips (tp=4, cp=2) | 96GB HBM per chip | $35.76 (full) / ~$4.47 (2 chips) |
| p5.48xlarge | 1x H100 80GB | 80GB HBM3 | $34.61 (full) / ~$4.33 (1 GPU) |
| trn1.32xlarge | 2x Trainium1 chips (tp=4, cp=2) | 32GB HBM per chip | $21.50 (full) / ~$2.69 (2 chips) |
| g6.4xlarge | 1x L4 24GB | 24GB GDDR6 | $1.32 |

### 2.2 Software Stack

| Component | GPU Instances | Neuron Instances |
|-----------|--------------|-----------------|
| OS | Ubuntu 22.04 | Ubuntu 24.04 (Neuron DLAMI) |
| PyTorch | 2.6.0+cu126 | 2.5.1 (torch-neuronx) |
| diffusers | 0.37.1 | N/A (uses NxDI) |
| torchao | 0.17.0 | N/A |
| bitsandbytes | 0.45.5 | N/A |
| NxDI | N/A | neuronx-distributed-inference (2.27.0) |
| CUDA | 12.6 | N/A |

## 3. Test Methodology

### 3.1 Unified Parameters

| Parameter | Value |
|-----------|-------|
| Resolution | 1024 x 1024 |
| Guidance scale | 3.5 |
| Max sequence length | 512 |
| Prompt | "A cat holding a sign that says hello world" |
| Seed | 0 (`torch.Generator("cpu").manual_seed(0)`) |
| Warmup rounds | 5 |
| Inference steps | 15, 25, 50 |
| Precision | bfloat16 (unless noted) |

### 3.2 Timing Method

- **Warmup**: 5 full inference runs discarded to stabilize GPU/accelerator state
- **Timed run**: Single inference run measured end-to-end (prompt encoding + denoising + VAE decode)
- **Neuron**: Additional one-time compilation step (not included in timing)

## 4. FLUX.1-dev Results

### 4.1 Performance Comparison

| Instance | Accelerator | Method | 15 steps | 25 steps | 50 steps |
|----------|------------|--------|----------|----------|----------|
| trn2.48xlarge | 2x Trainium2 (tp=4, cp=2) | bf16 | **2.92s** | **4.63s** | **8.91s** |
| p5.48xlarge | 1x H100 80GB | bf16 | **2.92s** | **4.67s** | **9.21s** |
| trn1.32xlarge | 2x Trainium1 (tp=4, cp=2) | bf16 | 4.69s | 7.53s | 14.66s |
| g6.4xlarge | 1x L4 24GB | NF4 no offload | — | 50.45s | — |
| g6.4xlarge | 1x L4 24GB | FP8 + model_cpu_offload | 56.01s | 85.30s | 158.05s |

**Key findings:**

- Trainium2 matches H100 performance (4.63s vs 4.67s at 25 steps), within measurement variance
- Trainium1 is ~1.6x slower than H100 but at lower cost
- L4 requires quantization (NF4/FP8) due to 24GB VRAM constraint; best case 50.45s (NF4)

### 4.2 L4 Quantization & Offload Strategies (25 steps)

The L4 has only 24GB VRAM. FLUX.1-dev transformer alone is 23.8GB in bf16, so quantization and/or CPU offloading is required.

| Method | 25 steps | Notes |
|--------|----------|-------|
| NF4 (4-bit) no offload | **50.45s** | Fastest on L4. ~6GB transformer fits in VRAM with T5 |
| NF4 + model_cpu_offload | 58.01s | Offload overhead adds ~8s |
| FP8 (torchao) + model_cpu_offload | 85.30s | FP8 transformer ~12GB, must offload |
| INT8 + sequential_cpu_offload | 112.94s | Layer-by-layer offload, very slow |

### 4.3 Cost Efficiency (25 steps)

| Instance | Accelerator | $/hr | Latency | Images/hr | Images/$ |
|----------|------------|------|---------|-----------|----------|
| trn1.32xlarge | 2x Trainium1 | $2.69 | 7.53s | 478 | **177.8** |
| p5.48xlarge | 1x H100 | $4.33 | 4.67s | 771 | **178.1** |
| trn2.48xlarge | 2x Trainium2 | $4.47 | 4.63s | 778 | **174.0** |
| g6.4xlarge | 1x L4 (NF4) | $1.32 | 50.45s | 71 | **54.1** |
| g6.4xlarge | 1x L4 (FP8) | $1.32 | 85.30s | 42 | **32.0** |

Trainium1, H100, and Trainium2 all deliver ~175 images per dollar — nearly identical cost efficiency despite different latencies. L4 is ~3x less cost-efficient.

## 5. SDXL Results

### 5.1 H100 vs Trainium2 (25 steps, 1024x1024)

| Model | H100 (tp=1) | Trn2 (tp=1) | Trn2 / H100 |
|-------|------------|------------|--------------|
| stable-diffusion-xl-base-1.0 | 2.27s | 5.74s | 2.53x slower |

Note: Neuron SDXL currently only supports tp=1. Higher tensor parallelism requires significant modifications not yet available.

### 5.2 H100 Multi-Step (p5.48xlarge, tp=1)

| Model | 15 steps | 25 steps | 50 steps |
|-------|----------|----------|----------|
| SDXL | 1.30s | 2.04s | 3.94s |
| FLUX.1-dev | 2.91s | 4.76s | 9.39s |

## 6. FLUX.2-dev Results

| Instance | Config | 25 steps |
|----------|--------|----------|
| p5.48xlarge (H100) | tp=1 (w/o text_encoder) | 14.41s |
| trn2.48xlarge | tp=8 | Not supported |

FLUX.2-dev is 32B parameters and cannot run directly on a single H100. The text_encoder is run separately and prompt_embeds are passed to the inference pipeline. Trainium2 does not have official FLUX.2-dev support at this time.

## 7. L4 vs RTX 4090 Analysis

Reference: [JarvisLabs - Best GPU for FLUX](https://jarvislabs.ai/ai-faqs/best-gpu-for-flux)

Despite having the same 24GB VRAM, the L4 is 2-3x slower than the RTX 4090 for diffusion models:

| Spec | NVIDIA L4 | RTX 4090 |
|------|-----------|----------|
| Memory Bandwidth | 300 GB/s | 1,008 GB/s |
| FP16 TFLOPS | 121 | 82.6 |
| TDP | 72W | 450W |

The L4 has higher theoretical TFLOPS, but diffusion inference is **memory-bandwidth bound**. The 4090's 3.4x higher bandwidth dominates in practice.

## 8. Pending Test Items

The following items are requested but not yet completed:

| Category | Model/Test | Status |
|----------|-----------|--------|
| Multimodal | Qwen2.5-VL-2B | Not started |
| Diffusion | S3Diff | Not started |
| Traditional CNN | YOLO | Not started |
| Traditional CNN | Real-ESRGAN | Not started |
| Resolution | 2K / 4K testing | Not started |
| Precision | FP8 on Trainium2 | Not started |
| Metrics | Peak VRAM recording | Not started |
| Metrics | Cold start / warm start split | Not started |
| Metrics | Generated image samples | Partially done |
| GPU virtualization | 1/2 or 1/4 VRAM split | Not started |

## 9. Appendix: Reproduction

### GPU (H100)

```bash
pip install torch diffusers transformers accelerate safetensors sentencepiece protobuf
huggingface-cli login
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('black-forest-labs/FLUX.1-dev', local_dir='/home/ubuntu/models/FLUX.1-dev/')"
python benchmark_unified_gpu.py
```

### GPU (L4 - quantized)

```bash
pip install torch diffusers transformers accelerate safetensors sentencepiece protobuf
pip install bitsandbytes  # for NF4
pip install torchao       # for FP8
python benchmark_l4_fp8_torchao.py
python benchmark_l4_nf4_offload.py
```

### Neuron (Trn2 / Trn1)

```bash
source ~/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
huggingface-cli login
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('black-forest-labs/FLUX.1-dev', local_dir='/home/ubuntu/models/FLUX.1-dev/')"
python benchmark_unified_neuron.py
```

All benchmark scripts are available at: [github.com/xniwangaws/NeuronStuff/flux-benchmark](https://github.com/xniwangaws/NeuronStuff/tree/main/flux-benchmark)
