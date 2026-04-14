---
title: "AWS CV Model Inference Benchmark Report"
subtitle: ""
date: "2026-04-14"
---

## 1. Overview

This report presents inference performance benchmarks for computer vision models on AWS accelerators, including Trainium2, Trainium1, NVIDIA H100, and NVIDIA L4. The primary focus is on diffusion-based image generation models (FLUX.1-dev, SDXL, FLUX.2-dev), with benchmarks covering end-to-end latency, cost efficiency, and quantization strategies.

## 2. Test Environment

### 2.1 Full Instance Specifications

**Trainium Instances**

| Spec | trn2.48xlarge | trn1.32xlarge |
|------|--------------|---------------|
| Trainium Chips | 16 | 16 |
| Accelerator Memory (total) | 1.5 TB (96 GB/chip) | 512 GB (32 GB/chip) |
| vCPUs | 192 | 128 |
| Instance Memory | 2 TB | 512 GiB |
| Local NVMe Storage | 4 x 1.92 TB | 8 TB |
| Network Bandwidth | 3.2 Tbps | 800 Gbps |
| EFA/RDMA | Yes | Yes |
| EBS Bandwidth | 80 Gbps | 80 Gbps |
| On-Demand $/hr | $35.76 | $21.50 |

**GPU Instances**

| Spec | p5.48xlarge | g6.4xlarge |
|------|-----------|------------|
| GPUs | 8x H100 80GB (HBM3) | 1x L4 24GB (GDDR6) |
| GPU Memory (total) | 640 GB | 24 GB |
| vCPUs | 192 | 16 |
| Instance Memory | 2 TB | 64 GiB |
| Network Bandwidth | 3.2 Tbps | Up to 25 Gbps |
| On-Demand $/hr | $34.61 | $1.32 |

### 2.2 Benchmark Resource Usage

This benchmark does **not** use the full instance. Details below:

**trn2.48xlarge** — Used 2 of 16 chips, 192 GB of 1.5 TB, pro-rated ~$4.47/hr (2/16)

**trn1.32xlarge** — Used 2 of 16 chips, 64 GB of 512 GB, pro-rated ~$2.69/hr (2/16)

**p5.48xlarge** — Used 1 of 8 H100 GPUs, 80 GB of 640 GB, pro-rated ~$4.33/hr (1/8)

**g6.4xlarge** — Used 1 of 1 L4 GPU, 24 GB, full price $1.32/hr

Trainium benchmark uses NxDI with `world_size=8` (2 chips, each chip exposes 4 NeuronCores), `backbone_tp_degree=4`. In production, multiple model replicas can run on the remaining chips to maximize utilization.

### 2.3 Software Stack

| Component | GPU Instances | Neuron Instances |
|-----------|--------------|-----------------|
| OS | Ubuntu 22.04 (AWS DLAMI) | Ubuntu 24.04 (Neuron DLAMI) |
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

### 4.2 Cost Efficiency (25 steps)

| Instance | Accelerator | Cost | Latency | Img/hr | Img/Dollar | vs H100 |
|----------|------------|------|---------|--------|------------|---------|
| p5.48xlarge | 1x H100 | 4.33 | 4.67s | 771 | **178.1** | 100% |
| trn1.32xlarge | 2x Trainium1 | 2.69 | 7.53s | 478 | **177.8** | 100% |
| trn2.48xlarge | 2x Trainium2 | 4.47 | 4.63s | 778 | **174.0** | 98% |
| g6.4xlarge | 1x L4 (NF4) | 1.32 | 50.45s | 71 | 54.1 | 30% |
| g6.4xlarge | 1x L4 (FP8) | 1.32 | 85.30s | 42 | 32.0 | 18% |

Trainium1, H100, and Trainium2 all deliver ~175 images per dollar (within 2% of each other) — nearly identical cost efficiency despite very different latencies. L4 is 3-5x less cost-efficient.

## 5. SDXL Results

### 5.1 H100 vs Trainium2 (25 steps, 1024x1024)

| Model | H100 (tp=1) | Trn2 (tp=1) | Trn2 / H100 |
|-------|------------|------------|--------------|
| stable-diffusion-xl-base-1.0 | 2.27s | 5.74s | 2.53x slower |


### 5.2 H100 Multi-Step (p5.48xlarge, tp=1)

| Model | 15 steps | 25 steps | 50 steps |
|-------|----------|----------|----------|
| SDXL | 1.30s | 2.04s | 3.94s |
| FLUX.1-dev | 2.91s | 4.76s | 9.39s |

## 6. Pending Test Items

### 6.1 Not yet tested (planned)

| Category | Model/Test | Notes |
|----------|-----------|-------|
| Multimodal | Qwen2.5-VL-2B | Need to set up environment |
| Diffusion | S3Diff | Need to set up environment |
| Traditional CNN | YOLO (YOLOv8) | Neuron supported, see [sample-YOLOv8-neuron](https://github.com/aws-samples/sample-YOLOv8-neuron) |
| Diffusion | FLUX.2-dev | 32B model, H100 needs split text_encoder; Trn2 no NxDI support yet |
| Traditional CNN | Real-ESRGAN | Need to set up environment |
| Resolution | 2K / 4K testing | Rerun existing models at higher resolutions |
| Metrics | Peak VRAM recording | Add `torch.cuda.max_memory_allocated()` to scripts |
| Metrics | Cold start / warm start split | Separate load, first inference, and warm inference timing |
| Metrics | Generated image samples | Collect from each platform with same prompt/seed |

### 6.2 Known limitations

| Item | Status |
|------|--------|
| FP8 on Trainium2 (diffusion) | Not supported. NxDI diffusion pipeline currently runs bf16 only |
| FLUX.2-dev on Trainium2 | Not supported. No official NxDI support for FLUX.2-dev |
| SDXL on Trainium2 tp>1 | Not supported. Neuron SDXL only supports tp=1 currently |

### 6.3 Trainium Accelerator Virtualization

Unlike GPU MIG (Multi-Instance GPU), Trainium2 natively supports flexible chip partitioning through NeuronCore allocation. Each Trainium2 chip has 4 NeuronCores, and workloads can request any subset of chips on the instance.

For example, on a trn2.48xlarge (16 chips, 64 NeuronCores):

- **Full instance**: 16 chips (1.5 TB) — e.g., large LLM training
- **1/2 instance**: 8 chips (768 GB) — e.g., medium model inference
- **1/4 instance**: 4 chips (384 GB) — e.g., smaller model inference
- **1/8 instance**: 2 chips (192 GB) — e.g., FLUX.1-dev (this benchmark)
- **1/16 instance**: 1 chip (96 GB) — e.g., SDXL or small models

This is managed via `world_size` in NxDI or through Kubernetes Dynamic Resource Allocation (DRA). DRA enables attribute-based device selection and topology-aware scheduling, allowing multiple workloads to share a single trn2.48xlarge with isolated chip allocations. No special hardware configuration is needed — multiple model replicas can run concurrently on different chip subsets of the same instance.

## 7. Appendix: Reproduction

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
