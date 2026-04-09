# FLUX.1-dev Inference Benchmark

Benchmark FLUX.1-dev image generation across AWS Trainium and GPU instance types.

## Benchmark Parameters

| Parameter | Value |
|-----------|-------|
| Model | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| Resolution | 1024 x 1024 |
| Guidance scale | 3.5 |
| Max sequence length | 512 |
| Prompt | "A cat holding a sign that says hello world" |
| Seed | 0 (`torch.Generator("cpu").manual_seed(0)`) |
| Warmup rounds | 5 |
| Inference steps | 15, 25, 50 |

## Instance Specifications

| Instance | Accelerator | VRAM | On-Demand $/hr |
|----------|------------|------|----------------|
| trn2.48xlarge | 2x Trainium2 chips (tp=4, cp=2) | 96GB HBM per chip | $35.76 (full) / ~$4.47 (2 chips) |
| p5.48xlarge | 1x H100 80GB | 80GB HBM3 | $34.61 (full) / ~$4.33 (1 GPU) |
| trn1.32xlarge | 2x Trainium1 chips (tp=4, cp=2) | 32GB HBM per chip | $21.50 (full) / ~$2.69 (2 chips) |
| g6.4xlarge | 1x L4 24GB | 24GB GDDR6 | $1.32 |

## Results

### Main Comparison

| Instance | Accelerator | Method | 15 steps | 25 steps | 50 steps |
|----------|------------|--------|----------|----------|----------|
| trn2.48xlarge | 2x Trainium2 (tp=4, cp=2) | bf16 | **2.92s** | **4.63s** | **8.91s** |
| p5.48xlarge | 1x H100 80GB | bf16 | **2.92s** | **4.67s** | **9.21s** |
| trn1.32xlarge | 2x Trainium1 (tp=4, cp=2) | bf16 | 4.69s | 7.53s | 14.66s |
| g6.4xlarge | 1x L4 24GB | NF4 no offload | — | 50.45s | — |
| g6.4xlarge | 1x L4 24GB | FP8 + model_cpu_offload | 56.01s | 85.30s | 158.05s |

> Trn2 matches H100 performance at a fraction of the cost.

### L4 (g6) - Quantization & Offload Strategies (25 steps)

The L4 has only 24GB VRAM. FLUX.1-dev transformer alone is 23.8GB in bf16, so quantization and/or CPU offloading is required.

| Method | 25 steps | Notes |
|--------|----------|-------|
| NF4 (4-bit) no offload | **50.45s** | Fastest on L4. ~6GB transformer fits in VRAM with T5 |
| NF4 + model_cpu_offload | 58.01s | Offload overhead adds ~8s |
| FP8 (torchao) + model_cpu_offload | 85.30s | FP8 transformer ~12GB, must offload |
| INT8 + sequential_cpu_offload | 112.94s | Layer-by-layer offload, very slow |

## Cost Efficiency (25 steps)

| Instance | Accelerator | $/hr | Latency | Images/hr | Images/$ |
|----------|------------|------|---------|-----------|----------|
| trn1.32xlarge | 2x Trainium1 | $2.69 | 7.53s | 478 | **177.8** |
| p5.48xlarge | 1x H100 | $4.33 | 4.67s | 771 | **178.1** |
| trn2.48xlarge | 2x Trainium2 | $4.47 | 4.63s | 778 | **174.0** |
| g6.4xlarge | 1x L4 (NF4) | $1.32 | 50.45s | 71 | **54.1** |
| g6.4xlarge | 1x L4 (FP8) | $1.32 | 85.30s | 42 | **32.0** |

> At 25 steps, Trn1, H100, and Trn2 all deliver ~175 images per dollar — nearly identical cost efficiency despite very different latencies. L4 is 3x less cost-efficient due to quantization overhead and low memory bandwidth.

## Why is L4 so slow?

Reference: [JarvisLabs - Best GPU for FLUX](https://jarvislabs.ai/ai-faqs/best-gpu-for-flux)

Despite having the same 24GB VRAM as the RTX 4090, the L4 is 2-3x slower for diffusion models:

| Spec | NVIDIA L4 | RTX 4090 |
|------|-----------|----------|
| Memory Bandwidth | **300 GB/s** | **1,008 GB/s** |
| FP16 TFLOPS | 121 | 82.6 |
| TDP | 72W | 450W |

The L4 actually has higher theoretical TFLOPS, but diffusion model inference is **memory-bandwidth bound**. The 4090's 3.4x higher bandwidth makes it 2-3x faster in practice.

Additionally, on L4 the model doesn't fit in VRAM at full precision, requiring either:
- **Quantization** (NF4/FP8/INT8) to shrink the model
- **CPU offloading** to swap components in/out of GPU memory
- Both add overhead beyond the raw compute time

## Scripts

| Script | Description |
|--------|-------------|
| `benchmark_unified_gpu.py` | bf16 benchmark for GPU instances (H100, L40S). Single GPU, steps 15/25/50 |
| `benchmark_unified_neuron.py` | bf16 benchmark for Neuron instances (Trn2, Trn1). NxDI with world_size=8, backbone_tp=4 |
| `benchmark_l4_nf4_offload.py` | L4 quantization/offload comparison: NF4/INT8 x various offload strategies |
| `benchmark_l4_fp8_torchao.py` | L4 FP8 via torchao Float8WeightOnlyConfig + model_cpu_offload |

## Software Versions

| Component | GPU Instances | Neuron Instances |
|-----------|--------------|-----------------|
| PyTorch | 2.6.0+cu126 | 2.5.1 (torch-neuronx) |
| diffusers | 0.37.1 | N/A (uses NxDI) |
| torchao | 0.17.0 | N/A |
| bitsandbytes | 0.45.5 | N/A |
| NxDI | N/A | neuronx-distributed-inference |
| CUDA | 12.6 | N/A |

## How to Reproduce

### GPU (H100)

```bash
pip install torch diffusers transformers accelerate safetensors sentencepiece protobuf
huggingface-cli login
python -c "from huggingface_hub import snapshot_download; snapshot_download('black-forest-labs/FLUX.1-dev', local_dir='/home/ubuntu/models/FLUX.1-dev/')"
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
# Use Neuron DLAMI with pre-installed venv
source ~/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
huggingface-cli login
python -c "from huggingface_hub import snapshot_download; snapshot_download('black-forest-labs/FLUX.1-dev', local_dir='/home/ubuntu/models/FLUX.1-dev/')"
python benchmark_unified_neuron.py
```
