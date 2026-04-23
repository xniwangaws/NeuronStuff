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

### L4 (g6) - Skip-T5 (ComfyUI-style) Results (25 steps)

Mimics ComfyUI's "skip text encoder" pattern: replace T5-XXL output with zeros, keep CLIP-L.
T5-XXL is 9.4GB bf16 and runs once per generation; skipping it trades prompt quality for memory+latency.

| Config | Latency | T5 overhead | VRAM | Notes |
|--------|---------|-------------|------|-------|
| NF4 full (T5+CLIP, offload) | 59.06s | — | offload | Matches prior `benchmark_l4_nf4_offload.py` NF4 model_offload result |
| NF4 skip-T5 (no offload) | **49.96s** | **−9.10s (−15.4%)** | ~7GB | Fits entirely on L4 after dropping T5 |
| FP8 full (T5+CLIP, offload) | 65.93s | — | offload | Kijai/flux-fp8 single-file + `enable_layerwise_casting` (ComfyUI default: FP8 storage / bf16 compute) |
| FP8 skip-T5 (no offload) | **50.25s** | **−15.68s (−23.8%)** | ~15GB | Fits on L4; no offload needed |

> **T5 encoder overhead is real**: 9-16s per generation depending on precision. On memory-constrained L4 where T5 forces `model_cpu_offload`, skipping T5 also eliminates offload swap — net savings 15-24%.

### L4 Memory Math

| Component | Size (bf16) | Size (NF4) | Size (FP8) |
|-----------|-------------|------------|------------|
| CLIP-L (text_encoder) | 246 MB | — | — |
| T5-XXL (text_encoder_2) | 9.4 GB | — | — |
| DiT (transformer) | 23.8 GB | ~6 GB | ~12 GB |
| VAE | 160 MB | — | — |
| **Skip-T5 total (on-GPU)** | 24.2 GB (OOM) | **~6.4 GB** | **~12.4 GB** |

## Scripts

| Script | Description |
|--------|-------------|
| `benchmark_unified_gpu.py` | bf16 benchmark for GPU instances (H100, L40S). Single GPU, steps 15/25/50 |
| `benchmark_unified_neuron.py` | bf16 benchmark for Neuron instances (Trn2, Trn1). NxDI with world_size=8, backbone_tp=4 |
| `benchmark_l4_nf4_offload.py` | L4 quantization/offload comparison: NF4/INT8 x various offload strategies |
| `benchmark_l4_fp8_torchao.py` | L4 FP8 via torchao Float8WeightOnlyConfig + model_cpu_offload |
| `benchmark_comfyui_style_nf4.py` | L4 ComfyUI-style component loading, NF4 DiT, full vs skip-T5 modes |
| `benchmark_comfyui_style_fp8.py` | L4 ComfyUI-style, Kijai pre-quantized FP8 single-file + `enable_layerwise_casting`, full vs skip-T5 |

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
pip install torchao       # for FP8 via torchao runtime quantize

huggingface-cli login
python -c "from huggingface_hub import snapshot_download; snapshot_download('black-forest-labs/FLUX.1-dev', local_dir='/home/ubuntu/models/FLUX.1-dev/')"

# Original offload / quantize sweeps
python benchmark_l4_fp8_torchao.py
python benchmark_l4_nf4_offload.py

# ComfyUI-style (component loading + skip-T5 comparison)
python benchmark_comfyui_style_nf4.py --mode both

# FP8 comfy-style uses Kijai pre-quantized single-file (E4M3FN storage, bf16 compute)
wget https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors
python benchmark_comfyui_style_fp8.py --mode both
```

### Neuron (Trn2 / Trn1)

```bash
# Use Neuron DLAMI with pre-installed venv
source ~/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
huggingface-cli login
python -c "from huggingface_hub import snapshot_download; snapshot_download('black-forest-labs/FLUX.1-dev', local_dir='/home/ubuntu/models/FLUX.1-dev/')"
python benchmark_unified_neuron.py
```
