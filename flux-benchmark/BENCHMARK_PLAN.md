# FLUX.1-dev Benchmark Plan — Skip T5 & ComfyUI-Style Loading

## Context

Junyi Liu asked to align NVIDIA GPU benchmark results and investigate whether "skip text" (skipping T5-XXL encoder) affects latency. We need ComfyUI-style component-by-component loading benchmarks with full vs skip-T5 comparisons.

## EC2 Instance

- **Type**: g6.2xlarge (us-east-2)
- **Instance ID**: `i-0df296aa9230e6aae`
- **Public IP**: `52.15.255.177`
- **Specs**: 1x L4 24GB, 8 vCPU, 32GB RAM, $0.98/hr
- **SSH**: `ssh -i ~/.ssh/neuron-bench-us-east-2.pem ubuntu@52.15.255.177`
- **Model path**: `/home/ubuntu/models/FLUX.1-dev/` (download may still be in progress)
- **HF login**: done (token `REDACTED_HF_TOKEN`)
- **Deps installed**: torch, diffusers, transformers, accelerate, safetensors, sentencepiece, protobuf, bitsandbytes
- **Missing dep**: `torchao` (needed for FP8 scripts)
- **Scripts uploaded**: `benchmark_comfyui_style.py`, `benchmark_comfyui_style_nf4.py` already on instance at `/home/ubuntu/`

## Component Sizes (FLUX.1-dev)

| Component | Params | bf16 | NF4 | FP8 |
|-----------|--------|------|-----|-----|
| CLIP-L (text_encoder) | 123M | 246MB | — | — |
| T5-XXL (text_encoder_2) | 4.7B | 9.4GB | — | 4.7GB |
| DiT (transformer) | 12B | 23.8GB | 6GB | 12GB |
| VAE | 80M | 160MB | — | — |

## Scripts — Full Inventory

### Already in repo (diffusers FluxPipeline, full end-to-end)

| # | Script | Target | Offload? |
|---|--------|--------|----------|
| 1 | `benchmark_unified_gpu.py` | H100/L40S bf16 | No |
| 2 | `benchmark_unified_neuron.py` | Trn2/Trn1 via NxDI | No |
| 3 | `benchmark_l4_nf4_offload.py` | L4: NF4 + 6 offload strategies | Various |
| 4 | `benchmark_l4_fp8_torchao.py` | L4: FP8 torchao + offload | Yes |

### New this session (component loading, full vs skip-T5)

| # | Script | What it tests | Offload? | Status |
|---|--------|---------------|----------|--------|
| 5 | `benchmark_comfyui_style.py` | bf16 component loading. `--mode full/skip_t5/both` | Yes (DiT=23.8GB) | Done, on instance |
| 6 | `benchmark_comfyui_style_nf4.py` | NF4 DiT component loading. `--mode full/skip_t5/both` | full=Yes, skip=No (~6.4GB) | Done, on instance |
| 7 | `benchmark_comfyui_style_fp8.py` | FP8 DiT component loading. `--mode full/skip_t5/both` | full=Yes, skip=No (~12.4GB) | **NEEDS TO BE WRITTEN** |

### Script #7 spec

Same structure as #6 (`benchmark_comfyui_style_nf4.py`) but:
- Use `torchao.quantization.quantize_(transformer, Float8WeightOnlyConfig())` for FP8 quantization
- skip_t5 mode: FP8 DiT (~12GB) + CLIP-L (246MB) + VAE (160MB) = ~12.4GB -> `pipe.to("cuda:0")`, no offload needed
- full mode: add T5-XXL bf16 (9.4GB) = ~21.8GB total, tight on L4, use `model_cpu_offload`

## Execution Steps

1. **Check model download** — `ls /home/ubuntu/models/FLUX.1-dev/` should have transformer/, text_encoder/, text_encoder_2/, vae/, etc.
   - If not done: `hf download black-forest-labs/FLUX.1-dev --local-dir /home/ubuntu/models/FLUX.1-dev/`
2. **Install torchao** — `pip install torchao` (needed for FP8 scripts #4 and #7)
3. **Write script #7** — `benchmark_comfyui_style_fp8.py`, upload to instance
4. **Run benchmarks** (in this order, fastest first):
   ```bash
   # NF4 full vs skip (skip is fastest, ~6.4GB on GPU)
   python benchmark_comfyui_style_nf4.py --mode both
   
   # FP8 full vs skip (skip ~12.4GB on GPU, L4 has FP8 tensor cores)
   python benchmark_comfyui_style_fp8.py --mode both
   
   # bf16 full vs skip (both need offload, slowest)
   python benchmark_comfyui_style.py --mode both
   ```
5. **Collect results** — each script prints a summary table + T5 overhead calculation
6. **Update README.md** — add skip-T5 results section
7. **Commit and push** to `xniwangaws/NeuronStuff`
8. **Terminate instance** — `aws ec2 terminate-instances --instance-ids i-0df296aa9230e6aae --region us-east-2`

## Expected Results Table

| Config | 15 steps | 25 steps | 50 steps | Offload? |
|--------|----------|----------|----------|----------|
| NF4 skip-T5 (no offload) | ? | ? | ? | No |
| NF4 full (T5+CLIP, offload) | ? | ? | ? | Yes |
| FP8 skip-T5 (no offload) | ? | ? | ? | No |
| FP8 full (T5+CLIP, offload) | ? | ? | ? | Yes |
| bf16 skip-T5 (offload) | ? | ? | ? | Yes |
| bf16 full (T5+CLIP, offload) | ? | ? | ? | Yes |

Key metric: **T5 overhead = full_time - skip_time** for each config.
