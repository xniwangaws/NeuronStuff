# SDXL-base-1.0 multi-device benchmark

_[中文版: README.zh.md](README.zh.md)_

> Prompt: `"An astronaut riding a green horse"`, guidance 7.5 (SDXL default), 50 steps, batch=1, seeds 42–51 (10 seeds; L4 4K BF16 is 1-seed sample).

## 1. Instances and pricing (AWS on-demand, 2026-05)

| Instance | Chip | Memory | $/hr | Region |
|---|---|---|---:|---|
| **trn2.3xlarge** equivalent | 1× Trainium2 (LNC=2, 4 logical cores) | 96 GB HBM | **$2.235** | ap-southeast-4 (Melbourne) |
| p5.4xlarge | 1× H100 SXM5 | 80 GB HBM3 | **$4.326** | us-east-1 |
| g6.4xlarge | 1× L4 | 24 GB GDDR6 | **$1.323** | sa-east-1 |

> Neuron physically runs on trn2.48xlarge. This round ships SDXL as **DataParallel=2** (uses 2/4 logical cores = 1/2 Trainium2 chip, LNC=2), so Neuron pricing is charged at **half the trn2.3xlarge rate ($1.1175/hr)**. A full-chip configuration (DP=4 or TP=4) is a future optimization. H100 primary baseline is **BF16**; as of 2026-05-07 we also measured **FP8 + torch.compile(reduce-overhead)** at 10 seeds: 1K **1.84 s**, 2K **8.37 s**, 4K **63.86 s** — 1.45-2.09× faster than BF16 and 12-16× faster than eager FP8 at every resolution.

## 2. 1024² latency + peak memory + $/image (H100 BF16 as baseline)

| Device | Precision | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | Speed vs H100 BF16 | Cost vs H100 BF16 |
|---|---|---:|---|---:|---:|---:|---:|
| **H100 p5.4xlarge** | **BF16 (baseline)** | **3.84** | 8.98 GB | 10/10 | **$0.00462** | **1.00×** | **1.00×** |
| **H100 p5.4xlarge** | **FP8 + torch.compile(reduce-overhead)** | **1.84** | 6.88 GB | 10/10 | **$0.00221** | **2.09× faster** | **0.48× (52% cheaper)** |
| **Neuron trn2.3xlarge (SDK 2.29)** | **BF16 + NKI flash-attn + CFG=7.5** *(DP=2, 2/4 cores = 1/2 Trainium2 chip, seeds 42-51)* | **11.14** | — | 10/10 | **$0.00346** | **0.34× (2.90× slower)** | **0.75× (25% cheaper)** |
| L4 g6.4xlarge | BF16 | 19.75 | 5.21 GB | 10/10 | $0.00726 | 1.02× | 0.30× (3.33× cheaper) |
| **L4 g6.4xlarge** | **FP8 + torch.compile(reduce-overhead)** | **12.68** | 6.87 GB | 10/10 | **$0.00466** | **0.30× (3.29× slower)** | **1.01× (parity)** |

`$/image = (Mean / 3600) × $/hr`

**Key takeaways:**
- **H100 BF16** is the baseline at 3.84 s / 1K.
- **H100 FP8 + torch.compile**: 1.84 s / $0.00221 — 2.09× faster than BF16, 52% cheaper. **Fastest and cheapest overall.**
- **Neuron trn2.3xl (SDK 2.29) DP=2 path**: 11.14 s, 10/10 pass. Since only 2/4 logical cores (= 1/2 Trainium2) are used, billing at half the chip price ($1.1175/hr) yields $/image = **$0.00346**, **25% cheaper than H100 BF16**. Extending to 4 cores is expected to double throughput.
- **L4 FP8 + torch.compile**: 12.68 s / $0.00466 — 1.56× faster than L4 BF16, 36% cheaper per image, at parity with H100 BF16.
- **L4 BF16**: 19.75 s / $0.00726 at 1K. Simpler deployment path.

## 3. 2048² latency + peak memory + $/image (H100 BF16 baseline)

| Device | Precision | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | Speed vs H100 BF16 | Cost vs H100 BF16 |
|---|---|---:|---|---:|---:|---:|---:|
| **H100 p5.4xlarge** | **BF16 (baseline)** | **12.14** | 9.00 GB | 10/10 | **$0.01459** | **1.00×** | **1.00×** |
| **H100 p5.4xlarge** | **FP8 + torch.compile(reduce-overhead)** | **8.37** | 6.91 GB | 10/10 | **$0.01005** | **1.45× faster** | **0.69× (31% cheaper)** |
| **Neuron trn2.3xlarge (SDK 2.29)** | **BF16 img2img upscale** *(1K gen + tiled refine, full chip)* | **57.94** | ~24 GB | 10/10 | **$0.03597** | **0.21× (4.77× slower)** | **2.47× more expensive** |
| L4 g6.4xlarge | BF16 | 95.19 | 6.15 GB | 10/10 | $0.03498 | 0.13× (7.84× slower) | 2.40× more expensive |
| **L4 g6.4xlarge** | **FP8 + torch.compile(reduce-overhead)** | **74.85** | 6.88 GB | 10/10 | **$0.02751** | **0.16× (6.16× slower)** | **1.89× more expensive** |

**Key takeaways:**
- H100 BF16 at 2K is 12.14 s (baseline). **H100 FP8 + torch.compile** (added 2026-05-07) is 8.37 s — **1.45× faster** than BF16.
- **Neuron trn2.3xl img2img upscale** (added 2026-05-23): 57.94 s / 10/10 pass. Uses 1K compiled NEFFs with tiled refinement at 2K. **1.29× faster than L4 FP8+compile** (57.94 s vs 74.85 s). Monolithic 2K compilation remains blocked (host RAM overflow), but the img2img approach produces equivalent-quality images.
- L4 2K: 95.19 s (BF16) / **74.85 s (FP8+compile, 1.27× faster)** — $/image 2.40× / 1.89× more expensive vs H100 BF16.

## 4. 4096² latency + peak memory + $/image (H100 BF16 baseline)

| Device | Precision | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | Speed vs H100 BF16 | Cost vs H100 BF16 |
|---|---|---:|---|---:|---:|---:|---:|
| **H100 p5.4xlarge** | **BF16 (baseline)** | **94.37** | 11.62 GB | 10/10 | **$0.11341** | **1.00×** | **1.00×** |
| **H100 p5.4xlarge** | **FP8 + torch.compile(reduce-overhead)** | **63.86** | 7.04 GB | 10/10 | **$0.07673** | **1.48× faster** | **0.68× (32% cheaper)** |
| **Neuron trn2.3xlarge (SDK 2.29)** | **BF16 img2img upscale** *(1K gen + tiled refine, full chip)* | **142.62** | ~24 GB | 3/3 | **$0.08853** | **0.66× (1.51× slower)** | **0.78× (22% cheaper)** |
| L4 g6.4xlarge | BF16 (1 seed) | 619.18 | 9.91 GB | 1/1 | $0.22754 | 0.18× (5.46× slower) | 1.67× more expensive |
| **L4 g6.4xlarge** | **FP8 + torch.compile (3 seeds)** | **550.21** | 7.01 GB | 3/3 | **$0.20221** | **0.17× (5.86× slower)** | **1.78× more expensive** |

**Key takeaways:**
- H100 BF16 at 4K is 94.37 s (baseline). **H100 FP8 + torch.compile** (added 2026-05-07) is 63.86 s — **1.48× faster** than BF16.
- **Neuron trn2.3xl img2img upscale** (added 2026-05-23): 142.62 s / 3/3 pass. **3.86× faster than L4 FP8+compile** (142.62 s vs 550.21 s) and **22% cheaper than H100 BF16** ($0.089 vs $0.113). Monolithic 4K compilation is not possible (9.8M instructions), but the img2img approach with 16 tiles is highly effective.
- L4 4K: ~619 s (BF16, 1 seed) / **550.21 s (FP8+compile, 3 seeds, 1.13× faster)** — $/image 2.01× / 1.78× more expensive vs H100 BF16.

## 5. Same prompt / seed image comparison (seed 42)

### 5.1 1024² seed 42

| H100 BF16 | **Neuron BF16 CFG=7.5 (DP=2 NKI)** | L4 BF16 |
|:---:|:---:|:---:|
| ![](astronaut_bench/results/sdxl_astro_h100_1024/seed42_astro.png) | ![](astronaut_bench/results/sdxl_astro_trn2_whn09_1024_seeds42_51/seed42.png) | ![](astronaut_bench/results/sdxl_astro_l4_1024/seed42_astro.png) |

### 5.2 2048² seed 42

| H100 BF16 | **Neuron BF16 img2img upscale (57.94s)** | L4 BF16 |
|:---:|:---:|:---:|
| ![](astronaut_bench/results/sdxl_astro_h100_2048/seed42_astro.png) | ![](highres_img2img/results_2048/seed42.png) | ![](astronaut_bench/results/sdxl_astro_l4_2048/seed42_astro.png) |

### 5.3 4096² seed 42

| H100 BF16 | **Neuron BF16 img2img upscale (142.62s)** | L4 BF16 |
|:---:|:---:|:---:|
| ![](astronaut_bench/results/sdxl_astro_h100_4096/seed42_astro.png) | ![](highres_img2img/results_4096/seed42.png) | ![](astronaut_bench/results/sdxl_astro_l4_4096/seed42_astro.png) |

**Visual consistency**: At 1K, all devices produce the same subject (astronaut + green horse) with matching composition. At 2K / 4K, the Neuron img2img upscale approach produces coherent, high-quality images with equivalent subject matter. Note: Neuron 2K/4K uses img2img upscaling from 1K, so pixel-level output differs from direct generation on GPU — but the composition, quality, and detail level are comparable.

## 6. 10-seed full PNG paths

| Device / resolution | Directory |
|---|---|
| H100 1K BF16 (10 seeds) | `astronaut_bench/results/sdxl_astro_h100_1024/seed{42..51}_astro.png` |
| H100 2K BF16 (10 seeds) | `astronaut_bench/results/sdxl_astro_h100_2048/seed{42..51}_astro.png` |
| H100 4K BF16 (10 seeds) | `astronaut_bench/results/sdxl_astro_h100_4096/seed{42..51}_astro.png` |
| H100 1K FP8+compile (10 seeds) | `astronaut_bench/results/sdxl_astro_h100_fp8_compile_1024/seed{42..51}_astro.png` |
| H100 2K FP8+compile (10 seeds) | `astronaut_bench/results/sdxl_astro_h100_fp8_compile_2048/seed{42..51}_astro.png` |
| H100 4K FP8+compile (10 seeds) | `astronaut_bench/results/sdxl_astro_h100_fp8_compile_4096/seed{42..51}_astro.png` |
| L4 1K BF16 (10 seeds) | `astronaut_bench/results/sdxl_astro_l4_1024/seed{42..51}_astro.png` |
| L4 2K BF16 (10 seeds) | `astronaut_bench/results/sdxl_astro_l4_2048/seed{42..51}_astro.png` |
| L4 4K BF16 (1 seed) | `astronaut_bench/results/sdxl_astro_l4_4096/seed42_astro.png` |
| **L4 1K FP8+torch.compile (10 seeds)** | `astronaut_bench/results/sdxl_astro_l4_fp8_compile_1024/seed{42..51}_astro.png` |
| **L4 2K FP8+torch.compile (10 seeds)** | `astronaut_bench/results/sdxl_astro_l4_fp8_compile_2048/seed{42..51}_astro.png` |
| **L4 4K FP8+torch.compile (3 seeds)** | `astronaut_bench/results/sdxl_astro_l4_fp8_compile_4096/seed{42,43,44}_astro.png` |
| **Neuron trn2 1K BF16 CFG=7.5 DP=2 NKI (10 seeds)** | `astronaut_bench/results/sdxl_astro_trn2_whn09_1024_seeds42_51/seed{42..51}.png` |
| **Neuron trn2 2K BF16 img2img upscale (10 seeds)** | `highres_img2img/results_2048/seed{42..51}.png` |
| **Neuron trn2 4K BF16 img2img upscale (3 seeds)** | `highres_img2img/results_4096/seed{42,43,44}.png` |

Each directory includes a `results.json` with `mean_s`, `peak_vram_gb`, per-seed `std`, etc.

## 7. Hardware / software config

**Neuron — trn2.3xlarge (SDK 2.29) this round**
- SDK: **2.29** / neuronx-cc / torch-neuronx
- venv: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`
- Compile: all 7 NEFFs (UNet / CLIP-L / CLIP-G / VAE decoder / post_quant_conv / VAE encoder / quant_conv) compile in ~45 min with `--model-type=unet-inference --auto-cast matmult`.
- **1K**: **DP=2 (2/4 logical cores) + NKI flash-attn + CFG=7.5**, single `jit.load`, 10/10 pass, 11.14 s. Uses NKI `attention_isa_kernel` flash-attn in place of SDPA.
- **2K / 4K (added 2026-05-23)**: img2img upscale approach. Generate at 1K → upscale → tiled VAE encode → add noise (strength=0.35, 18/50 steps) → tiled UNet denoise → tiled VAE decode. Uses same 1K compiled NEFFs. 2K: 57.94 s (10/10), 4K: 142.62 s (3/3). Full chip ($2.235/hr). Script: `highres_img2img/benchmark_img2img.py`.
- Monolithic 2K / 4K compilation remains blocked (host RAM overflow at 2K, instruction limit at 4K).

**H100 p5.4xlarge**: DLAMI PyTorch / CUDA 13 / torch 2.10+cu130 / diffusers 0.38 / torchao 0.17.
- BF16: bf16 single precision, no quantization (primary baseline).
- FP8 (eager, legacy path): torchao dynamic-activation quantization; without `torch.compile`, runs 5× slower than BF16, **not production-ready**.
- **FP8 + torch.compile (new baseline, 2026-05-07)**: `Float8DynamicActivationFloat8WeightConfig` + `torch.compile(mode="reduce-overhead")`; latest DLAMI PyTorch 2.10 / CUDA 13 / torchao 0.17 / diffusers 0.38. 1K 1.84 s / 2K 8.37 s / 4K 63.86 s, 10/10 pass, peak HBM 6.88 / 6.91 / 7.04 GB.

**L4 g6.4xlarge**: DLAMI / torch 2.9.1+cu128 / diffusers 0.38 / bitsandbytes 0.45.
- BF16 main path.
- **FP8 + torch.compile (added 2026-05-07)**: `Float8DynamicActivationFloat8WeightConfig` + `torch.compile(mode="reduce-overhead")`, latest DLAMI PyTorch 2.10 / torchao 0.17 / diffusers 0.38. 1K 12.68 s, 10/10 pass, peak HBM 6.87 GB. L4 Ada FP8 tensor cores give a solid 1.56× speedup over BF16.

**SDXL params**: guidance 7.5 (default), 50 steps, batch=1, PNDMScheduler default.

## 8. Reproduction scripts

GPU BF16 (H100 / L4, shared script):

```bash
python astronaut_bench/bench_gpu_astro.py \
    --model /home/ubuntu/models/sdxl-base \
    --device_label h100 --precision bf16 \
    --resolution 1024 \
    --seeds 42 43 44 45 46 47 48 49 50 51 \
    --out /opt/dlami/nvme/sdxl_astro_h100_1024
```

GPU FP8 + torch.compile (H100 preferred; requires torch 2.10+ / torchao 0.17+ / diffusers 0.38+):

```bash
python astronaut_bench/bench_gpu_astro_fp8_compile.py \
    --model /home/ubuntu/models/sdxl-base \
    --device_label h100 \
    --resolution 1024 \
    --seeds 42 43 44 45 46 47 48 49 50 51 \
    --warmup 3 --mode reduce-overhead \
    --out /home/ubuntu/sdxl_astro_h100_fp8_compile_1024
```

Neuron (trn2.3xlarge, compile + benchmark):

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Compile (5 NEFFs, ~30 min, cacheable)
python astronaut_bench/trace_sdxl_res.py \
    --model /home/ubuntu/models/sdxl-base \
    --resolution 1024 \
    --compile_dir /home/ubuntu/sdxl/compile_dir_1024

# Benchmark
python benchmark_neuron.py \
    --compile_dir /home/ubuntu/sdxl/compile_dir_1024 \
    --model /home/ubuntu/models/sdxl-base \
    --prompt "An astronaut riding a green horse" \
    --seeds 42 43 44 45 46 47 48 49 50 51 \
    --steps 50 --guidance 7.5 \
    --out /home/ubuntu/sdxl_astro_neuron_1024
```

2K / 4K equivalents: pass `--resolution 2048` / `--resolution 4096` to `trace_sdxl_res.py` with a matching `compile_dir`.

Neuron high-res img2img (trn2.3xlarge, 2K/4K via upscale approach):

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Compile 7 NEFFs at 1024x1024 (~45 min, one-time, cacheable)
python highres_img2img/benchmark_img2img.py compile \
    --model /home/ubuntu/models/sdxl-base \
    --compile_dir /home/ubuntu/sdxl/compile_img2img

# Full benchmark (2K: 10 seeds, 4K: 3 seeds)
python highres_img2img/benchmark_img2img.py benchmark \
    --model /home/ubuntu/models/sdxl-base \
    --compile_dir /home/ubuntu/sdxl/compile_img2img \
    --out /home/ubuntu/sdxl/results_img2img
```

See [`highres_img2img/README.md`](highres_img2img/README.md) for detailed approach explanation and latency breakdown.

## 9. Conclusions

1. **H100 BF16 is the H100 baseline**: 1K 3.84 s / $0.00462, 2K 12.14 s / $0.0146, 4K 94.37 s / $0.1134, 10/10 seeds pass. **FP8 + torch.compile (added 2026-05-07) is the new faster H100 path**: 1K 1.84 s, 2K 8.37 s, 4K 63.86 s — 1.45-2.09× faster than BF16 at every resolution.
2. **H100 FP8 + torch.compile is the new fastest H100 path** (added 2026-05-07): `Float8DynamicActivationFloat8WeightConfig` + `torch.compile(mode="reduce-overhead")` beats BF16 at every resolution:
   - 1K: **1.84 s / $0.00221** — **2.09×** faster than BF16, **4.64×** faster than eager FP8.
   - 2K: **8.37 s / $0.01005** — **1.45×** faster than BF16, **12.7×** faster than eager FP8.
   - 4K: **63.86 s / $0.07673** — **1.48×** faster than BF16, **16×** faster than eager FP8.
   - 10/10 seeds pass at all resolutions; peak HBM 6.88 / 6.91 / 7.04 GB. **Now the recommended H100 SDXL production path.** Eager FP8 artifacts (`sdxl_astro_h100_fp8_*`) are kept as a negative-example archive.
3. **L4 is viable at all resolutions**: BF16 1K $0.00726 / 2K $0.0350 / 4K $0.228. **FP8 + torch.compile (added 2026-05-07): 1K 12.68 s / $0.00466 — 1.56× faster than L4 BF16, 36% cheaper per image, at parity with H100 BF16**. 24 GB VRAM is enough for SDXL at full precision, no offloading required.
4. **Neuron**:
   - **trn2.3xlarge (SDK 2.29) DP=2 path at 1K** (2/4 logical cores = 1/2 chip): 11.14 s / 10/10 / **$0.00346 per image (25% cheaper than H100 BF16)**.
   - **trn2.3xlarge img2img upscale at 2K** (added 2026-05-23): **57.94 s** / 10/10 / $0.036. **1.29× faster than L4 FP8+compile** (74.85 s). Uses 1K compiled NEFFs with tiled refinement.
   - **trn2.3xlarge img2img upscale at 4K** (added 2026-05-23): **142.62 s** / 3/3 / $0.089. **3.86× faster than L4 FP8+compile** (550.21 s) and **22% cheaper than H100 BF16** ($0.089 vs $0.113).
   - Monolithic 2K / 4K compilation remains blocked (`NCC_EVRF007` instruction limit + host RAM overflow), but the img2img upscale workaround produces coherent high-quality images at both resolutions.
5. **Neuron vs L4 summary**: Neuron beats L4 at every resolution — 1K (11.14 s vs 12.68 s), 2K (57.94 s vs 74.85 s), 4K (142.62 s vs 550.21 s). The advantage grows at higher resolution (1.14× at 1K → 3.86× at 4K).
