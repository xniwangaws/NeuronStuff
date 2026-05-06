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
| Neuron **trn2.48xlarge (SDK 2.27)** | BF16 **tp=1** *(reference, single Trainium2)* | **5.74** | — | — | **$0.00356** | **1.49× faster** | **0.77× (23% cheaper)** |
| **Neuron trn2.3xlarge (SDK 2.29)** | **BF16 + NKI flash-attn + CFG=7.5** *(DP=2, 2/4 cores = 1/2 Trainium2 chip, seeds 42-51)* | **11.14** | — | 10/10 | **$0.00346** | **0.34× (2.90× slower)** | **0.75× (25% cheaper)** |
| **Neuron trn2.3xlarge (SDK 2.29, batch=2 BF16 CFG=7.5)** | **BF16 batch=2 + NKI flash-attn + CFG=7.5** | **13.262** | — | **10/10** | **$0.00823** | **0.29× (3.45× slower)** | **1.78× more expensive** |
| L4 g6.4xlarge | BF16 | 19.75 | 5.21 GB | 10/10 | $0.00726 | 1.02× | 0.30× (3.33× cheaper) |

`$/image = (Mean / 3600) × $/hr`

**Key takeaways:**
- **H100 BF16** is the baseline at 3.84 s / 1K.
- **Neuron trn2.3xl (SDK 2.27) tp=1 reference**: 5.74 s / $0.00356 — **23% cheaper than H100 BF16** (AWS official, no SDK 2.29 regression).
- **Neuron trn2.3xl (SDK 2.29) DP=2**: 11.14 s, 10/10 pass. Since only 2/4 logical cores (= 1/2 Trainium2) are used, billing at half the chip price ($1.1175/hr) yields $/image = **$0.00346**, **25% cheaper than H100 BF16**. Extending to 4 cores is expected to double throughput and close the gap to SDK 2.27 tp=1.
- **Neuron trn2.3xl (SDK 2.29) batch=2 CFG=7.5 workaround**: 13.262 s / $0.00823 (full-chip billing). 1.78× more expensive than H100 BF16 but restores full CFG=7.5 prompt adherence.
- **L4 BF16**: 19.75 s / $0.00726 at 1K. Recommended L4 production path.

## 3. 2048² latency + peak memory + $/image (H100 BF16 baseline)

| Device | Precision | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | Speed vs H100 BF16 | Cost vs H100 BF16 |
|---|---|---:|---|---:|---:|---:|---:|
| **H100 p5.4xlarge** | **BF16 (baseline)** | **12.14** | 9.00 GB | 10/10 | **$0.01459** | **1.00×** | **1.00×** |
| **H100 p5.4xlarge** | **FP8 + torch.compile(reduce-overhead)** | **8.37** | 6.91 GB | 10/10 | **$0.01005** | **1.45× faster** | **0.69× (31% cheaper)** |
| Neuron trn2.3xl | BF16 | **compile blocked** | — | — | — | — | — |
| L4 g6.4xlarge | BF16 | 95.19 | 6.15 GB | 10/10 | $0.03498 | 0.13× (7.84× slower) | 2.40× more expensive |

**Key takeaways:**
- H100 BF16 at 2K is 12.14 s (baseline). **H100 FP8 + torch.compile** (added 2026-05-07) is 8.37 s — **1.45× faster** than BF16.
- L4 2K is 95.19 s, **$/image 2.40× more expensive** than H100 BF16.
- **Neuron trn2.3xl SDK 2.29 2K/4K cannot compile** (see details below).

## 4. 4096² latency + peak memory + $/image (H100 BF16 baseline)

| Device | Precision | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | Speed vs H100 BF16 | Cost vs H100 BF16 |
|---|---|---:|---|---:|---:|---:|---:|
| **H100 p5.4xlarge** | **BF16 (baseline)** | **94.37** | 11.62 GB | 10/10 | **$0.11341** | **1.00×** | **1.00×** |
| **H100 p5.4xlarge** | **FP8 + torch.compile(reduce-overhead)** | **63.86** | 7.04 GB | 10/10 | **$0.07673** | **1.48× faster** | **0.68× (32% cheaper)** |
| Neuron trn2.3xl | BF16 | **compile blocked (UNet 9.8M instr > 5M limit)** | — | — | — | — | — |
| L4 g6.4xlarge | BF16 (1 seed) | 619.18 | 9.91 GB | 1/1 | $0.22754 | 0.18× (5.46× slower) | 1.67× more expensive |

**Key takeaways:**
- H100 BF16 at 4K is 94.37 s (baseline). **H100 FP8 + torch.compile** (added 2026-05-07) is 63.86 s — **1.48× faster** than BF16.
- L4 4K is ~619 s (1-seed sample), $/image 2.01× more expensive.
- Neuron trn2.3xl 4K cannot compile — UNet generates 9.8M instructions, exceeds the 5M `NCC_EVRF007` hard limit.

## 5. Same prompt / seed image comparison (seed 42)

### 5.1 1024² seed 42

| H100 BF16 | **Neuron BF16 CFG=7.5 (batch=2)** | **Neuron BF16 CFG=7.5 (DP=2 NKI)** | L4 BF16 |
|:---:|:---:|:---:|:---:|
| ![](astronaut_bench/results/sdxl_astro_h100_1024/seed42_astro.png) | ![](astronaut_bench/results/sdxl_astro_trn2_1024_cfg/seed42.png) | ![](astronaut_bench/results/sdxl_astro_trn2_whn09_1024_seeds42_51/seed42.png) | ![](astronaut_bench/results/sdxl_astro_l4_1024/seed42_astro.png) |

### 5.2 2048² seed 42

| H100 BF16 | Neuron BF16 (2K compile blocked) | L4 BF16 |
|:---:|:---:|:---:|
| ![](astronaut_bench/results/sdxl_astro_h100_2048/seed42_astro.png) | compile blocked (see §3) | ![](astronaut_bench/results/sdxl_astro_l4_2048/seed42_astro.png) |

### 5.3 4096² seed 42

| H100 BF16 | Neuron BF16 (4K compile blocked) | L4 BF16 |
|:---:|:---:|:---:|
| ![](astronaut_bench/results/sdxl_astro_h100_4096/seed42_astro.png) | compile blocked (see §4) | ![](astronaut_bench/results/sdxl_astro_l4_4096/seed42_astro.png) |

**Visual consistency**: At 1K / 2K, H100 / L4 / Neuron (CFG=7.5) seed 42 all produce the same subject (astronaut + green horse). Neuron 2K / 4K is blocked on the `NCC_EVRF007` compiler ceiling (see §3 / §4).

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
| **Neuron trn2 1K BF16 CFG=7.5 batch=2 (10 seeds)** | `astronaut_bench/results/sdxl_astro_trn2_1024_cfg/seed{42..51}.png` |
| **Neuron trn2 1K BF16 CFG=7.5 DP=2 NKI (10 seeds)** | `astronaut_bench/results/sdxl_astro_trn2_whn09_1024_seeds42_51/seed{42..51}.png` |
| Neuron trn2 2K / 4K | compile blocked (see §3 / §4) |

Each directory includes a `results.json` with `mean_s`, `peak_vram_gb`, per-seed `std`, etc.

## 7. Hardware / software config

**Neuron — trn2.48xlarge (SDK 2.27) reference**
- AWS official benchmark: SDXL-base-1.0 @ 1024², tp=1 (single Trainium2 chip), **5.74 s / image**.
- Under SDK 2.27 the DataParallel + FP32 NEFF combination works as intended; none of the SDK 2.29 `NRT_RESOURCE` / DataParallel-scatter regressions apply.

**Neuron — trn2.3xlarge (SDK 2.29) this round**
- SDK: **2.29** / neuronx-cc / torch-neuronx
- venv: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`
- Compile: all 5 NEFFs (UNet / CLIP-L / CLIP-G / VAE decoder / post_quant_conv) compile in ~30 min with PR #149 style flags (`--model-type=unet-inference -O1`).
- Run (CFG=7.5 batch=2 workaround): **BF16 + batch=2 UNet NEFF + CFG=7.5 + single `jit.load`**, 10/10 pass, 13.262 s. UNet recompile uses `--model-type=unet-inference --auto-cast matmult --auto-cast-type bf16`; sample shape `[2, 4, 128, 128]`; timestep stays 0-dim scalar; `encoder_hidden_states [2, 77, 2048]`; `text_embeds [2, 1280]`; `time_ids [2, 6]`. Text encoders / VAE decoder / post_quant_conv reuse the 1K batch=1 NEFFs (CFG does not affect those components).
- The AWS official notebook combination (FP32 + DataParallel[0,1] + batch=2 CFG) exceeds per-NC HBM on trn2.3xlarge at LNC=2 (`NRT_RESOURCE`); SDK 2.29 `DataParallel` scatter has a bug on scalar timestep input. The two workarounds above are the current bypasses.
- 2K / 4K cannot compile on SDK 2.29: see §3 / §4.

**H100 p5.4xlarge**: DLAMI PyTorch / CUDA 13 / torch 2.10+cu130 / diffusers 0.38 / torchao 0.17.
- BF16: bf16 single precision, no quantization (primary baseline).
- FP8 (eager, legacy path): torchao dynamic-activation quantization; without `torch.compile`, runs 5× slower than BF16, **not production-ready**.
- **FP8 + torch.compile (new baseline, 2026-05-07)**: `Float8DynamicActivationFloat8WeightConfig` + `torch.compile(mode="reduce-overhead")`; latest DLAMI PyTorch 2.10 / CUDA 13 / torchao 0.17 / diffusers 0.38. 1K 1.84 s / 2K 8.37 s / 4K 63.86 s, 10/10 pass, peak HBM 6.88 / 6.91 / 7.04 GB.

**L4 g6.4xlarge**: DLAMI / torch 2.9.1+cu128 / diffusers 0.38 / bitsandbytes 0.45 (NF4 toolchain available but this round tests BF16).

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

## 9. Conclusions

1. **H100 BF16 is the H100 baseline**: 1K 3.84 s / $0.00462, 2K 12.14 s / $0.0146, 4K 94.37 s / $0.1134, 10/10 seeds pass. **FP8 + torch.compile (added 2026-05-07) is the new faster H100 path**: 1K 1.84 s, 2K 8.37 s, 4K 63.86 s — 1.45-2.09× faster than BF16 at every resolution.
2. **H100 FP8 + torch.compile is the new fastest H100 path** (added 2026-05-07): `Float8DynamicActivationFloat8WeightConfig` + `torch.compile(mode="reduce-overhead")` beats BF16 at every resolution:
   - 1K: **1.84 s / $0.00221** — **2.09×** faster than BF16, **4.64×** faster than eager FP8.
   - 2K: **8.37 s / $0.01005** — **1.45×** faster than BF16, **12.7×** faster than eager FP8.
   - 4K: **63.86 s / $0.07673** — **1.48×** faster than BF16, **16×** faster than eager FP8.
   - 10/10 seeds pass at all resolutions; peak HBM 6.88 / 6.91 / 7.04 GB. **Now the recommended H100 SDXL production path.** Eager FP8 artifacts (`sdxl_astro_h100_fp8_*`) are kept as a negative-example archive.
3. **L4 is viable at all resolutions**: BF16 1K $0.00726 / 2K $0.0350 / 4K $0.228. 24 GB VRAM is enough for SDXL at full precision, no offloading required.
4. **Neuron**:
   - **trn2.48xlarge (SDK 2.27) tp=1 reference**: 5.74 s / $0.00356 per image — **23% cheaper than H100 BF16** (AWS official reference).
   - **trn2.3xlarge (SDK 2.29) DP=2 path** (2/4 logical cores = 1/2 chip): 11.14 s / 10/10 / **$0.00346 per image (25% cheaper than H100 BF16)**.
   - **trn2.3xlarge (SDK 2.29) batch=2 CFG=7.5 workaround** (full chip): 13.262 s / 10/10 / $0.00823 per image. 1.51× faster than the older batch=1 workaround and restores full CFG=7.5 prompt adherence (seed 42 green horse returns). Key fixes: (i) recompile UNet at batch=2 (accommodates CFG's automatic uncond/cond duplicate), (ii) force `TextEncoderOutputWrapper` to return `text_embeds=None` for CLIP-L so diffusers 0.38 uses CLIP-G's 1280-d pooled (fixes the `[1,768] expected [1,1280]` regression), (iii) keep a single `torch.jit.load` (sidesteps the SDK 2.29 `DataParallel` scatter bug), (iv) skip FP32 auto-cast, use `--auto-cast matmult --auto-cast-type bf16`.
   - **trn2.3xlarge 2K / 4K compile blocked**: 2K VAE decoder generates 7.7M instructions / 4K UNet generates 9.8M instructions, both exceed the `NCC_EVRF007` 5M hard limit; `--optlevel=1` does not help. In addition, on 2K the UNet `walrus_driver` backend eats >124 GB RAM, exceeding the 128 GB host RAM on trn2.3xlarge.
5. **Next steps**: (a) ✅ H100 FP8 retested with `torch.compile(mode="reduce-overhead") + CUDA graphs` — see 1K/2K/4K results above; (b) Neuron trn2 2K/4K still blocked on `NCC_EVRF007` (2K VAE 7.69M > 5M, confirmed still present on SDK 2.29). Possible follow-ups: UNet tensor-parallel splitting, or compile on a high-host-RAM instance (r7i) and migrate the NEFFs.
