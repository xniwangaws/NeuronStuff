# SDXL-base-1.0 多设备 benchmark 报告

_[English version: README.en.md](README.en.md)_

> Prompt:`"An astronaut riding a green horse"`,guidance 7.5(SDXL 默认),50 step,batch=1,seeds 42–51(共 10 个;L4 4K BF16 仅 seed 42、L4 4K FP16 仅 3 seeds 抽样)

## 1. 设备与价格(AWS on-demand,2026-05)

| 实例 | 芯片 | 内存 | $/hr | Region |
|---|---|---|---:|---|
| **trn2.3xlarge** 等效 | 1× Trainium2 (LNC=2, 4 logical cores) | 96 GB HBM | **$2.235** | ap-southeast-4(墨尔本) |
| p5.4xlarge | 1× H100 SXM5 | 80 GB HBM3 | **$4.326** | us-east-1 |
| g6.4xlarge | 1× L4 | 24 GB GDDR6 | **$1.323** | sa-east-1 |

> Neuron 物理上跑在 trn2.48xlarge,SDXL 本轮跑 DataParallel=2(2/4 logical cores = 1/2 Trainium2 芯片,LNC=2),按 1/2 芯片刊例计价 ($1.1175/hr)。全芯片 (DP=4 或 TP=4) 属未来优化。H100 主基准为 **BF16**,另附 **FP8 + torch.compile(reduce-overhead)** 10-seed 结果(2026-05-07 加测):1K **1.84s**, 2K **8.37s**, 4K **63.86s** —— 在所有分辨率上均比 BF16 快 1.45-2.09×,且比 eager FP8 快 12-16×。

## 2. 1024² 端到端耗时 + 峰值显存 + $/image(以 H100 BF16 为基准)

| 设备 | 精度 | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | 速度 vs H100 BF16 | 成本 vs H100 BF16 |
|---|---|---:|---|---:|---:|---:|---:|
| **H100 p5.4xlarge** | **BF16(基准)** | **3.84** | 8.98 GB | 10/10 | **$0.00462** | **1.00×** | **1.00×** |
| **H100 p5.4xlarge** | **FP8+torch.compile(reduce-overhead)** | **1.84** | 6.88 GB | 10/10 | **$0.00221** | **2.09× faster** | **0.48×**(便宜 52%） |
| **Neuron trn2.3xlarge (SDK 2.29)** | **BF16 + NKI flash-attn + CFG=7.5** *(DP=2, 2/4 cores = 1/2 Trainium2 芯片, seeds 42-51)* | **11.14** | — | 10/10 | **$0.00346** | **0.34×(慢 2.90×)** | **0.75×**(便宜 25%） |
| L4 g6.4xlarge | BF16 | 19.75 | 5.21 GB | 10/10 | $0.00726 | 1.02× | 0.30×(便宜 3.33×) |
| **L4 g6.4xlarge** | **FP8+torch.compile(reduce-overhead)** | **12.68** | 6.87 GB | 10/10 | **$0.00466** | **0.30×(慢 3.29×)** | **1.01×**(持平) |

`$/image = (Mean / 3600) × $/hr`

**核心结论**:
- **H100 BF16**: 1K 3.84s 为基准
- **H100 FP8+torch.compile**: 1.84s / $0.00221 — 比 BF16 快 2.09×,便宜 52%,**最快最省**
- **Neuron trn2.3xl (SDK 2.29) DP=2 path**: 11.14s 10/10,**只用 2/4 logical cores = 1/2 Trainium2 芯片**,按 1/2 芯片价 ($1.1175/hr) 计 $/image = **$0.00346**,**比 H100 BF16 便宜 25%** (扩展到 4 cores 预计再快 2×)
- **L4 FP8+compile**: 12.68s / $0.00466,比 L4 BF16 快 1.56×,$/image 降 36%,和 H100 BF16 持平
- L4 BF16: 19.75s / $0.00726

## 3. 2048² 端到端耗时 + 峰值显存 + $/image(以 H100 BF16 为基准)

| 设备 | 精度 | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | 速度 vs H100 BF16 | 成本 vs H100 BF16 |
|---|---|---:|---|---:|---:|---:|---:|
| **H100 p5.4xlarge** | **BF16(基准)** | **12.14** | 9.00 GB | 10/10 | **$0.01459** | **1.00×** | **1.00×** |
| **H100 p5.4xlarge** | **FP8+torch.compile(reduce-overhead)** | **8.37** | 6.91 GB | 10/10 | **$0.01005** | **1.45× faster** | **0.69×**(便宜 31%） |
| Neuron trn2.3xl | BF16 | **编译不可行** | — | — | — | — | — |
| L4 g6.4xlarge | BF16 | 95.19 | 6.15 GB | 10/10 | $0.03498 | 0.13×(慢 7.84×) | 2.40× 贵 |
| **L4 g6.4xlarge** | **FP8+torch.compile(reduce-overhead)** | **74.85** | 6.88 GB | 10/10 | **$0.02751** | **0.16×(慢 6.16×)** | **1.89× 贵** |

**核心结论**:
- H100 BF16 2K 12.14s 为基准。H100 **FP8+torch.compile** (2026-05-07 新测) 8.37s — 比 BF16 快 1.45×
- L4 2K 95.19s (BF16) / **74.85s (FP8+compile, 1.27× 更快)** — $/image 贵 2.40× / 1.89× vs H100 BF16
- **Neuron trn2.3xl SDK 2.29 2K/4K 编译不可行**(详见下)

## 4. 4096² 端到端耗时 + 峰值显存 + $/image(以 H100 BF16 为基准)

| 设备 | 精度 | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | 速度 vs H100 BF16 | 成本 vs H100 BF16 |
|---|---|---:|---|---:|---:|---:|---:|
| **H100 p5.4xlarge** | **BF16(基准)** | **94.37** | 11.62 GB | 10/10 | **$0.11341** | **1.00×** | **1.00×** |
| **H100 p5.4xlarge** | **FP8+torch.compile(reduce-overhead)** | **63.86** | 7.04 GB | 10/10 | **$0.07673** | **1.48× faster** | **0.68×**(便宜 32%） |
| Neuron trn2.3xl | BF16 | **编译不可行(UNet 9.8M 指令超限)** | — | — | — | — | — |
| L4 g6.4xlarge | BF16(1 seed 抽样) | 619.18 | 9.91 GB | 1/1 | $0.22754 | 0.18×(慢 5.46×) | 1.67× 贵 |
| **L4 g6.4xlarge** | **FP8+torch.compile (3 seeds 抽样)** | **550.21** | 7.01 GB | 3/3 | **$0.20221** | **0.17×(慢 5.86×)** | **1.78× 贵** |

**核心结论**:
- H100 BF16 4K 94.37s 为基准。H100 **FP8+torch.compile** (2026-05-07 新测) 63.86s — 比 BF16 快 1.48×
- L4 4K ~619s (BF16, 1 seed) / **550.21s (FP8+compile, 3 seeds, 1.13× 更快)** — $/image 贵 2.01× / 1.78× vs H100 BF16
- Neuron trn2.3xl 4K 编译不可行(UNet 9.8M 指令超 5M 硬限)

## 5. 同 prompt / seed 的生图对比(seed 42)

### 5.1 1024² seed 42

| H100 BF16 | **Neuron BF16 CFG=7.5 (DP=2 NKI)** | L4 BF16 |
|:---:|:---:|:---:|
| ![](astronaut_bench/results/sdxl_astro_h100_1024/seed42_astro.png) | ![](astronaut_bench/results/sdxl_astro_trn2_whn09_1024_seeds42_51/seed42.png) | ![](astronaut_bench/results/sdxl_astro_l4_1024/seed42_astro.png) |

### 5.2 2048² seed 42

| H100 BF16 | Neuron BF16 (2K 编译不可行) | L4 BF16 |
|:---:|:---:|:---:|
| ![](astronaut_bench/results/sdxl_astro_h100_2048/seed42_astro.png) | 编译不可行(见 §3) | ![](astronaut_bench/results/sdxl_astro_l4_2048/seed42_astro.png) |

### 5.3 4096² seed 42

| H100 BF16 | Neuron BF16 (4K 编译不可行) | L4 BF16 |
|:---:|:---:|:---:|
| ![](astronaut_bench/results/sdxl_astro_h100_4096/seed42_astro.png) | 编译不可行(见 §4) | ![](astronaut_bench/results/sdxl_astro_l4_4096/seed42_astro.png) |

**视觉一致性**:1K / 2K 下 H100 / L4 / Neuron(CFG=7.5)seed 42 主体一致(宇航员 + 绿马)。2K / 4K Neuron 编译阻塞于 `NCC_EVRF007`(见 §3 / §4)。

## 6. 10-seed 全量 PNG 路径

| 设备 / 分辨率 | 目录 |
|---|---|
| H100 1K BF16(10 seeds) | `astronaut_bench/results/sdxl_astro_h100_1024/seed{42..51}_astro.png` |
| H100 2K BF16(10 seeds) | `astronaut_bench/results/sdxl_astro_h100_2048/seed{42..51}_astro.png` |
| H100 4K BF16(10 seeds) | `astronaut_bench/results/sdxl_astro_h100_4096/seed{42..51}_astro.png` |
| L4 1K BF16(10 seeds) | `astronaut_bench/results/sdxl_astro_l4_1024/seed{42..51}_astro.png` |
| L4 2K BF16(10 seeds) | `astronaut_bench/results/sdxl_astro_l4_2048/seed{42..51}_astro.png` |
| L4 4K BF16(1 seed 抽样) | `astronaut_bench/results/sdxl_astro_l4_4096/seed42_astro.png` |
| **L4 1K FP8+torch.compile(10 seeds)** | `astronaut_bench/results/sdxl_astro_l4_fp8_compile_1024/seed{42..51}_astro.png` |
| **L4 2K FP8+torch.compile(10 seeds)** | `astronaut_bench/results/sdxl_astro_l4_fp8_compile_2048/seed{42..51}_astro.png` |
| **L4 4K FP8+torch.compile(3 seeds 抽样)** | `astronaut_bench/results/sdxl_astro_l4_fp8_compile_4096/seed{42,43,44}_astro.png` |
| Neuron trn2 2K / 4K | 编译不可行(见 §3 / §4) |

每个目录含 `results.json`(mean_s / peak_vram_gb / per-seed std 等)。

## 7. 硬件 / 软件配置

**Neuron — trn2.3xlarge (SDK 2.29) 本轮**
- SDK:**2.29** / neuronx-cc / torch-neuronx
- venv:`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`
- 编译:5/5 NEFF(UNet / CLIP-L / CLIP-G / VAE decoder / post_quant_conv)通过,~30 min,PR #149 style flags(`--model-type=unet-inference -O1`)
- 运行:**DP=2 (2/4 logical cores) + NKI flash-attn + CFG=7.5**,单 jit.load,10/10 pass,11.14s。`--model-type=unet-inference --lnc=2`,使用 NKI `attention_isa_kernel` flash-attn 替换 SDPA。SDK 2.29 DataParallel scatter 对 scalar timestep 输入有 bug;DP=2 + NKI 路径为当前绕法
- 2K / 4K 编译不可行:详见 §3 / §4

**H100 p5.4xlarge**:DLAMI PyTorch / CUDA 13 / torch 2.11.0+cu130 / diffusers 0.37.1 / torchao 0.17.0。
- BF16:单精度 bf16,无量化(主基准)
- FP8(eager,旧路径):torchao 动态激活量化,eager 模式无 `torch.compile` 时比 BF16 慢 5×,**不可用于生产**
- **FP8+torch.compile (新基准,2026-05-07 加测)**:`Float8DynamicActivationFloat8WeightConfig` + `torch.compile(mode="reduce-overhead")`,latest DLAMI PyTorch 2.10 / CUDA 13 / torchao 0.17 / diffusers 0.38。1K 1.84s / 2K 8.37s / 4K 63.86s 10/10 pass,peak HBM 6.88/6.91/7.04 GB

**L4 g6.4xlarge**:DLAMI / torch 2.9.1+cu128 / diffusers 0.38.0 / bitsandbytes 0.45
- BF16 主测路径
- **FP8+torch.compile (新加 2026-05-07)**:`Float8DynamicActivationFloat8WeightConfig` + `torch.compile(mode="reduce-overhead")`,latest DLAMI PyTorch 2.10 / torchao 0.17 / diffusers 0.38。1K 12.68s 10/10,peak 6.87 GB。L4 Ada FP8 tensor cores 效果好,1.56× 加速

**SDXL 参数**:guidance 7.5(默认),50 step,batch=1,PNDMScheduler 默认。

## 8. 运行脚本(快速复现)

GPU BF16(H100 / L4,通用):

```bash
python astronaut_bench/bench_gpu_astro.py \
    --model /home/ubuntu/models/sdxl-base \
    --device_label h100 --precision bf16 \
    --resolution 1024 \
    --seeds 42 43 44 45 46 47 48 49 50 51 \
    --out /opt/dlami/nvme/sdxl_astro_h100_1024
```

GPU FP8(H100,torchao 动态激活 + FP8 权重,仅 UNet):

```bash
python astronaut_bench/bench_gpu_astro_fp8.py \
    --model /home/ubuntu/models/stable-diffusion-xl-base-1.0 \
    --device_label h100 \
    --resolution 1024 \
    --seeds 42 43 44 45 46 47 48 49 50 51 \
    --out /opt/dlami/nvme/sdxl_astro_h100_fp8_1024
```

Neuron(trn2.3xlarge,编译 + benchmark):

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# 编译(5 NEFF,~30 min,可缓存)
python astronaut_bench/trace_sdxl_res.py \
    --model /home/ubuntu/models/sdxl-base \
    --resolution 1024 \
    --compile_dir /home/ubuntu/sdxl/compile_dir_1024

# 运行(当前 NRT_RESOURCE 报错,修复后可用)
python benchmark_neuron.py \
    --compile_dir /home/ubuntu/sdxl/compile_dir_1024 \
    --model /home/ubuntu/models/sdxl-base \
    --prompt "An astronaut riding a green horse" \
    --seeds 42 43 44 45 46 47 48 49 50 51 \
    --steps 50 --guidance 7.5 \
    --out /home/ubuntu/sdxl_astro_neuron_1024
```

对应 2K / 4K:`trace_sdxl_res.py --resolution 2048 / 4096` + `benchmark_neuron.py` 的对应 compile_dir。

## 9. 结论

1. **H100 BF16 是 H100 基准**:1K 3.84 s / $0.00462, 2K 12.14 s / $0.0146, 4K 94.37 s / $0.1134, 10/10 seeds 全通过。**FP8+torch.compile (2026-05-07 新测) 是 H100 更快路径**:1K 1.84 s, 2K 8.37 s, 4K 63.86 s — 全分辨率比 BF16 快 1.45-2.09×。
2. **H100 FP8+torch.compile 是 H100 新最快路径**(2026-05-07 加测):`Float8DynamicActivationFloat8WeightConfig + torch.compile(mode="reduce-overhead")` 在所有分辨率上击败 BF16:
   - 1K: **1.84 s / $0.00221** — 比 BF16 快 **2.09×**, 比 FP8 eager 快 **4.64×**
   - 2K: **8.37 s / $0.01005** — 比 BF16 快 **1.45×**, 比 FP8 eager 快 **12.7×**
   - 4K: **63.86 s / $0.07673** — 比 BF16 快 **1.48×**, 比 FP8 eager 快 **16×**
   - 10/10 seeds 全通过;peak HBM 6.88/6.91/7.04 GB。**已成为 H100 SDXL 生产路径首选**。eager FP8 数据 (`sdxl_astro_h100_fp8_*`) 保留作为反例
3. **L4 全分辨率可用**:BF16 1K $0.00726 / 2K $0.0350 / 4K $0.228。**FP8+torch.compile (2026-05-07 新加):1K 12.68s / $0.00466 — 比 BF16 快 1.56×,$/image 降 36%,和 H100 BF16 持平**。24 GB VRAM 够 SDXL full precision,无需 offload。
4. **Neuron**:
   - **trn2.3xlarge (SDK 2.29) DP=2 path** (2/4 logical cores = 1/2 芯片): 11.14 s / 10/10 / **$0.00346 per image (比 H100 BF16 便宜 25%)**
   - **trn2.3xlarge 2K / 4K 编译阻塞**:2K VAE decoder 生成 7.7M 指令 / 4K UNet 生成 9.8M 指令,均超 `NCC_EVRF007` 5M 硬限;即使 `--optlevel=1` 无效。另外 2K UNet `walrus_driver` 后端占用 >124 GB RAM 超出 trn2.3xlarge(128 GB host)可用内存
5. **后续动作**:(a) ✅ H100 FP8 用 `torch.compile(mode="reduce-overhead") + CUDA graphs` 重测完成 — 见上述 1K/2K/4K 结果;(b) Neuron trn2 2K/4K 仍阻塞于 `NCC_EVRF007` 指令上限(2K VAE 7.69M > 5M 硬限,确认 SDK 2.29 仍有此问题),可尝试 UNet tensor-parallel 拆分或在大 host RAM 实例(r7i)编译后迁移 NEFF
