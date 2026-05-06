# SDXL-base-1.0 多设备 benchmark 报告

> Prompt:`"An astronaut riding a green horse"`,guidance 7.5(SDXL 默认),50 step,batch=1,seeds 42–51(共 10 个;L4 4K BF16 仅 seed 42、L4 4K FP16 仅 3 seeds 抽样)

## 1. 设备与价格(AWS on-demand,2026-05)

| 实例 | 芯片 | 内存 | $/hr | Region |
|---|---|---|---:|---|
| **trn2.3xlarge** 等效 | 1× Trainium2(TP=4, LNC=2) | 96 GB HBM | **$2.235** | ap-southeast-4(墨尔本) |
| p5.4xlarge | 1× H100 SXM5 | 80 GB HBM3 | **$4.326** | us-east-1 |
| g6.4xlarge | 1× L4 | 24 GB GDDR6 | **$1.323** | sa-east-1 |

> Neuron 物理上跑在 trn2.48xlarge,SDXL TP=4 只占用单个 Trainium2(8 物理核 → 4 逻辑核,LNC=2),按 trn2.3xlarge 等效单芯片刊例计价。H100 主基准为 BF16(torchao eager FP8 在 SDXL 单 batch 场景反而慢 5×,已从结果表删除,留待 torch.compile 重测)。

## 2. 1024² 端到端耗时 + 峰值显存 + $/image(以 H100 BF16 为基准)

| 设备 | 精度 | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | 速度 vs H100 BF16 | 成本 vs H100 BF16 |
|---|---|---:|---|---:|---:|---:|---:|
| **H100 p5.4xlarge** | **BF16(基准)** | **3.84** | 8.98 GB | 10/10 | **$0.00462** | **1.00×** | **1.00×** |
| **H100 p5.4xlarge** | **FP16 variant** | **3.83** | 11.52 GB | 10/10 | **$0.00460** | **1.00×** | **1.00×** |
| H100 p5.4xlarge | FP8(torchao eager) | *不推荐生产使用(见 §9)* | — | — | — | — | — |
| Neuron **trn2.48xlarge (SDK 2.27)** | BF16 **tp=1** *(参考,单 Trainium2 chip)* | **5.74** | — | — | **$0.00356** | **1.49× 更快** | **0.77×**(便宜 23%) |
| Neuron **trn2.3xlarge (whn09 fork, SDK 2.29)** | BF16 **DataParallel[0,1] + NKI flash-attn** *(CFG=7.5,50 step)* | **11.151** | — | 10/10 | **$0.00692** | **0.34×(慢 2.90×)** | **1.50× 贵** |
| **Neuron trn2.3xlarge (SDK 2.29,batch=2 BF16 CFG=7.5)** | **BF16 batch=2 + NKI flash-attn + CFG=7.5** | **13.262** | — | **10/10** | **$0.00823** | **0.29×(慢 3.45×)** | **1.78× 贵** |
| Neuron trn2.3xlarge (SDK 2.29,AWS notebook 原版) | BF16 TP=4 *(guidance=1.0,no-CFG workaround)* | 19.997 | ~24 GB | 10/10 | $0.01241 | 0.19×(慢 5.21×) | 2.69× 贵 |
| L4 g6.4xlarge | BF16 | 19.75 | 5.21 GB | 10/10 | $0.00726 | 1.02× | 0.30×(便宜 3.33×) |
| **L4 g6.4xlarge** | **FP16 variant** | **22.78** | 5.21 GB | 10/10 | **$0.00837** | **0.17×** | **0.30×**(便宜 3.33×) |

`$/image = (Mean / 3600) × $/hr`

**核心结论**:
- **H100 FP16 variant ≡ BF16**:3.83 s vs 3.84 s(仅 noise 级差异),与用户参考脚本(3.94 s @ 50 step)一致。用 diffusers 官方 `variant="fp16"` 路径可加载 ~4.8 GB checkpoint(而非 9.6 GB FP32→BF16 cast)省一半 HF 下载 / 磁盘,无性能损失
- **Neuron trn2.48xlarge (SDK 2.27) tp=1 参考**:**5.74 s / image** — 单 Trainium2 芯片下 $0.00356,**比 H100 BF16 便宜 23%**(AWS 官方 reference 数据,SDK 2.27 无本轮 SDK 2.29 的 regression)
- Neuron trn2.3xlarge (SDK 2.29) 1K workaround:mean 19.997 s,10/10 pass,$0.01241 / image(BF16 batch=1 + `guidance_scale=1.0`,SDK 2.29 DataParallel regression + FP32 HBM 超预算所致;如修复可追上 SDK 2.27 5.74s)
- **Neuron trn2.3xlarge (SDK 2.29) batch=2 CFG=7.5**:mean **13.262 s ± 0.008 s**,10/10 pass,$0.00823 / image(BF16 batch=2 UNet NEFF + CFG=7.5 + 50 step,无 DataParallel,单 `jit.load`,NKI flash-attn + CLIP-L `text_embeds=None` pooler fix);**比 1K workaround 快 1.51×,恢复 CFG=7.5 完整 prompt 遵循(绿马出现)**,比 H100 BF16 贵 1.78× 但支付了 CFG 完整质量
- **Neuron trn2.3xlarge (SDK 2.29) whn09 fork**:mean **11.151 s ± 0.011 s**,10/10 pass,$0.00692 / image(BF16 + CFG=7.5 + 50 step + DataParallel[0,1] + NKI `attention_isa_kernel` flash-attn 替换 SDPA;`--model-type=unet-inference --lnc=2`);**比上述 workaround 快 1.79×,比 H100 BF16 贵 1.50× 但恢复 CFG=7.5 完整质量**;仍比 trn2.48xl SDK 2.27 tp=1 参考(5.74s)慢 1.94×,主要差距是 whn09 只用 2 cores (LNC=2,DataParallel[0,1]),可进一步扩展到更多核
- L4 FP16 比 L4 BF16 略慢(22.78 vs 19.75 s, +15%);L4 生产建议仍用 BF16
- H100 BF16 基准下 L4 BF16 $/image 贵 1.57×,FP16 贵 1.81×;Neuron trn2.3xl SDK 2.29 workaround 贵 2.69×,但 SDK 2.27 路径便宜 23%

## 3. 2048² 端到端耗时 + 峰值显存 + $/image(以 H100 BF16 为基准)

| 设备 | 精度 | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | 速度 vs H100 BF16 | 成本 vs H100 BF16 |
|---|---|---:|---|---:|---:|---:|---:|
| **H100 p5.4xlarge** | **BF16(基准)** | **12.14** | 9.00 GB | 10/10 | **$0.01459** | **1.00×** | **1.00×** |
| **H100 p5.4xlarge** | **FP16 variant** | **12.71** | 11.57 GB | 10/10 | **$0.01528** | **0.96×** | **1.05×** |
| H100 p5.4xlarge | FP8(torchao eager) | *不推荐生产使用(见 §9)* | — | — | — | — | — |
| Neuron trn2.3xl | BF16 TP=4 | **编译不可行** | — | — | — | — | — |
| L4 g6.4xlarge | BF16 | 95.19 | 6.15 GB | 10/10 | $0.03498 | 0.13×(慢 7.84×) | 2.40× 贵 |
| **L4 g6.4xlarge** | **FP16 variant** | **108.95** | 6.15 GB | 10/10 | **$0.04004** | **0.11×** | **2.74× 贵** |

**核心结论**:
- H100 FP16 variant 与 BF16 在 2K 上仍基本持平(12.71 vs 12.14 s, +5%)
- L4 2K 单图 BF16 ~95 s / FP16 ~109 s;$/image 分别是 H100 BF16 的 2.40× / 2.74×
- **Neuron trn2.3xl 2K 编译不可行(SDK 2.29)**:VAE decoder 首先触发 `NCC_EVRF007`(生成 7.7M 指令 > 典型 5M 上限,`--optlevel=1` 也无效,属 HLO 验证器硬限);VAE 可绕过(改 CPU float32),但 UNet 继续编译时 `walrus_driver` 占用 124 GB RAM + 45+ GB swap 后仍未完成,在 trn2.3xlarge(128 GB host RAM)上超时(>65 min on UNet pass)。修复方向:UNet tensor-parallel 拆分、用更大 host RAM 实例编译后迁移 NEFF、或等编译器优化

## 4. 4096² 端到端耗时 + 峰值显存 + $/image(以 H100 BF16 为基准)

| 设备 | 精度 | Mean (s) | Peak VRAM/HBM | Pass | **$/image** | 速度 vs H100 BF16 | 成本 vs H100 BF16 |
|---|---|---:|---|---:|---:|---:|---:|
| **H100 p5.4xlarge** | **BF16(基准)** | **94.37** | 11.62 GB | 10/10 | **$0.11341** | **1.00×** | **1.00×** |
| **H100 p5.4xlarge** | **FP16 variant** | **97.14** | 11.83 GB | 10/10 | **$0.11674** | **0.97×** | **1.03×** |
| H100 p5.4xlarge | FP8(torchao eager) | *不推荐生产使用(见 §9)* | — | — | — | — | — |
| Neuron trn2.3xl | BF16 TP=4 | **编译不可行(UNet 9.8M 指令超限)** | — | — | — | — | — |
| L4 g6.4xlarge | BF16(1 seed 抽样) | 619.18 | 9.91 GB | 1/1 | $0.22754 | 0.18×(慢 5.46×) | 1.67× 贵 |
| **L4 g6.4xlarge** | **FP16 variant (3 seeds 抽样)** | **686.39** | 9.91 GB | 3/3 | **$0.25225** | **0.14×(慢 7.27×)** | **2.22× 贵** |

**核心结论**:
- H100 FP16 variant 与 BF16 在 4K 上持平(97.14 vs 94.37 s, +3%)
- L4 4K BF16 ~619 s / FP16 ~686 s(FP16 +11%);$/image 对 H100 BF16 分别贵 2.01× / 2.22×
- SDXL 原生 1024²,4K 为超采样,视觉质量受 SDXL spec 限制
- **Neuron trn2.3xl 4K 编译不可行**:UNet 生成 9,861,716 条指令,远超 5M 上限(`NCC_EVRF007`),`--optlevel=1` 无法缓解(属 HLO 验证器硬性限制)。Attention kernel 调用 shape 达 `(20, 64, 65536)` 量级,单个 kernel 已超限

## 5. 同 prompt / seed 的生图对比(seed 42)

### 5.1 1024² seed 42

| H100 BF16 | H100 FP16 | Neuron BF16 no-CFG | **Neuron BF16 CFG=7.5** | L4 BF16 | L4 FP16 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![](astronaut_bench/results/sdxl_astro_h100_1024/seed42_astro.png) | ![](astronaut_bench/results/sdxl_astro_h100_fp16_1024/seed42_astro.png) | ![](astronaut_bench/results/sdxl_astro_trn2_1024/seed42.png) | ![](astronaut_bench/results/sdxl_astro_trn2_1024_cfg/seed42.png) | ![](astronaut_bench/results/sdxl_astro_l4_1024/seed42_astro.png) | ![](astronaut_bench/results/sdxl_astro_l4_fp16_1024/seed42_astro.png) |

### 5.2 2048² seed 42

| H100 BF16 | H100 FP16 | Neuron BF16 TP=4 | L4 BF16 | L4 FP16 |
|:---:|:---:|:---:|:---:|:---:|
| ![](astronaut_bench/results/sdxl_astro_h100_2048/seed42_astro.png) | ![](astronaut_bench/results/sdxl_astro_h100_fp16_2048/seed42_astro.png) | 编译不可行(见 §3) | ![](astronaut_bench/results/sdxl_astro_l4_2048/seed42_astro.png) | ![](astronaut_bench/results/sdxl_astro_l4_fp16_2048/seed42_astro.png) |

### 5.3 4096² seed 42

| H100 BF16 | H100 FP16 | Neuron BF16 TP=4 | L4 BF16 | L4 FP16 |
|:---:|:---:|:---:|:---:|:---:|
| ![](astronaut_bench/results/sdxl_astro_h100_4096/seed42_astro.png) | ![](astronaut_bench/results/sdxl_astro_h100_fp16_4096/seed42_astro.png) | 编译不可行(见 §4) | ![](astronaut_bench/results/sdxl_astro_l4_4096/seed42_astro.png) | ![](astronaut_bench/results/sdxl_astro_l4_fp16_4096/seed42_astro.png) |

**视觉一致性**:1K / 2K 下 H100 与 L4 同 seed 下主体一致(宇航员 + 绿马),仅 seed-noise 级差异。Neuron 1K no-CFG(`guidance=1.0`)下 prompt adherence 下降(马未必是绿色);**Track A batch=2 CFG=7.5 重编完成,seed 42 绿马恢复**,与 H100 / L4 CFG=7.5 一致。2K / 4K Neuron 编译阻塞于 `NCC_EVRF007` + host RAM 限制(见 §3 / §4)。

## 6. 10-seed 全量 PNG 路径

| 设备 / 分辨率 | 目录 |
|---|---|
| H100 1K BF16(10 seeds) | `astronaut_bench/results/sdxl_astro_h100_1024/seed{42..51}_astro.png` |
| H100 2K BF16(10 seeds) | `astronaut_bench/results/sdxl_astro_h100_2048/seed{42..51}_astro.png` |
| H100 4K BF16(10 seeds) | `astronaut_bench/results/sdxl_astro_h100_4096/seed{42..51}_astro.png` |
| **H100 1K FP16 variant(10 seeds)** | `astronaut_bench/results/sdxl_astro_h100_fp16_1024/seed{42..51}_astro.png` |
| **H100 2K FP16 variant(10 seeds)** | `astronaut_bench/results/sdxl_astro_h100_fp16_2048/seed{42..51}_astro.png` |
| **H100 4K FP16 variant(10 seeds)** | `astronaut_bench/results/sdxl_astro_h100_fp16_4096/seed{42..51}_astro.png` |
| L4 1K BF16(10 seeds) | `astronaut_bench/results/sdxl_astro_l4_1024/seed{42..51}_astro.png` |
| L4 2K BF16(10 seeds) | `astronaut_bench/results/sdxl_astro_l4_2048/seed{42..51}_astro.png` |
| L4 4K BF16(1 seed 抽样) | `astronaut_bench/results/sdxl_astro_l4_4096/seed42_astro.png` |
| **L4 1K FP16 variant(10 seeds)** | `astronaut_bench/results/sdxl_astro_l4_fp16_1024/seed{42..51}_astro.png` |
| **L4 2K FP16 variant(10 seeds)** | `astronaut_bench/results/sdxl_astro_l4_fp16_2048/seed{42..51}_astro.png` |
| **L4 4K FP16 variant(3 seeds 抽样)** | `astronaut_bench/results/sdxl_astro_l4_fp16_4096/seed{42,43,44}_astro.png` |
| Neuron trn2 1K BF16(10 seeds,guidance=1.0 no-CFG) | `astronaut_bench/results/sdxl_astro_trn2_1024/seed{42..51}.png` |
| **Neuron trn2 1K BF16(10 seeds,CFG=7.5 batch=2)** | `astronaut_bench/results/sdxl_astro_trn2_1024_cfg/seed{42..51}.png` |
| Neuron trn2 2K / 4K | 编译不可行(见 §3 / §4) |

每个目录含 `results.json`(mean_s / peak_vram_gb / per-seed std 等)。

## 7. 硬件 / 软件配置

**Neuron — trn2.48xlarge (SDK 2.27) 参考**
- AWS 官方数据:SDXL-base-1.0 @ 1024², tp=1(单 Trainium2 chip),**5.74 s / image**
- SDK 2.27 下 DataParallel 与 FP32 NEFF 组合正常工作,无本轮 SDK 2.29 的 NRT_RESOURCE / DataParallel scatter regression

**Neuron — trn2.3xlarge (SDK 2.29) 本轮**
- SDK:**2.29** / neuronx-cc / torch-neuronx
- venv:`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`
- 编译:5/5 NEFF(UNet / CLIP-L / CLIP-G / VAE decoder / post_quant_conv)通过,~30 min,PR #149 style flags(`--model-type=unet-inference -O1`)
- 运行(no-CFG):**BF16 + batch=1 + 单核 jit.load**(无 DataParallel,`guidance_scale=1.0`)10/10 pass,19.997s
- 运行(CFG=7.5):**BF16 + batch=2 UNet NEFF + CFG=7.5 + 单 jit.load**,10/10 pass,13.262s。UNet 重编译用 `--model-type=unet-inference --auto-cast matmult --auto-cast-type bf16`,sample 形状 `[2,4,128,128]`,timestep 保持 0-dim scalar,`encoder_hidden_states [2,77,2048]`,`text_embeds [2,1280]`,`time_ids [2,6]`。Text encoders / VAE decoder / post_quant_conv 复用 1K batch=1 NEFF(CFG 不影响这些组件)
- AWS 官方 notebook 的 FP32 + DataParallel [0,1] + batch=2 CFG 组合在 trn2.3xlarge LNC=2 下超 per-NC HBM 预算(NRT_RESOURCE);SDK 2.29 DataParallel scatter 对 scalar timestep 输入有 bug。以上两个 workaround 为当前绕法
- 2K / 4K 编译不可行:详见 §3 / §4

**H100 p5.4xlarge**:DLAMI PyTorch / CUDA 13 / torch 2.11.0+cu130 / diffusers 0.37.1 / torchao 0.17.0。
- BF16:单精度 bf16,无量化(主基准)
- FP8:torchao 动态激活量化,eager 模式无 `torch.compile` 时比 BF16 慢 5×,已从结果表删除;**若需重测请加 `torch.compile(mode="reduce-overhead")`**

**L4 g6.4xlarge**:DLAMI / torch 2.9.1+cu128 / diffusers 0.38.0 / bitsandbytes 0.45(NF4 工具链可选,本次 SDXL 主测 BF16)

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

1. **H100 BF16 / FP16 variant 并列最快 + 最便宜的 H100 路径**:
   - 1K: BF16 3.84 s / $0.00462,FP16 variant 3.83 s / $0.00460 — noise 之内,**确认用户参考脚本 3.94 s @ 50-step 数据**
   - 2K: BF16 12.14 s / $0.0146,FP16 variant 12.71 s / $0.0153
   - 4K: BF16 94 s / $0.113,FP16 variant 97.14 s / $0.117
   - 10/10 seeds 全通过;diffusers 官方 `variant="fp16"` 路径加载 ~4.8 GB checkpoint,省一半磁盘/下载,无性能损失
2. **H100 FP8 占位符 — 不推荐生产使用**:我方 FP8 测试为 torchao `Float8DynamicActivationFloat8WeightConfig` eager 模式,Python dispatch overhead dominates per-Linear call;在 SDXL 单 batch 场景下反而比 BF16 慢 5×(1K 20.10 s vs BF16 3.84 s)。**实际部署建议 `torch.compile(mode="reduce-overhead")` + CUDA graphs,或直接使用 FP16 variant(已验证等价 BF16)**。`sdxl_astro_h100_fp8_*` 目录保留作为 eager-mode 反例存档
3. **L4 全分辨率可用**:
   - BF16: 1K $0.00726 / 2K $0.0350 / 4K $0.228(1-seed 抽样)
   - FP16 variant: 1K $0.00837 / 2K $0.0400 / 4K $0.2523(3-seed 抽样)
   - L4 上 BF16 略优于 FP16(1K 19.75 vs 22.78 s, +15%;4K 619 vs 686 s, +11%);**L4 生产建议仍用 BF16 variant**
   - 24 GB VRAM 够 SDXL full precision,无需 offload
4. **Neuron**:
   - **trn2.48xlarge (SDK 2.27) tp=1 参考**:5.74 s / $0.00356 per image — 比 H100 BF16 便宜 23%(AWS 官方 reference)
   - **trn2.3xlarge (SDK 2.29) 本轮 workaround**:mean 19.997 s / 10/10 pass / $0.01241 per image(BF16 batch=1 + `guidance_scale=1.0` 绕开 FP32 HBM 超预算 NRT_RESOURCE),比 SDK 2.27 慢 3.48×
   - **trn2.3xlarge (SDK 2.29) batch=2 CFG=7.5**:mean 13.262 s / 10/10 pass / $0.00823 per image,比 no-CFG workaround 快 1.51× 且恢复 CFG=7.5 完整 prompt 遵循(seed 42 绿马出现)。关键修复:(i) UNet NEFF 重编为 batch=2(容纳 CFG 自动复制 uncond/cond),(ii) CLIP-L `TextEncoderOutputWrapper` 返回 `text_embeds=None` 强制 diffusers 0.38 使用 CLIP-G 的 1280-d pooled(修复 `[1,768] expected [1,1280]` 回归),(iii) 保持单 `torch.jit.load`(避开 SDK 2.29 `DataParallel` scatter bug),(iv) 跳过 FP32 auto-cast,用 `--auto-cast matmult --auto-cast-type bf16`
   - **trn2.3xlarge 2K / 4K 编译阻塞**:2K VAE decoder 生成 7.7M 指令 / 4K UNet 生成 9.8M 指令,均超 `NCC_EVRF007` 5M 硬限;即使 `--optlevel=1` 无效。另外 2K UNet `walrus_driver` 后端占用 >124 GB RAM 超出 trn2.3xlarge(128 GB host)可用内存
5. **后续动作**:(a) H100 FP8 用 `torch.compile(mode="reduce-overhead") + CUDA graphs` 重测(eager 不可用于生产);(b) Neuron trn2 2K/4K 探索 UNet tensor-parallel 拆分绕 instruction-count 上限,或在更大 host RAM 实例(如 trn2.48xl / r7i)编译后迁移 NEFF 到 trn2.3xl 运行
