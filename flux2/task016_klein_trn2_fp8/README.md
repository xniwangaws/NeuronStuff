# FLUX.2-klein-base-9B 1024² benchmark — Neuron trn2.3xlarge（BF16 / FP8，全 Neuron pipeline）

> Prompt:`"A cat holding a sign that says hello world"`,guidance 4.0（klein-base 为非蒸馏模型,走 classic CFG,**每步 2 次 DiT 前向**),50 steps,batch=1,max_sequence_length=512,seeds 42–51(共 10 个)

> **2026-09-11 更新**:原始移植把 Qwen3-8B text encoder 和 VAE decoder 留在 CPU 上,占了每张图 14.3 s(35%)。本次把两者搬到 Neuron 并缓存 RoPE 表,**BF16 端到端从 41.79 s 降到 25.94 s(1.61×)**,10/10 seeds 通过,与原路径出图 SSIM 0.988。FP8 all-Linear W8A8 叠加后见 §2 表格。§9 解释了为什么同样是 DiT 的 FLUX.1-lite 在 trn2 上只要 6.5 s,§10 给出与 L20 对比时必须核对的口径。

## 1. 设备与价格(AWS,2026-09)

| 实例 | 芯片 | 内存 | $/hr | Region |
|---|---|---|---:|---|
| **trn2.3xlarge**(Capacity Block) | 1× Trainium2(LNC=2 → 4 逻辑核,DiT TP=4) | 96 GB HBM | **$2.235** | ap-southeast-4 |
| p5.4xlarge(2026-05 参考) | 1× H100 SXM5 | 80 GB HBM3 | $4.326 | ap-northeast-1 |
| g6.4xlarge(2026-05 参考) | 1× L4 | 24 GB GDDR6 | $1.323 | sa-east-1 |

本目录的 Neuron 数字全部在**原生 `trn2.3xlarge`** 上测得(不是在 trn2.48xlarge 上切核)。H100 / L4 行来自 2026-05 的同 prompt / 同 seed / 同步数跑法([`../REPORT_flux2_klein.md`](../REPORT_flux2_klein.md)),仅作横向参考。

## 2. 1024² 端到端耗时 + HBM + $/image(以原始移植 BF16 为基准)

| 设备 | 精度 / 路径 | Mean (s) | Pass | **$/image** | vs 原始移植 | vs H100 BF16 |
|---|---|---:|---:|---:|---:|---:|
| trn2.3xl(2026-09-03) | BF16 TP=4,**TE + VAE 在 CPU**(原始移植,基准) | 41.785 | 10/10 | $0.02594 | **1.00×** | 0.58×(慢 1.73×) |
| trn2.3xl(2026-09-03) | FP8 MLP W8A8,TE + VAE 在 CPU | 38.562 | 10/10 | $0.02394 | 1.08× | 0.62× |
| trn2.3xl(2026-09-03) | FP8 all-Linear W8A8,TE + VAE 在 CPU | 37.893 | 10/10 | $0.02353 | 1.10× | 0.64× |
| **trn2.3xl(2026-09-11)** | **BF16 TP=4,TE + VAE 在 Neuron,RoPE 缓存** | **25.939** | **10/10** | **$0.01610** | **1.61×** | 0.93×(慢 1.08×) |
| trn2.3xl(2026-09-11) | FP8 all-Linear W8A8 + TE/VAE Neuron | 测试中(10 seed 运行中,下次提交补充) | | | | |
| H100 p5.4xlarge(2026-05) | BF16,diffusers eager | 24.10 | 10/10 | $0.02896 | 1.73× | 1.00× |
| H100 p5.4xlarge(2026-05) | FP8(torchao),eager | 21.18 | 10/10 | $0.02545 | 1.97× | 1.14× |
| L4 g6.4xlarge(2026-05) | FP8(BFL 官方 `klein-9b-fp8` 蒸馏 ckpt,强制 50 步,**无 CFG**) | 77.25 | 10/10 | $0.02839 | 0.54× | 0.31× |

`$/image = (Mean / 3600) × $/hr`。稳态数字,不含一次性编译和模型加载。

**核心结论**:
- 原始移植的 41.8 s 里有 **14.3 s 花在 CPU 上的 text encoder(5.1 s)和 VAE decode(9.2 s)**,这是 GPU 端不存在的额外损耗;搬到 Neuron 后两项合计 **0.41 s**。
- 全 Neuron 路径 BF16 **25.94 s**,单图成本 **$0.0161**,比 H100 BF16 便宜 44%、比 H100 FP8 便宜 37%,速度已接近 H100 BF16 eager(慢 8%)。
- 10 个 seed 的 stdev 只有 0.015 s(原路径 0.16 s):CPU 阶段去掉后延时抖动几乎消失。
- 剩下的 25.3 s 几乎全是 DiT 的 **100 次前向**(每次 253 ms)。要再快只能动 DiT:FP8(见上表)、编译选项、以及 CFG 语义本身(§9)。
- L4 那一行跑的是蒸馏 ckpt 且无 CFG(50 次前向而不是 100 次),与其它行不是同一个工作量,仅供参考。

## 3. 阶段拆分:时间去哪了(插桩实测,1024² / 50 步 / CFG)

给 `pipe.text_encoder.forward`、`pipe.transformer.forward`、`pipe.vae.decode` 加计时器(`src/bench_klein_1k.py` 的 `StageTimer`,10 seed 均值):

| 阶段 | 原始移植(CPU TE/VAE) | 全 Neuron 路径 | 做法 |
|---|---:|---:|---|
| Qwen3-8B text encoder × 2(cond + uncond) | 5.09 s | **0.12 s** | 只算前 27 层(pipeline 只用 hidden_states 9/18/27),切 3 段各 5.6–7.7 GB 放逻辑核 1/2/3 |
| DiT 100 次前向,TP=4 | 26.83 s(268 ms/次) | **25.29 s(253 ms/次)** | RoPE 表原来每次前向都在 CPU 重算,接上 `image_rotary_emb_cache_context()` |
| VAE decode | 9.24 s | **0.29 s** | 抄 NxDI FLUX.1 VAE 的做法:`--model-type=unet-inference` + GroupNorm 用 FP32 算,放逻辑核 0 |
| scheduler / 其它 | 0.26 s | 0.24 s | |
| **合计** | **41.4 s** | **25.9 s** | |

### 冷启动 / 加载(一次性)

| 阶段 | 耗时 | 说明 |
|---|---:|---|
| DiT 编译(BF16,1K) | 157 s | NEFF 可缓存 |
| Text encoder 3 段编译 | 452 + 364 + 362 s | 一次性;27 层整段编译要 1227 s 且 18.9 GB 装不进单核(见 §6) |
| VAE decoder 编译 | 558 s | 一次性 |
| 加载(仅 DiT) | 20 s | |
| 加载(DiT + TE 3 段 + VAE) | 401 s | 19 GB TorchScript 段的 `torch.jit.load` 占大头,后续可改为 NEFF 直载 |
| **稳态** | **25.94 s / image** | |

### 两个 Neuron 组件的数值验证

| 组件 | 对照 | 结果 |
|---|---|---|
| TE(3 段串联) | 完整 36 层模型经 diffusers 自己的 `_get_qwen3_prompt_embeds` 得到的 `[1, 512, 12288]` | 截断的 CPU 路径逐元素**完全一致**(max abs 0);Neuron 输出 cos 1.000,mean abs 0.067(参照范数 4.0e4),max abs 128 出现在单个 massive-activation 通道上 = 该量级下 1 个 BF16 ulp |
| VAE decode | CPU BF16 `vae.decode` | cos 0.99978,mean abs 0.0010,max abs 0.053(输出范围 [-1, 1]) |
| 端到端出图(10 seed) | 原始移植的 BF16 PNG | **SSIM 0.9876 / PSNR 28.3 dB**,MAE 0.0165 |

## 4. 同 prompt / seed 的生图对比(seed 42)

| BF16,TE/VAE 在 CPU(原始移植) | **BF16,全 Neuron** | FP8 all-Linear,TE/VAE 在 CPU | **FP8 all-Linear,全 Neuron** |
|:---:|:---:|:---:|:---:|
| ![](results/bf16/seed42_cat.png) | ![](results/bf16_neuron_aux/seed42_cat.png) | ![](results/fp8_all_linear/seed42_cat.png) | 测试中 |

10 seed 三列对比图(原始移植 BF16 / 全 Neuron BF16 / 全 Neuron FP8 all-Linear)将随 FP8 结果一起提交到 `results/comparison_grid_cpu_aux_vs_neuron_aux.png`。旧的 BF16 / MLP-FP8 / all-Linear-FP8 网格仍在 [`results/comparison_grid_bf16_mlp_all_linear.png`](results/comparison_grid_bf16_mlp_all_linear.png)。

**视觉一致性(10 seed 逐张人工核对)**:BF16 全 Neuron 与原始移植在全部 10 个 seed 上构图、猫的品种与姿态、牌子位置、字体风格一致,10/10 文字清晰可读,无噪声、色偏或伪影;差异在毛发纹理、背景抱枕花纹(seed 45)、门把手位置(seed 48)、笔画粗细这一级。唯一一处语义级差别是 seed 48:原路径写 "hello World",全 Neuron 路径写 "hello world",两者都是 prompt 的合法输出——这是 text encoder 数值微差被 100 步 diffusion 轨迹放大的结果,也是该 seed SSIM 最低(0.958)的原因。逐 seed SSIM 0.958–0.998,PSNR 21.6–34.4 dB。FP8 all-Linear 的漂移更大(vs BF16 SSIM 0.93),与 09-03 的结论相同。

## 5. 10-seed 全量 PNG 与结果文件

| 路径 | 目录 |
|---|---|
| BF16,TE/VAE 在 CPU(原始移植) | `results/bf16/seed{42..51}_cat.png`,`results/bf16/full_results.json` |
| **BF16,全 Neuron** | `results/bf16_neuron_aux/seed{42..51}_cat.png`,`results/bf16_neuron_aux/results.json`(含逐 seed 阶段拆分) |
| FP8 MLP W8A8,TE/VAE 在 CPU | `results/fp8_dynamic/` |
| FP8 all-Linear W8A8,TE/VAE 在 CPU | `results/fp8_all_linear/` |
| FP8 all-Linear W8A8,全 Neuron | `results/fp8_all_linear_neuron_aux/`(测试中) |
| 阶段拆分原始数据 | `results/stage_breakdown/cpu_aux/breakdown.json`,`results/stage_breakdown/neuron_aux/breakdown_fast.json` |
| TE / VAE 组件验证 | `results/neuron_aux_reports/te_validation_report.json`,`vae_trace_report.json` |
| 画质对比 | `results/bf16_neuron_aux/comparison_vs_cpu_aux.json`,`results/comparison_all_linear_vs_bf16.json`,`results/comparison_vs_bf16.json` |
| 日志 | `logs/`(`te_trace_attempt1_27layers.log` 和 `vae_trace_attempt1.log` 是失败的第一次尝试,保留作证据) |

## 6. 硬件 / 软件配置

**Neuron(trn2.3xlarge,ap-southeast-4c,Capacity Block)**
- AMI:Neuron SDK 2.29.1 DLAMI(`ami-07240db66d6961eea`)
- `neuronx-cc 2.24.8799.0` / `torch-neuronx 2.9.0.2.13.26312` / `neuronx-distributed 0.18.27753` / `neuronx-distributed-inference 0.9.17334` / `libneuronxla 2.2.16408` / runtime + collectives `2.34.10.0` / dkms `2.30.2.0`
- PyTorch 2.9.1 / diffusers 0.37.1 / transformers 4.57.6 / Python 3.12
- venv:`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`
- 拓扑:`NEURON_LOGICAL_NC_CONFIG=2`,`NEURON_RT_VISIBLE_CORES=0-3`

**逻辑核占用(全 Neuron 路径)**

| 逻辑核 | 常驻内容 |
|---|---|
| 0–3 | DiT(NxDI,TP=4,每核一个 shard) |
| 1 / 2 / 3 | Qwen3 text encoder 段 a / b / c(7.7 / 5.6 / 5.6 GB NEFF) |
| 0 | VAE decoder |

TE 在 DiT 之前运行、VAE 在之后,和 DiT 不抢算力,只共享 HBM。

**H100 p5.4xlarge(2026-05)**:DLAMI PyTorch 2.9 / torch 2.9.1+cu128 / diffusers 0.38.0 / FP8 via torchao。
**L4 g6.4xlarge(2026-05)**:BFL 官方 flux2 repo + `torch._scaled_mm` per-tensor E4M3 shim / `FLUX.2-klein-9b-fp8` 蒸馏 ckpt。

**实现**:DiT 基于 AWS NxDI [PR #146](https://github.com/aws-neuron/neuronx-distributed-inference/pull/146)(`contrib/flux2-klein`,Jim Burtoft)的 `NeuronFlux2KleinTransformer`(TP=4,NKI `attention_cte` flash attention);text encoder 与 VAE 用 `torch_neuronx.trace`(`src/neuron_aux.py`);其它组件(tokenizer、scheduler、VAE BatchNorm 反归一化)沿用 diffusers `Flux2KleinPipeline`。

### text encoder 为什么要切三段

- 完整 27 层一次 trace:编译通过(1227 s),数值与 CPU 参照完全一致,但 NEFF **18.9 GB**;trn2 的 96 GB HBM 按物理核对切成 4 份约 22 GiB,加载时 `NRT_RESOURCE`(`logs/te_trace_attempt1_27layers.log`)。
- 按 hidden_states 9 / 18 / 27 的天然边界切 3 段,每段是一个正规的 `transformers.Qwen3Model`(层切片、`norm` 换成 Identity、共享权重存储、b/c 段走 `inputs_embeds`),复用 HF 自己的 mask / RoPE,不手写 attention。
- 三段串联 60.7 ms(每段 ≈ 20 ms),每张图调用 2 次(cond + uncond)。

### VAE 为什么第一次编译失败

- 默认 model type + BF16 GroupNorm:`NCC_IXTP002`,10.19M 条指令超过 10M 上限(`logs/vae_trace_attempt1.log`)。
- 改用 NxDI FLUX.1 VAE decoder 的同一套设置(`--model-type=unet-inference -O1`,52 个 GroupNorm 用 FP32 计算)后编译通过,285 ms。klein 的 `AutoencoderKLFlux2` 与 FLUX.1 VAE 同为 `[128, 256, 512, 512]` 结构、同为 128×128 latent 网格,只是 latent 通道 32 vs 16。

## 7. 运行脚本(快速复现)

脚本假定:模型在 `/mnt/nvme/flux2-klein/weights`,源码在 `/mnt/nvme/flux2-klein/src`,venv 为 `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`。第一次运行会编译 DiT,以及(默认开启的)text encoder 三段和 VAE decoder 到 `FLUX2_AUX_COMPILE_DIR`(默认 `/mnt/nvme/flux2-klein/compiled_aux_1024`);这两类产物只依赖分辨率,不依赖 DiT 精度,BF16 和 FP8 共用。

```bash
# BF16,全 Neuron(默认)
scripts/run_bf16_remote.sh --steps 50 --seeds 42 43 44 45 46 47 48 49 50 51 --warmups 1

# BF16,原始移植(TE / VAE 在 CPU),用于复现基准
FLUX2_OUTPUT_DIR=/mnt/nvme/flux2-klein/outputs_bf16_cpu_aux \
scripts/run_bf16_remote.sh --no-neuron-aux --steps 50 --seeds 42 43 44 45 46 47 48 49 50 51 --warmups 1

# FP8 all-Linear W8A8 + 全 Neuron:先离线导出 per-row E4M3 checkpoint,再跑
python src/export_fp8_checkpoint.py --scope all_linear \
  --output /mnt/nvme/flux2-klein/checkpoints/fp8_all_linear_per_row
FLUX2_FP8_SCOPE=all_linear FLUX2_FP8_ACTIVATION=dynamic scripts/run_fp8_remote.sh \
  --transformer-checkpoint /mnt/nvme/flux2-klein/checkpoints/fp8_all_linear_per_row \
  --steps 50 --seeds 42 43 44 45 46 47 48 49 50 51 --warmups 1

# 单独编译 / 验证 text encoder 三段与 VAE(独立于 pipeline)
NEURON_RT_VISIBLE_CORES=1 python src/trace_text_encoder_neuron.py   # 三段编译 + CPU 参照对比
python src/validate_te_segments.py                                    # 三段分放核 1/2/3 的数值与延时
NEURON_RT_VISIBLE_CORES=0 python src/trace_vae_neuron.py             # VAE 编译 + CPU 参照对比

# 阶段拆分 / 画质对比 / 对比图
python src/breakdown_timing.py        # 原始移植(CPU TE/VAE)分阶段计时
python src/breakdown_timing_fast.py   # 全 Neuron 路径分阶段计时
python scripts/compare_bf16_fp8.py --bf16-dir results/bf16 --fp8-dir results/bf16_neuron_aux --output results/bf16_neuron_aux/comparison_vs_cpu_aux.json
python scripts/make_labeled_grid.py --column "BF16 CPU TE/VAE=results/bf16" --column "BF16 all Neuron=results/bf16_neuron_aux" --column "FP8 all-Linear all Neuron=results/fp8_all_linear_neuron_aux" --output results/comparison_grid_cpu_aux_vs_neuron_aux.png
```

## 8. 结论

1. **klein-base 1024² / 50 步 / CFG 在 trn2.3xlarge 上 BF16 25.94 s,10/10 pass**,比 09-03 报告的 41.79 s 快 1.61×;$/image **$0.0161**,比 H100 BF16 便宜 44%。
2. 之前"没优势"的直接原因是**移植不完整**:text encoder 和 VAE 留在 CPU,占 35%。这不是 Trainium 的算力问题。
3. 剩余时间 98% 在 DiT 的 100 次前向。klein 每次前向约 76 TFLOP,253 ms 对应 ≈300 TFLOPS,即 trn2 BF16 峰值(≈667 TFLOPS)的 **~45%**,与 FLUX.1-lite 在同一 SDK 上的 ~41% 一致;DiT 实现本身没有明显低效(§9)。
4. FP8 all-Linear W8A8 在 CPU-aux 路径上给了 9.3%,叠加在全 Neuron 路径上的数字见 §2;代价是画质漂移(SSIM 0.93),MLP-only 是更保守的折中。
5. 与 L20 对比之前必须先对齐口径(§10):按 L20 的 BF16 峰值算,这个工作量在 L20 上**不可能低于 64 s**。

## 9. 为什么同样是 DiT,FLUX.1-lite-8B 在 trn2 上是 6.5 s 而 klein 是 26 s

两者在 Neuron 上的**每 FLOP 效率一样**,差在工作量:

| | FLUX.1-lite-8B(TP=4,[HANDSON](https://github.com/qingzwang/neuronx-distributed-inference/blob/model/flux1-lite-8B/contrib/models/flux.1-lite-8B/HANDSON.md)) | FLUX.2-klein-base-9B(TP=4,本目录) |
|---|---:|---:|
| 结构 | 8 双流 + 38 单流,d=3072(24×128),MLP 4× GELU | 8 双流 + 24 单流,d=4096(32×128),MLP 3× SwiGLU |
| 1024² token 数 | 4096 图 + 512 文 | 4096 图 + 512 文 |
| 每次前向 FLOPs(估算) | ≈ 60 TFLOP | ≈ 76 TFLOP(1.27×) |
| 实测每次前向 | 217 ms | 253 ms |
| 达到的算力 / MFU | ≈ 276 TFLOPS / 41% | ≈ 300 TFLOPS / 45% |
| guidance | **蒸馏,CFG 关**:每步 1 次前向 | **非蒸馏,classic CFG**:每步 2 次前向 |
| 步数 | 28 | 50 |
| **每张图前向次数** | **28** | **100(3.6×)** |
| **DiT 时间 / 图** | **6.1 s** | **25.3 s(4.2×)** |

1.27 × 3.6 ≈ 4.5×,与实测 4.2× 吻合。也就是说 klein 在 trn2 上"慢"不是移植质量问题,而是 klein-base 这个 checkpoint 本身每张图要算 4.5 倍的 FLOPs。同一条结论对任何硬件都成立,包括 L20。

## 10. 与 L20 对比时的口径核对

客户侧反馈 "L20 BF16 与 trn2 TP=4 基本持平(≈ 42 s)"。按算力核一下:

- klein-base 1024² / 50 步 / CFG:100 次前向 × 76 TFLOP ≈ **7.6 PFLOP / 图**。
- L20 BF16 Tensor Core 峰值 **119.5 TFLOPS**(dense)。7.6 PFLOP ÷ 119.5 TFLOPS = **64 s,这是 100% 利用率的下限**;按 FLUX.1-lite 在 L20 上反推的 ~60% 利用率,应在 **~105 s**。
- "42 s" 需要 181 TFLOPS BF16,是 L20 峰值的 152%,**不可能是同一个工作量**。

因此对比前需要拿到客户 L20 的:模型变体(`klein-base-9B` 还是蒸馏的 `klein-9b`)、分辨率、步数、`guidance_scale`(≤1 或蒸馏版都会让前向次数减半)、精度(FP8 峰值 239 TFLOPS)、是否 torch.compile / TensorRT、以及 42 s 是否包含 text encoder 和 VAE。在同一口径下,trn2.3xlarge 的 25.9 s 应当是 L20 的 ~4×。

## 11. FP8 W8A8 实现细节(2026-09-03,保留)

FP8 权重不来自单独下载的 checkpoint:起点是 Hugging Face BF16 checkpoint,`src/export_fp8_checkpoint.py` 把选定范围转换成含 E4M3 权重与 FP32 per-output-row scale 的 NxDI 分片 checkpoint。

| Scope | E4M3 per-row 权重 | 动态 E4M3 激活 | FP8 权重数 | checkpoint 大小 |
|---|---|---|---:|---:|
| `mlp` | 所有 block 的 MLP projection | MLP Linear 输入 | 120 | 12.12 GB |
| `mlp_attention` | MLP + Q/K/V + attention 输出 projection | 对应 Linear 输入 | 280 | 9.44 GB |
| `all_linear` | Transformer 内全部 289 个 Linear/GEMM | 全部 Transformer Linear 输入 | 289 | 9.09 GB |

`all_linear` 是本实验中最接近 "模型 W8A8" 的含义:所有 Transformer Linear/GEMM 用 FP8 权重 + FP8 激活;normalization、RoPE、softmax、残差和 NKI attention kernel 仍为 BF16/FP32。weight-only 模式(E4M3 存储、BF16 GEMM 前反量化)是 W8A16 存储路径,没有加速(41.99 s)。

离线导出的 checkpoint 经正常 NxDI 分片路径加载,与加载时量化路径在同 seed 4-step 测试中 PNG 逐字节一致(`results/checkpoint_equivalence.json`)。

实现要点:
1. 权重按输出行(`axis=0`)量化为 E4M3,scale 形状 `[out_features, 1]`,FP32。
2. Trainium2 上 `float8_e4m3fn` HLO 需要 `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3`,且必须与 NxDI 的 `--verify-hlo=true` 合并在同一个 `--internal-hlo2tensorizer-options` 参数里。
3. NxD 动态激活量化产生 `[batch, sequence, 1]` 的 scale;shim 避免在乘回 3D 输出前多加一维。
4. timestep / modulation projection 输入是 2D;all-linear 路径临时加一个长度 1 的 sequence 维以保持 scale 可广播。
5. diffusers 0.37.1 `Flux2KleinPipeline` 不接受 `negative_prompt`,使用 Klein Base 内置的空 unconditional prompt。
6. timestep 正弦嵌入分母为 `half_dim`,与参考实现一致。

10-seed 画质(vs 原始移植 BF16):MLP W8A8 SSIM 0.9564 / PSNR 22.87 dB;all-Linear W8A8 SSIM 0.9282 / PSNR 19.96 dB。这些是诊断性像素指标,不是感知质量评测;diffusion 轨迹会放大小的数值差异。

## 12. 目录内容

- `src/modeling_flux2_klein.py`:NxDI FLUX.2 Klein transformer(TP=4,可选 FP8)
- `src/neuron_aux.py`:Neuron text encoder(Qwen3 三段)与 VAE decoder 的编译 / 加载 / pipeline 替换
- `src/application.py`:compile / load / generate 编排,把 Neuron TE、VAE、RoPE 缓存接进 `Flux2KleinPipeline`
- `src/bench_klein_1k.py`:benchmark 驱动(含 `StageTimer` 阶段计时,`--no-neuron-aux` 回退原始移植)
- `src/export_fp8_checkpoint.py`:离线 per-row E4M3 导出
- `src/trace_text_encoder_neuron.py`、`src/validate_te_segments.py`、`src/trace_vae_neuron.py`:独立的组件编译与验证脚本
- `src/breakdown_timing.py`、`src/breakdown_timing_fast.py`:两条路径的分阶段计时
- `scripts/`:启动包装、画质对比、对比图
- `results/`、`logs/`:见 §5
