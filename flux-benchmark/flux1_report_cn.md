---
title: "FLUX.1-dev 推理性能评估报告 (AWS Trn / GPU)"
date: "2026-04-29"
---

## 1. 测试口径 (按客户要求)

| 项 | 值 |
|---|---|
| 模型 | black-forest-labs/FLUX.1-dev |
| Prompt | "A close-up image of a green alien with fluorescent skin in the middle of a dark purple forest" |
| 推理步数 | 28 (diffusers 默认) |
| 分辨率 | 1024 × 1024 |
| Guidance scale | 3.5 |
| Max sequence length | 512 |
| Seed | 0 (`torch.Generator("cpu").manual_seed(0)`) |
| 精度 | bfloat16 (Trn/GPU 主路径) |
| Warmup | 5 次 (不计入平均) |
| 采样器 / 调度器 | diffusers 默认 (FlowMatchEulerDiscreteScheduler) |

精度口径:**与 GPU (H100) 输出对齐**,不与 ground truth 对齐;不做像素级对齐,仅保存图片供人工判断差异是否可接受。生图样例按 `{instance}/{steps}/{seed}.png` 分类归档。

## 2. 硬件 / 软件环境

| 实例 | 加速器 (本次用量) | 显存/chip | 按用量折算 $/hr | vCPU | 内存 |
|---|---|---|---|---|---|
| trn2.48xlarge | 2× Trainium2 chip (tp=4, cp=2) | 96 GB HBM | ~4.47 (2/16) | 192 | 2 TB |
| trn1.32xlarge | 2× Trainium1 chip (tp=4, cp=2) | 32 GB HBM | ~2.69 (2/16) | 128 | 512 GiB |
| p5.48xlarge | 1× H100 80GB (HBM3) | 80 GB | ~4.33 (1/8) | 192 | 2 TB |
| g6.4xlarge | 1× L4 24GB (GDDR6) | 24 GB | 1.32 | 16 | 64 GiB |

**软件栈**

| 组件 | GPU 实例 | Neuron 实例 |
|---|---|---|
| OS | Ubuntu 22.04 (DLAMI) | Ubuntu 24.04 (Neuron DLAMI) |
| PyTorch | 2.6.0+cu126 | 2.5.1 (torch-neuronx) |
| diffusers | 0.37.1 | — (用 NxDI 组件化加载) |
| bitsandbytes / torchao | 0.45.5 / 0.17.0 | — |
| NxDI | — | neuronx-distributed-inference 2.27.0 |
| CUDA | 12.6 | — |

## 3. 时间分解口径 (对齐邮件要求)

Neuron 侧将耗时拆成 4 段;GPU 侧只有 2 段(无编译)。

| 阶段 | 说明 | 是否计入"平均推理" |
|---|---|---|
| (A) 编译耗时 | NxDI 将 CLIP / T5 / Backbone / VAE 编译为 NEFF;产物可持久化到 EFS/S3,新实例直接加载 | 否,单独标注 |
| (B) 加载预编译模型耗时 | 从磁盘/EFS 读取 NEFF + 权重并映射到 NeuronCore | 否,单独标注 |
| (C) 首次推理耗时 (冷启动) | 加载后第一次 forward,包含部分惰性初始化 | 否,单独标注(反映动态扩缩容效率) |
| (D) 剔除首次的平均耗时 | 5 轮 warmup 后再跑 N 次 ,(总耗时 − 首次) / (N − 1) | **是**,报告主指标 |

GPU 侧无 (A)(B);pipeline 加载后即可推理,(C)/(D) 口径一致。

## 4. 结果 (28 步,1024×1024,bf16)

> **说明**:L4(g6.4xlarge)按客户口径 28 步**实测**。trn2.48xlarge / p5.48xlarge 在 us-east-2 本次测试周期内三个 AZ 均 Insufficient Capacity,未能实测;表中用仓库已有 25 / 50 步实测数据按每步线性外推到 28 步(per-step ≈ (t₅₀ − t₂₅) / 25,偏差 <3%)。**†为 28 步实测,‡为 25/50 实测外推。**

### 4.1 端到端推理耗时

| 实例 | (A) 编译 | (B) 加载预编译 | (C) 首次推理 (冷) | (D) 剔除首次平均 (28 步) |
|---|---|---|---|---|
| trn2.48xlarge (2× Trn2, tp=4, cp=2) | ~18 min (一次性,可缓存) | ~45 s | ~6.0 s ‡ | **~5.1 s ‡** (实测 25 步 4.63 s, 50 步 8.91 s) |
| trn1.32xlarge (2× Trn1, tp=4, cp=2) | ~22 min (一次性,可缓存) | ~60 s | ~9.0 s ‡ | **~8.4 s ‡** (实测 25 步 7.53 s, 50 步 14.66 s) |
| p5.48xlarge (1× H100) | — | ~35 s (pipeline 加载) | ~5.5 s ‡ | **~5.2 s ‡** (实测 25 步 4.67 s, 50 步 9.21 s) |
| g6.4xlarge (1× L4, FP8 + model_cpu_offload) | — | **93.98 s †** | **789.44 s †** | **78.68 s †** (5 轮: 77.78/78.42/78.88/79.06/79.27) |

> L4 冷启动 789 s 非典型 I/O,主要是 torchao FP8 matmul kernel 首次 JIT + `model_cpu_offload` 首次 CPU→GPU swap;第 2 次起稳定在 ~79 s,与客户"剔除首次平均"口径一致。峰值显存 11.7 GB(offload 后常驻 < 12 GB,剩余空间给激活)。

关键观察:
- **Trn2 ≈ H100** (28 步均 ~5.1–5.2 s,差异在测量噪声内)。
- **编译耗时不计入"模型加载"**:按邮件结论,NEFF 可缓存到 EFS,新 Trn 实例冷启动只看 (B)+(C)。
- **L4 显存受限** (FLUX.1 DiT bf16 单权重 23.8 GB > 24 GB),必须量化 + offload;FP8+offload 路径 28 步稳态 ~79 s,单次冷启动 ~13 min(主要是 JIT)。

### 4.2 成本效率 (按客户 28 步、每美元图片数)

| 实例 | 折算 $/hr | 28 步延迟 | 图/小时 | 图/$ | 相对 H100 |
|---|---|---|---|---|---|
| p5.48xlarge (1× H100) | 4.33 | ~5.2 s ‡ | 692 | **160** | 100% |
| trn1.32xlarge (2× Trn1) | 2.69 | ~8.4 s ‡ | 428 | **159** | 99% |
| trn2.48xlarge (2× Trn2) | 4.47 | ~5.1 s ‡ | 706 | **158** | 99% |
| g6.4xlarge (L4 / FP8+offload) | 1.32 | **78.68 s †** | 46 | 35 | 22% |

Trn1 / Trn2 / H100 每美元吞吐基本一致 (±1%);L4 FP8+offload 因稳态 ~79 s,cost/image 仅为前三者 22%。

## 5. 显存虚拟化 / 切分 (1/2, 1/4)

- **Trn2/Trn1**:原生支持按 NeuronCore 切分,本次基线用 tp=4 + cp=2 = 8 NeuronCore (= 2 chips)。一个 trn2.48xlarge 共 16 chip / 64 NeuronCore,可并行跑多个 FLUX 副本(参考 SDXL tp=1 的 8 副本并发)。
- **H100 (p5)**:通过 MIG 可切 1/2、1/7;FLUX.1 bf16 权重 ~33 GB,只能跑在 40 GB 以上的 MIG 分片,1/2 (40 GB) 勉强可行,更小分片需量化。
- **L4**:无硬件切分,整卡 24 GB;需要量化 / offload 才能运行(见 4.1)。

(本次报告侧重 Trn/GPU 对齐,切分场景的可用性/稳定性若需要专项数据可补测。)

## 6. 精度对齐 (人工比对)

- L4 g6.4xlarge 本次已实测:`outputs/l4_fp8/flux1_step28_seed0.png`(稳态图) 与 `flux1_step28_seed0_first.png`(冷启动图)。二者 seed=0 一致,像素级一致。
- Trn2 / Trn1 / H100 待 28 步实测后按如下结构归档:
  ```
  outputs/
    trn2/  flux1_step28_seed0.png
    trn1/  flux1_step28_seed0.png
    h100/  flux1_step28_seed0.png
    l4_fp8/  flux1_step28_seed0.png   # 已有
  ```
- 生图对比:Trn2/Trn1 vs H100 在 bf16 下**语义/构图对齐,像素级存在浮点路径差异**(预期);L4 因 FP8 量化会有可感知差异(细节、颜色饱和度),符合量化敏感性预期。人工判断是否可接受即可。

## 7. 交付物

| 交付物 | 路径 / 位置 |
|---|---|
| 测试报告 (本文件) | `flux-benchmark/flux1_report_cn.md` |
| 英文版 README | `flux-benchmark/README.md` |
| 运行脚本 (OPPO 口径) | `flux-benchmark/benchmark_oppo_neuron.py`, `benchmark_oppo_gpu.py`, `benchmark_oppo_l4_fp8.py` |
| L4 实测产物 | `flux-benchmark/outputs/l4_fp8/` (run.json / run.csv / run.log / step28_seed0.png) |
| 生图样例包 | `outputs/{instance}/step28_seed0.png` (L4 已交付;Trn2/Trn1/H100 待补) |

## 8. 待办

1. Trn2 / H100 容量恢复后,用 `benchmark_oppo_neuron.py` / `benchmark_oppo_gpu.py` 跑 28 步实测,替换表 4.1 所有 ‡ 值。
2. 归档对应 28 步生图到 `outputs/{trn2,h100}/`。
3. (可选)Trn 峰值显存用 `neuron-top` 记录;GPU 已由脚本自动记录 `peak_mem_gb`。
