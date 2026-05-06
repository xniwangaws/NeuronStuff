# S3Diff 超分辨率 Benchmark — AWS Trainium2 vs NVIDIA H100/L4

**模型**: S3Diff (ECCV 2024) — 一步扩散 4× 超分辨率 (SD-Turbo + LoRA)
**测试**: cat, bus 图像, 输入 256×256, 输出 1K/2K/4K
**精度**: BF16

---

## 1. 核心指标 — 客户关心的耗时

> 客户定义的 warm_mean 计算公式:
> `warm_mean = (N 次总耗时 − 首次推理耗时) / (N − 1)`

| 实例 | 加速器 | 分辨率 | 编译耗时 (s) | 加载 (s) | 冷启动 (s) | Warm 平均 (s) | 峰值显存 (GB) |
|---|---|---|---|---|---|---|---|
| **trn2.3xlarge** | **1× Trainium2** | **1K** (256→1024) | **~90** | **13.4** | **96.08** | **6.14** | ~20 |
| trn2.3xlarge | 1× Trainium2 | 2K (512→2048) | ~40 | 13.4 | 100.78 | 60.26 | ~22 |
| trn2.3xlarge | 1× Trainium2 | 4K (1024→4096) | ~51 | 13.4 | 354.49 | 303.18 | ~24 |
| p5.4xlarge | 1× H100 80GB | 1K | — (无预编译) | 4.80 | 9.64 | 1.26 | 9.0 |
| p5.4xlarge | 1× H100 80GB | 2K | — | 4.02 | 24.67 | 24.26 | 15.7 |
| p5.4xlarge | 1× H100 80GB | 4K | — | 4.03 | 109.71 | 107.54 | 42.2 |
| g6.4xlarge | 1× L4 24GB | 1K | — | 4.52 | 6.41 | 2.34 | 7.9 |
| g6.4xlarge | 1× L4 24GB | 2K | — | 4.07 | 29.11 | 28.45 | 15.2 |
| g6.4xlarge | 1× L4 24GB | 4K | — | 4.08 | 132.42 | 130.63 | 16.5 |

说明:
- **编译耗时** (Trn2): `torch.compile(backend="neuron")` 产生的 NEFF 首次 JIT 编译. H100/L4 不需要 (eager CUDA).
- **加载**: S3Diff 各组件从磁盘加载 + 权重移动到加速器. Trn2 含 NEFF cache 检查.
- **冷启动**: 第一次推理. 注: Trn2 冷启动 = 编译 + 推理, 客户公式 (N=3 Trn2, N=10 GPU).
- **Warm 平均**: 客户公式 `(total_N - cold) / (N-1)`. 复用 NEFF cache, 无重编译.
- **峰值显存**: 单卡 / 单核 HBM (Trn2 LNC=2 每核 24GB; GPU 整卡 VRAM).

### 1.1 按需价格 & 每张图成本

| 实例 | 按需价 $/hr | 1K $/图 | 2K $/图 | 4K $/图 |
|---|---|---|---|---|
| trn2.3xlarge (Trainium2) | $2.235 | $0.00381 | $0.0374 | $0.1882 |
| p5.4xlarge (H100) | $4.326 | $0.00151 | $0.0292 | $0.1293 |
| **g6.4xlarge (L4)** | **$1.323** | **$0.00086** | **$0.0105** | **$0.0480** |

> 计算公式: `$/图 = $/hr × warm_s / 3600`
> 价格以 AWS on-demand 官方价为准 (trn2.3xlarge 以 Melbourne 为基准, 其他 us-east-1).

**成本效率**: L4 (g6.4xlarge) 在该任务是性价比最高; H100 约 L4 的 1.75-2.8×; Trn2 约 L4 的 3.6-4.4×.

---

## 2. 精度 / 画质对比 (PSNR)

| 设备 | 1K PSNR (dB) | 2K PSNR (dB) | 4K PSNR (dB) |
|---|---|---|---|
| AWS Trn2.3xlarge | 43.10 (vs CPU fp32) | 40.60 (vs H100) | 37.07 (vs H100) |
| NVIDIA H100 BF16 | 45.10 (vs CPU fp32) | reference | reference |
| NVIDIA L4 BF16 | 45.15 (vs CPU fp32) | — | — |

说明:
- 1K PSNR 基准 = CPU fp32 输出 (同 pipeline, fp32)
- 2K / 4K PSNR 基准 = H100 bf16 输出 (CPU fp32 在高分辨率耗时过长)
- Trn2 相比 H100 ~2 dB 差距主要来自 BF16 matmul 累加噪声, 业界普遍, 肉眼无差别

---

## 3. 生图示例 — 公交车

测试图来源: `https://ultralytics.com/images/bus.jpg`

### 3.1 输入 256×256 LQ

![Bus LQ 256](customer_report/images/input_bus_LQ_256.png)

### 3.2 Trn2 1K 输出 (1024×1024, 4× 超分)

![Trn2 Bus 1K](customer_report/images/trial6_bus_1k.png)

### 3.3 Trn2 2K 输出 (2048×2048, 输入 512→2K)

![Trn2 Bus 2K](customer_report/images/trial6_bus_2k.png)

### 3.4 Trn2 4K 输出 (4096×4096, 输入 1024→4K)

![Trn2 Bus 4K](customer_report/images/trial6_bus_4k.png)

---

## 4. 硬件 / 软件配置

### AWS Trn2.3xlarge

| 项目 | 值 |
|---|---|
| 实例类型 | trn2.3xlarge (LNC=2, 4 logical NeuronCores, 96 GB HBM) |
| vCPU / 内存 | 12 vCPU / 96 GB |
| AMI | Deep Learning AMI Neuron (Ubuntu 24.04) 20260502 |
| Neuron SDK | 2.29 |
| Python | 3.12 |
| PyTorch | 2.10 (eager) |
| torch-neuronx | 0.1.0+5e711e8 (eager) |
| neuronx-cc | 2.24.8799.0+6f62ff7c |
| diffusers | 0.34.0 |
| peft | 0.19.1 |
| transformers | 4.57.3 |

### NVIDIA GPU

- **H100**: AWS p5.4xlarge (1× H100 80GB), PyTorch 2.1 + CUDA 12.1, diffusers 0.34
- **L4**: AWS g6.4xlarge (1× L4 24GB), PyTorch 2.1 + CUDA 12.1, diffusers 0.34

### 关键运行参数

| 参数 | 值 |
|---|---|
| Batch size | 1 |
| Diffusion steps | 1 (S3Diff 一步模型) |
| Scheduler | DDPMScheduler |
| Seed | 123 (固定, 可跨栈对比) |
| 正向 prompt | "A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting." |
| 负向 prompt | "oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth" |
| Tile (Trn2) | latent_tiled_size=96, overlap=32, vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224 |
| 编译方式 (Trn2) | `torch.compile(backend="neuron", dynamic=False, fullgraph=False)`, 作用于 16 个 `Transformer2DModel`, 其他模块 eager |
| 编译 flags | `--auto-cast=matmult -O1` |

---

## 5. 运行命令 (复现)

```bash
# 1K
python customer_report/scripts/phase_bisect_hires.py --scope t2d \
    --lq_image <path_to_LQ_256.png> --lq_size 256 \
    --output_image /tmp/out_1k.png --num_inferences 3

# 2K
python customer_report/scripts/phase_bisect_hires.py --scope t2d \
    --lq_image <path_to_LQ_512.png> --lq_size 512 \
    --output_image /tmp/out_2k.png --num_inferences 3

# 4K
python customer_report/scripts/phase_bisect_hires.py --scope t2d \
    --lq_image <path_to_LQ_1024.png> --lq_size 1024 \
    --output_image /tmp/out_4k.png --num_inferences 2
```

---

## 6. 目录结构

| 路径 | 内容 |
|---|---|
| `README.md` | 本报告 (客户交付) |
| `customer_report/data/s3diff_benchmark.csv` | 原始测试数据 (每次 run 明细) |
| `customer_report/images/` | 生图样例: cat 1K/2K/4K + bus 1K + GPU 对比 + CPU 参考 |
| `customer_report/scripts/` | 可复现运行脚本 |
| `customer_report/logs/` | 每次测试原始 stdout |
| `docs/archive/` | 旧 README 备份 (`README_original.md`, `README_zh_original.md`) |
| `phase3/` | 早期 trace mode 方案 (24.9s, PSNR 24.55, 带 seam artifact) |
| `phase_e/` | Eager mode 实验 (8.6s) |
| `phase_r/` | DeModLoRA 算法优化 + selective compile bisect (6.14s, Trial 6 winner) |
| `phase_b/` | Trace-ready full custom UNet (基础设施, 供未来 trace API 使用) |
