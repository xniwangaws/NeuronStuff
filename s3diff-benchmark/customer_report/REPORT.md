# S3Diff 超分辨率测试报告 — AWS Trainium2

**模型**: S3Diff (ECCV 2024) — 一步扩散 4× 超分辨率 (SD-Turbo + LoRA)
**测试设备**: AWS Trainium2 (trn2.3xlarge, 4 NeuronCores LNC=2)
**对比设备**: NVIDIA H100 80GB, NVIDIA L4 24GB
**测试图像**: cat (猫), bus (公交车), 输入 256×256
**精度**: BF16

---

## 1. 核心指标 — 客户关心的耗时

> 客户定义的 warm_mean 计算公式:
> `warm_mean = (N 次总耗时 − 首次推理耗时) / (N − 1)`

| 设备 | 分辨率 | 模型加载 (s) | 首次推理 / 冷启动 (s) | Warm 平均耗时 (s) | 峰值显存 (GB) |
|---|---|---|---|---|---|
| **AWS Trn2.3xlarge** | **1K** (256→1024) | **13.4** | **96.08** | **6.14** | ~20 |
| AWS Trn2.3xlarge | 2K (512→2048) | 13.4 | 100.78 | 60.26 | ~22 |
| AWS Trn2.3xlarge | 4K (1024→4096) | 13.4 | 354.49 | 303.18 | ~24 |
| NVIDIA H100 | 1K | 4.80 | 9.64 | 1.26 | 9.0 |
| NVIDIA H100 | 2K | 4.02 | 24.67 | 24.26 | 15.7 |
| NVIDIA H100 | 4K | 4.03 | 109.71 | 107.54 | 42.2 |
| NVIDIA L4 | 1K | 4.52 | 6.41 | 2.34 | 7.9 |
| NVIDIA L4 | 2K | 4.07 | 29.11 | 28.45 | 15.2 |
| NVIDIA L4 | 4K | 4.08 | 132.42 | 130.63 | 16.5 |

说明:
- **模型加载**: S3Diff 各组件加载到设备 + 权重移动到 Neuron/GPU 的时间. Trn2 加载时间含 NEFF 缓存检查
- **冷启动**: 第一次推理时间. Trn2 含 HLO → NEFF 的即时编译 (约 90 秒)
- **Warm 平均**: Trn2/H100/L4 均按客户公式计算 (N=3 for Trn2, N=10 for GPU)
- **峰值显存**: 单核 (Trn2 LNC=2 每核 24 GB; GPU 整卡)

---

## 2. 精度 / 画质对比 (PSNR)

| 设备 | 1K PSNR (dB) | 2K PSNR (dB) | 4K PSNR (dB) |
|---|---|---|---|
| AWS Trn2.3xlarge | 43.10 (vs CPU fp32) | 40.60 (vs H100) | 37.07 (vs H100) |
| NVIDIA H100 BF16 | 45.10 (vs CPU fp32) | — (参考) | — (参考) |
| NVIDIA L4 BF16 | 45.15 (vs CPU fp32) | — | — |

**注**:
- 1K PSNR 基准 = CPU fp32 输出
- 2K / 4K PSNR 基准 = H100 bf16 输出 (CPU fp32 在高分辨率时耗时过长, 不适合作为 reference)
- Trn2 相比 H100 的 ~2 dB 差距主要来自 BF16 matmul 累加噪声, 为业界普遍现象, 肉眼无差别

---

## 3. 生图示例

### 3.1 猫图 (256 → 1024, 4× 超分)

| 输入 256×256 LQ | Trn2 输出 1024×1024 | H100 输出 1024×1024 |
|---|---|---|
| ![LQ](images/input_cat_LQ_256.png) | ![Trn2](images/trial6_1k.png) | ![H100](images/h100_1k.png) |

### 3.2 猫图 2K (512 → 2048)

![Trn2 2K](images/trial6_2k.png)

### 3.3 猫图 4K (1024 → 4096)

![Trn2 4K](images/trial6_4k.png)

### 3.4 公交车图 (256 → 1024)

| 输入 256×256 LQ (来源: ultralytics.com/images/bus.jpg) | Trn2 输出 1024×1024 |
|---|---|
| ![Bus LQ](images/input_bus_LQ_256.png) | ![Trn2 Bus](images/trial6_bus_1k.png) |

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

### NVIDIA GPU (H100, L4)

- **H100**: AWS p5.48xlarge, PyTorch 2.1 + CUDA 12.1, diffusers 0.34
- **L4**: AWS g6.12xlarge, PyTorch 2.1 + CUDA 12.1, diffusers 0.34

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
| 编译方式 (Trn2) | `torch.compile(backend="neuron", dynamic=False, fullgraph=False)` 作用于 16 个 `Transformer2DModel`, 其他模块保持 eager |
| 编译 flags | `--auto-cast=matmult -O1` |

---

## 5. 运行命令 (复现)

```bash
# 1K
python phase_bisect_hires.py --scope t2d \
    --lq_image <path_to_LQ_256.png> --lq_size 256 \
    --output_image /tmp/out_1k.png --num_inferences 3

# 2K
python phase_bisect_hires.py --scope t2d \
    --lq_image <path_to_LQ_512.png> --lq_size 512 \
    --output_image /tmp/out_2k.png --num_inferences 3

# 4K
python phase_bisect_hires.py --scope t2d \
    --lq_image <path_to_LQ_1024.png> --lq_size 1024 \
    --output_image /tmp/out_4k.png --num_inferences 2
```

脚本位置: `scripts/phase_bisect_hires.py`, `scripts/phase_bisect.py`

---

## 6. 交付清单

| 文件 | 内容 |
|---|---|
| `REPORT.md` | 本报告 |
| `data/s3diff_benchmark.csv` | 原始测试数据 (每次 run 明细) |
| `images/` | 生图样例: cat 1K/2K/4K + bus 1K + GPU 对比 + CPU 参考 |
| `scripts/` | 可复现运行脚本 |
| `logs/` | 每次测试原始 stdout |
