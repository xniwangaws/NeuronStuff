# S3Diff 超分辨率 Benchmark — AWS Trainium2 vs NVIDIA L4

**模型**: S3Diff (ECCV 2024) — 一步扩散 4× 超分辨率 (SD-Turbo + LoRA)
**测试**: bus 图像, 4× 超分:
- **0.5K**: 输入 128×128 → 输出 512×512 (single tile)
- **1K**: 输入 256×256 → 输出 1024×1024 (9 tiles)
**精度**: BF16

> Trn2 使用 [AWS Neuron PR #149](https://github.com/aws-neuron/neuronx-distributed-inference/pull/149) (Jim Burtoft) 的 `torch_neuronx.trace()` + 固定 512 像素 tile + Gaussian blend 方案。

---

## ⭐ 核心结果

### 主表 (warm mean / 每张图成本)

| 设备 | 128→512 | 256→1024 | Pass |
|---|---|---|---|
| **Neuron trn2.3xl PR 149 (BF16, 整机)** | 0.545s / $0.00034 | 4.81s / $0.00299 | 5/5 |
| **Neuron trn2.3xl PR 149 (BF16, 1/4 核算)** | 0.545s / **$0.00008** | 4.81s / **$0.00075** | 5/5 |
| L4 24GB BF16 | 0.914s / $0.00034 | 2.34s / $0.00086 | 5/5 |

### 成本效率 (L4 = 100% 基准)

| 设备 | 128→512 成本效率 | 256→1024 成本效率 |
|---|---|---|
| **Neuron trn2.3xl 整机** | 100% | 29% |
| **Neuron trn2.3xl (1/4 核算)** | **397%** ⭐ | **115%** ⭐ |
| **L4 24GB** | **100%** (ref) | **100%** (ref) |

> 效率 = L4 每张图成本 / 对应实例每张图成本. 值越高越省钱.

### 关键发现

- **128→512 (single tile)**: Trn2 比 L4 **快 1.68×** (0.545s vs 0.914s)，整机价格也比 L4 便宜（$0.00034 持平），1/4 核算便宜 **4×**
- **256→1024 (9 tiles)**: Trn2 比 L4 慢 2× (4.81s vs 2.34s)，因为固定 512 tile 在 1K 输出需要 9 个 tile，GPU 单 pass 完成；整机贵 3.5×，但 1/4 核算依然便宜 **15%**
- **拐点**: Trn2 在小分辨率 (128→512 单 tile) 上速度+成本双优；进入多 tile 区间后绝对速度落后，需靠 1/4 并发 (4 路 throughput) 才能 cost-win
- **Tile 数量**: 128→512 = 1 tile (HR=512 ≤ 512 tile_size), 256→1024 = 9 tiles (HR=1024, stride=384, 3×3 grid)
- **画质**: Trn2 1K PSNR 43.10 dB vs CPU fp32 (BF16 matmul 累加噪声 ~2 dB，肉眼无差别)

### 原始数据

价格计算: `$/图 = $/hr × warm_s / 3600`
- trn2.3xlarge 整机 = $2.235/hr (capacity block); 1/4 核算 = $0.559/hr (4 路并发)
- g6.4xlarge (L4) = $1.323/hr (on-demand)

---

## 1. 详细测试参数

| 参数 | 值 |
|---|---|
| Batch size | 1 |
| Diffusion steps | 1 (S3Diff 一步模型) |
| Scheduler | DDPMScheduler |
| Seed | 123 (固定, 可跨栈对比) |
| 正向 prompt | "high quality, highly detailed, clean" |
| 负向 prompt | "blurry, dotted, noise, raster lines, unclear, lowres, over-smoothed" |
| Guidance scale | 1.07 |
| Tile (Trn2 PR 149) | fixed pixel tile=512, overlap=128, Gaussian blending |
| 编译方式 (Trn2) | `torch_neuronx.trace()` 对 5 个组件分别编译 (DEResNet / Text Enc / VAE Enc / UNet / VAE Dec) |
| 编译 flags (LoRA 组件) | `--model-type=unet-inference -O1` (matmult 下 LoRA einsum 会 NaN) |
| 编译总耗时 | ~1357s (~22.6 分钟, 一次性) |
| 测试图像 | bus (来源 `https://ultralytics.com/images/bus.jpg`) |

---

## 2. 生图示例 — 公交车

测试图来源: `https://ultralytics.com/images/bus.jpg`

### 2.1 输入 LQ

256×256 LQ 输入 (用于 1K 输出测试):

![Bus LQ 256](customer_report/images/input_bus_LQ_256.png)

### 2.2 Trn2 PR 149 1K 输出 (1024×1024, 4× 超分)

![Trn2 Bus 1K](customer_report/images/trial6_bus_1k.png)

---

## 3. 硬件 / 软件配置

### AWS Trn2.3xlarge

| 项目 | 值 |
|---|---|
| 实例类型 | trn2.3xlarge (LNC=2, 4 logical NeuronCores, 96 GB HBM) |
| vCPU / 内存 | 12 vCPU / 96 GB |
| AMI | Deep Learning AMI Neuron (Ubuntu 24.04) 20260502 |
| Neuron SDK | 2.29 |
| Python | 3.12 |
| PyTorch | 2.9.1 |
| torch-neuronx | 2.9.0.2.13.26312+8e870898 |
| neuronx-cc | 2.0.243879.0a0+866424ce |
| diffusers | 0.34.0 |
| peft | 0.19.1 |
| transformers | 4.57.3 |

### NVIDIA L4

- AWS g6.4xlarge (1× L4 24GB)
- PyTorch 2.10 + CUDA 13.0
- diffusers 0.34

---

## 4. 运行命令 (复现)

### Trn2 (PR 149 方案)

```bash
# 代码来源: https://github.com/aws-neuron/neuronx-distributed-inference/pull/149
# 文件: contrib/models/S3Diff/src/{modeling_s3diff.py, generate_s3diff.py}

source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# 0.5K (128 → 512, single tile)
python generate_s3diff.py \
    --input_image <path_to_LQ_128.png> \
    --output_image /tmp/out_512.png \
    --compile_dir /tmp/s3diff_pr149_compiled \
    --num_images 5 --warmup_rounds 2 \
    --tile_size 512 --tile_overlap 128

# 1K (256 → 1024, 9 tiles)
python generate_s3diff.py \
    --input_image <path_to_LQ_256.png> \
    --output_image /tmp/out_1k.png \
    --compile_dir /tmp/s3diff_pr149_compiled \
    --num_images 5 --warmup_rounds 2 \
    --tile_size 512 --tile_overlap 128
```

### L4 (原生 diffusers)

S3Diff 仓库默认 `inference_s3diff.py` + `accelerate launch --mixed_precision=bf16`.

---

## 5. 目录结构

| 路径 | 内容 |
|---|---|
| `README.md` | 中文报告 (本文件) |
| `README_EN.md` | 英文报告 |
| `customer_report/data/s3diff_benchmark.csv` | 原始测试数据 |
| `customer_report/images/` | bus LQ 输入 + 输出图 |
| `backup/jim_port_attempt/` | Jim 早期 notebook 移植版 (`--auto-cast=matmult` 产生 NaN, 已被 PR 149 最新版取代) |
| `backup/phase3/`, `backup/phase_e/`, `backup/phase_r/`, `backup/phase_b/` | 早期 Trial 6 等方案备份 |
