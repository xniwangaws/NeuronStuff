# S3Diff 超分辨率基准测试 — 中文版

## 客户验证要点

- **任务类型**: 图像超分辨率 (x4 upscale), 使用 [S3Diff](https://github.com/ArcticHare105/S3Diff) (单步退化感知 SR, 基于 SD-Turbo UNet)
- **精度对齐**: 仅与 **GPU (H100/L4) 输出对齐**, 不与 ground truth 对齐
- **生成结果**: 图像保存在 `phase3/images/`, 由人工判断差异是否可接受 (不做像素级对齐)
- **模型版本**: S3Diff 官方 HF 权重 `zhangap/S3Diff` + `stabilityai/sd-turbo`, 基本验证

## 基准参数

| 参数 | 值 |
|------|-----|
| 模型 | [`zhangap/S3Diff`](https://huggingface.co/zhangap/S3Diff) LoRA delta + `stabilityai/sd-turbo` base |
| 任务 | x4 super-resolution (LQ→SR) |
| 精度 | BF16 (所有设备) |
| Guidance scale | 1.07 (S3Diff 默认) |
| Inference steps | 1 (S3Diff 是单步 SR) |
| Seed | 123 |
| Warmup | 1 (cold start 单独记录) |
| Timed runs | 10 |
| 测试图片 | `cat_LQ_256.png` (256×256 橘猫) + `bus_LQ_256.png` (来自 ultralytics) |

## 实例规格

| 实例 | 加速器 | 显存/HBM | 按需价 $/hr |
|------|--------|---------|-------------|
| `trn2.48xlarge` | 16x Trainium2 (本基准只用 1 core) | 96GB HBM per chip | $36 (整机) / ~$2.3 (1 core) |
| `trn2.3xlarge` | 1x Trainium2 | 96GB HBM | ~$4.5 |
| `p5.4xlarge` | 1x H100 80GB | 80GB HBM3 | ~$10.5 |
| `g6.4xlarge` | 1x L4 24GB | 24GB GDDR6 | $1.32 |

## 性能对比 (BF16, N=10, cat_LQ_256.png)

### 1K / 2K / 4K 延迟对比

| 设备 | 1K 输出 | 2K 输出 | 4K 输出 | 加速比 (vs H100) |
|------|--------|---------|---------|------------------|
| **H100 80GB** (p5.4xlarge) | **1.26s** | 24.26s | 107.54s | 1x (基准) |
| **L4 24GB** (g6.4xlarge) | 2.34s | 28.45s | 130.63s | 0.5-0.8x |
| **Trainium2** (Phase 3) | 24.91s | 82.67s | 360.10s | 0.04-0.3x |

> Trainium2 在 S3Diff 上**比 GPU 慢**, 主要 bottleneck 是 VAE decoder 在 Conv2d 密集工作负载下硬件利用率低.

### 精度对比 (PSNR vs CPU eager, 越高越好)

| 设备 | 1K PSNR | 主要 artifact |
|------|---------|---------------|
| H100 | 45.10 dB | 无可见 |
| L4 | 45.15 dB | 无可见 |
| **Trainium2 (Phase 3 v3)** | **24.55 dB** | tile 边界条纹 |

**GPU 之间 PSNR 45 dB (基本一致)**. Neuron 与 GPU 差距 ~20 dB, 主要来自 **VAE decoder tile 边界**和 **UNet BF16 累加漂移** (详见 `phase3/README.md`).

## 生成图片 (供人工判断)

所有测试图片在 `phase3/images/` 目录:

| 文件 | 设备 | 分辨率 | 说明 |
|------|------|-------|------|
| `cat_LQ_256_input.png` | — | 256×256 | 输入 LQ (橘猫) |
| `CPU_eager_cat_1K.png` | CPU eager | 1024×1024 | 官方 S3Diff 参考输出 |
| `H100_cat_{1K,2K,4K}.png` | H100 | 1K/2K/4K | GPU 参考 |
| `L4_cat_{1K,2K,4K}.png` | L4 | 1K/2K/4K | GPU 参考 |
| `Trn2v3_cat_{1K,2K,4K}.png` | Trainium2 | 1K/2K/4K | Neuron Phase 3 输出 |
| `Trn2v3_cat_1K_stitch_no_overlap.png` | Trainium2 | 1024×1024 | 无 tile overlap 版 (seam 最明显) |
| `Trn2v3_cat_1K_overlap16.png` | Trainium2 | 1024×1024 | tile overlap 16 (默认) |
| `Trn2v3_cat_1K_overlap32.png` | Trainium2 | 1024×1024 | tile overlap 32 (饱和) |

## 已知问题

1. **Trainium2 VAE decoder tile 边界条纹**: 1K 输出在 `row=511` 附近有水平色条. 根因是 Conv2d 在 Neuron Tensor Engine 利用率低 + 5M 指令限制导致必须 tile. 详见 `phase3/README.md` §"Why VAE decoder is slow".

2. **UNet BF16 累加漂移**: 在 full latent 128×128 下 attention 序列长度 16384, BF16 softmax 累加产生 `max|diff|≈22` per-tile. 已尝试: `--auto-cast=matmul` (未验证), 重编译更大 tile (失败), NKI flash attention (未集成).

3. **Phase 3 为缓解 seam 用 tile overlap + gaussian blend**: overlap 16 将 seam 从 73 → 39, overlap 32 达到饱和 ~38. 根本修复需要 NKI 或 block-by-block NEFF (未实现).

## 如何复现

### Neuron (trn2.48xlarge 或 trn2.3xlarge)

```bash
# 1. 启动 SDK 2.29 DLAMI (Ubuntu 24.04, 2026-04-10 build)
# 2. Clone repo + 安装依赖
git clone https://github.com/ArcticHare105/S3Diff ~/s3diff/repo
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
pip install diffusers==0.34.0 'peft>=0.15.0' accelerate einops omegaconf scipy

# 3. 应用 S3Diff 补丁 (.cuda()→.cpu(), torchvision shim)
bash phase3/scripts/patch_s3diff_v3.sh

# 4. Trace UNet (attribute-routed, ~40min compile)
python phase3/scripts/neuron_unet_trace_v3b.py \
  --latent 128 \
  --out neuron_out/unet_1K_v3b.pt2 \
  --lq_image cat_LQ_256.png

# 5. Trace VAE encoder + decoder (~5min + 40min)
python scripts/trace_vae.py --component encoder --enc_tile 256
python scripts/trace_vae.py --component decoder --dec_tile 64

# 6. 运行 e2e benchmark
python phase3/scripts/neuron_e2e_v3.py \
  --resolution 1K --num_runs 10 \
  --input_image cat_LQ_256.png
```

### GPU (H100 / L4)

```bash
# 1. 启动 DLAMI PyTorch 2.7 (Ubuntu 22.04)
git clone https://github.com/ArcticHare105/S3Diff ~/s3diff/repo
python3.10 -m venv venv && source venv/bin/activate
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install 'numpy<2' pyyaml einops timm scipy transformers==4.35.2 \
  opencv-python==4.6.0.66 torchmetrics lpips pyiqa omegaconf accelerate \
  diffusers==0.25.1 peft==0.10.0 'setuptools<81'

# 2. 运行官方 CLI (BF16)
accelerate launch --mixed_precision bf16 src/inference_s3diff.py ...
# 或直接用 bench.py (在 Phase 1/2 scripts/)
python bench.py --device cuda --resolution 1K --num_runs 10 \
  --input_image cat_LQ_256.png --dtype bf16
```

## 软件版本

| 组件 | Neuron | GPU |
|------|--------|-----|
| PyTorch | 2.9.1 (torch-neuronx 2.9.0.2.13) | 2.1.0+cu121 |
| Neuron SDK | 2.29.0 | — |
| diffusers | 0.34.0 | 0.25.1 (官方版本) |
| peft | 0.19.1 | 0.10.0 |
| NxDI | 0.9.0 | — |
| DLAMI | `Deep Learning AMI Neuron (Ubuntu 24.04) 20260410` | PyTorch 2.7 Ubuntu 22.04 |

## Phase 发展时间线

- **Phase 1** (2026-04-24→27): GPU baseline + Neuron UNet 初版 (baked de_mod, tile 96)
- **Phase 2** (2026-04-27): 追加 VAE encoder + decoder trace, 1K 加速到 14.7s
- **Phase 3** (2026-04-29): UNet attribute routing (image-agnostic NEFF), 同时发现 tile seam 修复瓶颈

详细文档 (英文) 见 `README.md` 和 `phase3/README.md`.
