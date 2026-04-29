# S3Diff Phase 3 — 中文报告

Phase 3 解决了 Phase 1/2 的一个结构问题 (UNet NEFF 绑定特定输入图像), 并尝试修复 tile seam artifact.

## Phase 3 做了什么

### 1. UNet **attribute-routed** trace (Plan B)
   - Phase 1/2 每张 LQ 图的 `de_mod` (LoRA 调制张量) 通过 runtime attribute 写入 UNet module → 被 trace bake 成常量 → **每张图都要重新 compile**.
   - Phase 3 在 UNet wrapper forward **内部** 做 `module.de_mod = unet_de_mods[i]`, 让 `unet_de_mods` 成为 trace 输入张量 → **同一 NEFF 支持任意 `deg_score`**.
   - **CTX-CHECK 验证通过**: 改变 `unet_de_mods` 后 output 随之变化 (sensitivity 0.172 > 0, 确认数据流入 HLO graph).

### 2. VAE decoder tile seam 缓解
   - Phase 2 baseline (无 overlap, simple stitch): seam luma **73** (row 511 很明显).
   - Phase 3 overlap=16 + gaussian blend: seam luma **39** (-46%).
   - overlap=32 + gaussian blend: seam luma **38** (饱和).
   - **结论**: Neuron 每 tile 数值漂移 (cosine 0.9955) 在 tile 边界放大, overlap 不能消除只能软化. GPU baseline 同指标 **1-3** (完全不可见).

### 3. 精度根因定位
   - 单模块精度对比 (Neuron vs CPU eager):
     - DEResNet / Text encoder / de_mod MLP: CPU eager (未 trace)
     - VAE encoder (tile 256): max\|diff\| 0.40, mean 0.011 ✅ 合格
     - **UNet Phase 1 (tile 96, baked)**: max\|diff\| 1.81, mean 0.063, cosine 0.9955 ⚠️
     - **UNet Phase 3 (lat 128 full, attr-routed)**: max\|diff\| **21.82**, mean 0.293 ❌ **12× 更差**
     - VAE decoder (tile 64): **max\|diff\| 0.00000** ✅ bit-exact
   - 主要 drift 来源: **UNet 在 full-latent 128×128 时 attention 序列长度 16384, BF16 softmax 累加误差 O(N²) 放大**.

## 基准结果 (cat LQ 256×256, BF16, N=10, seed 123)

### 延迟对比 (steady-state)

| 设备 | 分辨率 | Load | Cold | **Steady** | P50 | P95 | Peak mem | Seam luma | PSNR vs CPU |
|------|--------|------|------|------------|-----|-----|----------|-----------|-------------|
| H100 80GB | 1K | 4.80s | 9.64s | **1.26s** | 1.26 | 1.26 | 9.0 GB | 3.4 | **45.10 dB** |
| H100 80GB | 2K | 4.02s | 24.67s | 24.26s | 24.19 | 24.54 | 15.7 GB | 2.3 | — |
| H100 80GB | 4K | 4.03s | 109.71s | 107.54s | 107.33 | 108.67 | 42.2 GB | 1.6 | — |
| L4 24GB | 1K | 4.52s | 6.41s | 2.34s | 2.34 | 2.35 | 7.9 GB | 3.3 | 45.15 dB |
| L4 24GB | 2K | 4.07s | 29.11s | 28.45s | 28.47 | 28.64 | 15.2 GB | 2.3 | — |
| L4 24GB | 4K | 4.08s | 132.42s | 130.63s | 130.63 | 130.97 | 16.5 GB | 1.6 | — |
| **Trainium2 (v3)** | 1K | 55.90s | 24.94s | **24.91s** | 24.91 | 24.92 | — | **39.0** | **24.55 dB** |
| **Trainium2 (v3)** | 2K | 56.01s | 82.71s | **82.67s** | 82.68 | 82.68 | — | 33.3 | — |
| **Trainium2 (v3)** | 4K | 55.71s | 360.11s | **360.10s** | 360.11 | 360.13 | — | 18.6 | — |

**`Seam luma`** = 逐行 luma 差分峰值, 衡量 tile 边界可见度. GPU 值 1-3 人眼不可见, Trn2 18-39 仍然可见.

### 相对性能 (Trn2 vs GPU)

| 分辨率 | Trn2/H100 | Trn2/L4 |
|--------|-----------|---------|
| 1K | 19.8× 慢 | 10.6× 慢 |
| 2K | 3.4× 慢 | 2.9× 慢 |
| 4K | 3.3× 慢 | 2.8× 慢 |

1K 是离群点, 因为 VAE decoder tile overhead 占主导 (9 tiles × 2.5s ≈ 22s). 2K/4K 基本趋近于 Trainium2 硬件比 H100 大约慢 3× (合理的加速器对比).

## 精度对比 (1K cat)

| 对比 | PSNR | 说明 |
|------|------|------|
| H100 vs L4 (GPU-to-GPU) | **45.13 dB** | 基本一致, 证明 GPU 实现稳定 |
| Neuron v3 vs CPU eager | 24.55 dB | 有 tile seam artifact |
| Neuron v3 vs H100 | 24.56 dB | 同上 |

**内容保留验证** (mean RGB, 必须与输入接近): cat LQ (175.9, 148.7, 120.7), Trn2 输出 (174.9, 147.8, 119.9) — **像素级均值差异 < 1**, 内容正确, 只是有 tile 条纹 artifact.

## 附加测试: bus 图像 (2026-04-29 新增)

使用 `https://ultralytics.com/images/bus.jpg` 裁剪到 256×256 后作为 LQ.

| 指标 | 值 |
|------|-----|
| Input | `bus_LQ_256_input.png` (中心裁剪后 resize) |
| CPU eager 参考 time | 23.8s |
| Trn2 v3 steady (tile-96 NEFF) | 27.1s |
| PSNR Trn2 vs CPU | **16.76 dB** ⚠️ (比 cat 24.55 dB 还差) |
| Seam luma | 54 |
| Mean RGB (CPU) | (109.6, 109.7, 115.1) |
| Mean RGB (Trn2) | (110.1, 110.0, 115.3) ✅ 内容一致 |

### 为什么 bus 比 cat PSNR 更差?

1. **Bus 图像细节更丰富** (bus 窗户、文字、路面线条), 边界敏感度更高
2. **Bus 测试用了 tile-96 UNet NEFF (attribute-routed)**, 而 cat 用了 latent-128 single NEFF. 虽然 tile-96 per-tile drift 只有 6.05 (vs 128 的 21.82), 但 1K (latent 128) 必须切成 2 tiles, tile 边界 seam 抵消了精度改善 → **反而 PSNR 更差**.
3. 结论: **single-tile 128 NEFF 对 1K 更优**, tile-96 NEFF 预期用于 2K/4K (latent 256/512 必须切片时).

## 关键文件

```
s3diff-benchmark/phase3/
├── README.md                   # 英文详细文档 (含根因分析, 优化路线)
├── README_zh.md                # 本文件 (中文摘要)
├── scripts/
│   ├── neuron_unet_trace_v3b.py  # Plan B 属性路由 UNet trace
│   ├── neuron_e2e_v3.py          # e2e pipeline (UNet + VAE 都 traced)
│   └── patch_s3diff_v3.sh        # S3Diff 源码补丁 (.cuda→.cpu, torchvision shim)
├── images/
│   ├── cat_LQ_256_input.png, CPU_eager_cat_1K.png
│   ├── H100_cat_{1K,2K,4K}.png, L4_cat_{1K,2K,4K}.png
│   ├── Trn2v3_cat_{1K,2K,4K}.png
│   ├── Trn2v3_cat_1K_{stitch_no_overlap,overlap16,overlap32}.png  # 消融
│   └── bus_LQ_256_input.png, CPU_eager_bus_1K.png, Trn2v3_bus_1K.png
└── *.json                      # 10-run benchmark raw data
```

## 已知优化路线 (未实现)

基于 AWS [TEXT_TO_VIDEO_MODEL_PORTING.md](https://github.com/aws-neuron/neuronx-distributed-inference) 内部知识库:

1. **Block-by-block NEFF 切片** (预期 2× 加速 + 消除 seam):
   - 将 VAE decoder 拆成 6 个独立 NEFF (conv_in, mid_resnets, mid_attn, up_resnets×N, norm_out)
   - 每个 block 小到能在 latent 128 下 compile, 消除 tile
   - 约 1-2 天工作量

2. **NKI Conv2d kernel** (预期 2-3× 加速):
   - AWS 官方 [`conv2d_scaling_min` 模板](https://github.com/aws-neuron/neuronx-distributed-inference) 作为参考
   - 替换 VAE Conv2d 瓶颈 block
   - 约 3-5 天工作量

3. **`--auto-cast=matmul --auto-cast-type=bf16`** (optimum-neuron 官方 SD 设置):
   - Phase 3 用 `--auto-cast=none`, 保持数据类型严格一致 (但牺牲 ~46% 性能)
   - 切换到 `matmul` 可能提速 + 精度无明显损失 (HF optimum-neuron 默认)
   - 30 min 验证

4. **NKI flash attention** (解决 UNet BF16 累加漂移):
   - AWS SDXL notebook 使用 `neuronxcc.nki._private_kernels.attention.attention_isa_kernel`
   - 替换 diffusers 默认 attention 为 flash 实现, 修复 BF16 softmax 16K token drift
   - 1-2 天工作量

## 成本

- Phase 3 trn2.48xlarge capacity block: $474 (13h × 2 次)
- Phase 3 H100 + L4 re-benchmark with cat: ~$12
- 累计 (含 Phase 1/2): **~$1163**
