# OneFormer ConvNeXt-XL：L4 TensorRT 与 Trainium2

[English README](README.md)

这是 OneFormer ConvNeXt-XL ADE20K 640 × 640、batch 1 的固定形状推理实现。
模型、checkpoint、输入分辨率和网络结构均未修改。

## Summary

| 平台和执行路径 | Backbone | 完整 OneFormer core | 语义像素一致率 |
| --- | ---: | ---: | ---: |
| NVIDIA L4，PyTorch AMP BF16 | 54.57 ms | 114.16 ms | 99.9558% |
| NVIDIA L4，TensorRT BF16 Backbone + PyTorch head | **24.78 ms** | **89.81 ms** | 99.9568% |
| Trn2，Neuron BF16/LNC2 + NKI | 56.48 ms | 144.53 ms | **99.9805%** |

关键结论：

- TensorRT 相对 L4 PyTorch AMP，Backbone 加速 **2.20×**，完整 core
  加速 **1.27×**。
- Trn2 Backbone 比 L4 TensorRT 慢 **2.28×**，完整 core 慢 **1.61×**。
- Trn2 Backbone 与 L4 PyTorch AMP 接近：56.48 ms 对 54.57 ms；主要差距在
  Pixel Decoder 和 Transformer head。
- L4 的 89.81 ms 是 TensorRT Backbone + PyTorch head，不是全模型
  TensorRT engine。

## Trn2 延迟分解

| 组件 | 平均延迟 |
| --- | ---: |
| ConvNeXt-XL Backbone | 56.48 ms |
| Pixel Decoder | 69.95 ms |
| Task Encoder | 0.39 ms |
| Transformer Decoder | 23.18 ms |
| 完整模型 | **144.53 ms** |

完整模型 20 次稳定测试：p50 144.48 ms，p90 145.50 ms，范围
143.42–145.99 ms。

## 实现

最终路径使用 `torch_neuronx.trace`，不是 NxDI；编译配置为 BF16
auto-cast、`-O1` 和 LNC2。

### 最终启用的优化

| 优化 | 实现 |
| --- | --- |
| 静态形状专用化 | 固定 batch 1、640 × 640；shape-only 位置编码在计时路径外生成 |
| Channels-first Backbone | 保持 NCHW，使用等价的 1 × 1 Conv 替代 pointwise Linear，减少 layout 转换 |
| Backbone 组件合并 | chunk size 9，将 Backbone 从 37 个组件降到 7 个 |
| ConvNeXt NKI megakernel | Stage 0/1/2 按固定 C/H/W 专用化，融合 DWConv → LayerNorm → PWConv → GELU → PWConv → LayerScale → Residual |
| NKI 内存与精度优化 | 通道分块、SBUF 复用、PSUM FP32 累加；BF16 激活和矩阵乘，FP32 归一化统计与关键接口 |
| 固定 resize | 使用精确的 2× bilinear upsample 和偶数倍 downsample，替代通用插值图 |
| Pixel Decoder 单层融合 | 每层包含 value/offset/weight projection、NKI MSDA、output projection、residual/LN 和 FFN |
| MSDA 安全采样 | 将零贡献的远越界采样点清零并移动到安全地址，保持 zero-padding 语义 |
| Device-resident 执行 | 33 个组件放在同一 Neuron device，中间 tensor 通过 direct HBM chaining 传递 |

最终共有 33 个唯一组件、39 次运行时调用。固定 resize 和组件合并也是最终
性能的重要组成部分，并非只使用了 ConvNeXt NKI 与 MSDA。

### 测试后未采用

- Pixel Decoder 六层 stack：72.39 ms，慢于独立六层的 69.95 ms。
- Stage-0 纯 BF16 输出：局部 kernel 更快，但增加数值偏差且不兼容现有
  FP32 stage 接口。
- Native `grid_sample`：能够编译，但数值验证未通过。

## 结果文件

- 机器可读结果：`benchmarks/oneformer_convnext_xl_ade20k_640.json`
- 当前状态：`STATUS.md`
- NKI Backbone：`neuron_port/convnext_nki.py`、
  `neuron_port/convnext_stage1_nki.py`、
  `neuron_port/convnext_stage2_nki.py`
- 编译入口：`scripts/compile_convnext_stage_pipeline.py`
- 完整推理：`scripts/run_full_oneformer_pipeline.py`

编译后的 NEFF、模型权重和测试输入体积较大，不存入 Git。

## 测试口径

- 参数量：372,007,256
- 输入：1 × 3 × 640 × 640
- checkpoint SHA-256：
  `a022437a6cc16fd1485230670f2f7a3ed5e08ef9f08d3f67a42948e5a6a4d7ca`
- 延迟不包含图片预处理和最终语义后处理
- 结果只代表当前模型、静态输入和软件栈，不代表通用硬件性能
