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

- Backend：`torch_neuronx.trace`，不是 NxDI
- 精度：BF16 auto-cast
- Neuron 配置：LNC2
- 中间张量：device-resident、direct HBM chaining
- ConvNeXt Stage 0/1/2：自定义融合 NKI block
- Pixel Decoder：NKI Library MSDeformableAttention
- 组件数：33；运行时调用数：39

ConvNeXt 的自定义 NKI block 融合
DWConv → LayerNorm → PWConv → GELU → PWConv → LayerScale → Residual。
Pixel Decoder 的六层 stack 合并实验反而增加延迟，因此最终版本仍使用六个独立
encoder layer。

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
