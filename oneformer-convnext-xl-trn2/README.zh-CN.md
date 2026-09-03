# OneFormer ConvNeXt-XL：L4 TensorRT 与 Trainium2 性能对比

[English README](README.md)

测试模型为 OneFormer ConvNeXt-XL ADE20K，输入固定为 batch 1、
640 × 640，参数量 372M。

## 结论

- L4 使用 TensorRT 编译 Backbone 后，相对 PyTorch AMP BF16，
  Backbone 加速 **2.20×**，完整 OneFormer core 加速 **1.27×**。
- Trn2 相对 L4 TensorRT 混合路径，Backbone 延迟为 **4.22×**，
  完整 OneFormer core 延迟为 **6.16×**。
- 对这个单图、batch 1 的 CNN + deformable-attention workload，
  L4 + TensorRT 明显更适合低延迟推理。除非部署必须使用 Trainium，
  继续优化当前 Trn2 路径的投入产出比不高。

## Summary table

| 平台和执行路径 | Backbone | 完整 OneFormer core | 语义像素一致率 | 对比结论 |
| --- | ---: | ---: | ---: | --- |
| NVIDIA L4，PyTorch AMP BF16 | 54.57 ms | 114.16 ms | 99.9558% | GPU 基线 |
| NVIDIA L4，TensorRT BF16 Backbone + PyTorch head | 24.78 ms | 89.81 ms | 99.9568% | Backbone 2.20×；完整 core 1.27× |
| Trn2，Neuron BF16/LNC2 + NKI MSDA | 104.54 ms | 553.46 ms | 99.9802% | Backbone 延迟 4.22×；完整 core 延迟 6.16× |

## TensorRT 加速

TensorRT 加速以 L4 PyTorch AMP BF16 为基线：

| 测量范围 | PyTorch AMP BF16 | TensorRT BF16 | 加速 | 延迟降低 |
| --- | ---: | ---: | ---: | ---: |
| ConvNeXt-XL Backbone | 54.57 ms | 24.78 ms | **2.20×** | 54.6% |
| 完整 OneFormer core | 114.16 ms | 89.81 ms | **1.27×** | 21.3% |

完整 core 的 TensorRT 收益较小，是因为 TensorRT 只编译
ConvNeXt-XL Backbone；OneFormer Pixel Decoder 和 Transformer head
仍运行在原生 PyTorch。

## Trn2 延迟分解

| 组件 | 平均延迟 |
| --- | ---: |
| ConvNeXt-XL Backbone | 104.54 ms |
| Pixel Decoder | 310.32 ms |
| Task Encoder | 0.47 ms |
| Transformer Decoder | 142.95 ms |
| 完整模型 | 553.46 ms |

Trn2 使用 `torch_neuronx.trace`、BF16 auto-cast、LNC2，并在 Pixel
Decoder 中使用 NKI MSDeformableAttention。所有编译模块和中间
tensor 常驻同一 Neuron device，避免分段执行时往返 CPU。

## 测试口径

- 相同模型、checkpoint、batch 和输入分辨率
- 延迟不包含图片预处理和最终语义后处理
- “完整 OneFormer core”包含 Backbone 和 OneFormer head
- L4 TensorRT 路径是 TensorRT Backbone + PyTorch head，不是全模型
  TensorRT engine
- 结果仅代表当前模型、静态输入和软件栈，不代表 L4 与 Trainium2 的
  通用硬件性能

机器可读结果：
`benchmarks/oneformer_convnext_xl_ade20k_640.json`
