# OneFormer ConvNeXt-XL 在 AWS Trainium2 上的推理

[English README](README.md)

本项目将官方 OneFormer ADE20K ConvNeXt-XL 640 × 640 模型移植到
`trn2.3xlarge`。推理后端为 `torch_neuronx.trace`，Pixel Decoder 的
Multi-Scale Deformable Attention 使用融合 NKI kernel。

## 核心结论

在 batch 1、640 × 640、BF16 条件下：

- Trn2 Backbone 延迟为 L4 TensorRT Backbone 的 **6.11 倍**
  （151.37 ms 对 24.78 ms，多 126.59 ms）。
- Trn2 完整 OneFormer core 延迟为 L4 混合路径的 **7.47 倍**
  （670.60 ms 对 89.81 ms，多 580.79 ms）。
- 这里的 L4 89.81 ms 并不是“完整模型全部 TensorRT”：TensorRT 编译
  ConvNeXt-XL Backbone，OneFormer head 仍运行在原生 PyTorch。
- 引入融合 NKI MSDeformableAttention 后，Trn2 完整模型从
  4527.95 ms 降到 670.60 ms，提升 6.75 倍。

## Summary table

| 平台和执行路径 | Backbone 平均延迟 | 完整 OneFormer core 平均延迟 | 语义像素一致率 | Backbone 相对 L4 TensorRT | 完整 core 相对 L4 混合路径 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVIDIA L4，TensorRT BF16 Backbone + PyTorch head | 24.78 ms | 89.81 ms | 99.9568% | 基准 | 基准 |
| Trn2，Neuron BF16/LNC2 + 融合 NKI MSDA | 151.37 ms | 670.60 ms | 99.9802% | 延迟 6.11× | 延迟 7.47× |

对比使用相同的 OneFormer ConvNeXt-XL ADE20K 模型、batch 1 和
640 × 640 输入。延迟不包括图片预处理和最终语义后处理。

## Trn2 优化前后

| Trn2 路径 | Pixel Decoder | 完整模型 | 唯一编译产物 | 实际运行调用 | 语义像素一致率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 显式 bilinear gather 基线 | 4190.42 ms | 4527.95 ms | 78 | 99 | 99.9773% |
| 融合 NKI MSDA | 326.07 ms | 670.60 ms | 63 | 69 | 99.9802% |
| 改善 | **12.85×** | **6.75×** | 减少 15 个 | 减少 30 次 | 无精度回退 |

旧路径每个 Pixel Decoder encoder layer 都需要 projection、三个
sampler、combine 和 post-processing。三个 sampler 使用通用
`torch.gather` 实现 bilinear sampling，产生大量间接 DMA，并在六层中
重复执行 18 次，是原来 4.19 秒 Decoder 延迟的主要原因。

新路径把每个 encoder layer 的以下操作合并为一个 Neuron 调用：

1. Value、sampling offset 和 attention weight projection
2. Sampling location 计算
3. NKI MSDeformableAttention
4. Output projection
5. Residual、LayerNorm 和 FFN

因此 Pixel Decoder 从 23 个唯一产物、38 次运行调用，减少到
8 个产物、8 次调用。

## Trn2 延迟分解

| 组件 | 唯一产物 | 实际调用 | 平均延迟 |
| --- | ---: | ---: | ---: |
| ConvNeXt-XL Backbone | 37 | 37 | 151.37 ms |
| Pixel Decoder | 8 | 8 | 326.07 ms |
| Task Encoder | 1 | 1 | 0.35 ms |
| Transformer Decoder | 17 | 23 | 176.21 ms |
| 完整模型 | 63 | 69 | 670.60 ms |

融合后的六个 NKI encoder layer 平均每层约 12.64 ms，合计
75.86 ms。当前 Pixel Decoder 的新瓶颈已经转移到最后的 FPN output：

| Pixel Decoder 子组件 | 平均延迟 |
| --- | ---: |
| Input projection | 7.44 ms |
| 六个融合 NKI encoder layer | 75.86 ms |
| FPN upsample + convolution output | 255.18 ms |

下一步应优化或融合 FPN 的 160 × 160 bilinear upsample 和 convolution
路径，而不是继续优化 MSDeformableAttention。

## 模型与精度

- 模型：OneFormer ConvNeXt-XL，ADE20K，640 × 640
- 参数量：372,007,256
- 官方指标：PQ 50.1、AP 36.3、single-scale mIoU 57.4、
  multi-scale + flip mIoU 58.8
- Trn2：BF16 auto-cast，sampling locations 保留 FP32，LNC2
- 编译参数：
  `-O1 --auto-cast=all --auto-cast-type=bf16`
- 完整模型 p50：670.40 ms
- Class-logit cosine similarity：0.99723279
- Mask-logit cosine similarity：0.98865259
- 语义像素一致率：0.99980223

## 软件环境

- `torch-neuronx` 2.9.0.2.15.32035
- `neuronx-cc` 2.27.5334.0
- NKI 0.6
- Neuron Runtime 2.34.10
- NKI Library commit：
  `92d11f63a9a8ec1ade34e6e1a3b8db66ef31307e`

## 关键代码

- `neuron_port/nki_ops.py`：NKI kernel 封装、越界坐标处理和融合层
- `scripts/test_nki_msda_kernel.py`：单层 NKI 正确性及延迟验证
- `scripts/compile_pixel_decoder_nki_pipeline.py`：编译六个融合 encoder layer
- `scripts/run_full_oneformer_pipeline.py`：完整模型验证与稳定 benchmark
- `benchmarks/oneformer_convnext_xl_ade20k_640.json`：机器可读结果

## 运行 NKI Pixel Decoder

以下命令假定 Backbone、旧 Pixel Decoder input/output、Task Encoder 和
Transformer 产物已按照英文 README 完成编译：

```bash
MODEL_DIR=agent_artifacts/data/oneformer_ade20k_convnext_xl_hf
INPUTS=agent_artifacts/data/reference/inputs.pt
CACHE=agent_artifacts/data/hf_cache

export NEURON_LOGICAL_NC_CONFIG=2

python scripts/compile_pixel_decoder_nki_pipeline.py \
  --model-id "$MODEL_DIR" \
  --inputs "$INPUTS" \
  --cache-dir "$CACHE" \
  --existing-pixel-dir agent_artifacts/traces/pixel_decoder_micro_bf16_all \
  --output-dir agent_artifacts/traces/pixel_decoder_nki_bf16_lnc2 \
  --max-layers 6 \
  --lnc 2 \
  --warmup 3 \
  --runs 10 \
  --custom-grid-sample

python scripts/run_full_oneformer_pipeline.py \
  --model-id "$MODEL_DIR" \
  --inputs "$INPUTS" \
  --cache-dir "$CACHE" \
  --backbone-dir agent_artifacts/traces/convnext_stage_bf16_all \
  --pixel-decoder-dir agent_artifacts/traces/pixel_decoder_nki_bf16_lnc2 \
  --pixel-decoder-backend nki \
  --remaining-dir agent_artifacts/traces/oneformer_remaining_bf16_all \
  --transformer-dir agent_artifacts/traces/transformer_pipeline_bf16_all \
  --output agent_artifacts/results/trn2_full_oneformer_nki_bf16_lnc2.json \
  --warmup 3 \
  --runs 10 \
  --custom-grid-sample
```

## 限制

- 目前只验证静态 batch 1、RGB 640 × 640 输入。
- 目前只验证 semantic segmentation 推理。
- NKI MSDeformableAttention API 仍为 experimental。
- Shape-derived positional embeddings 在计时路径之外生成。
- 编译产物和模型权重不存储在 Git 仓库中。
- 这些结果描述当前移植和软件栈，不代表 Trainium2 与 L4 的通用性能结论。
