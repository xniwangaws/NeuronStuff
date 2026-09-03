# OneFormer ConvNeXt-XL on AWS Neuron

[中文 README](README.zh-CN.md)

This repository contains a fixed-shape AWS Neuron inference port of the
official OneFormer ConvNeXt-XL ADE20K 640 x 640 checkpoint. The model,
checkpoint, resolution, and network structure are unchanged.

## Summary

| Platform and path | Backbone | OneFormer core | Semantic agreement |
| --- | ---: | ---: | ---: |
| NVIDIA L4, PyTorch AMP BF16 | 54.57 ms | 114.16 ms | 99.9558% |
| NVIDIA L4, TensorRT BF16 backbone + PyTorch head | **24.78 ms** | **89.81 ms** | 99.9568% |
| Trn2, Neuron BF16/LNC2 + NKI | 56.48 ms | 144.53 ms | **99.9805%** |

TensorRT accelerates the L4 backbone by **2.20x** and the complete core by
**1.27x** relative to PyTorch AMP BF16. The Trn2 result is **2.28x** slower
than the TensorRT backbone and **1.61x** slower for the complete core.

The L4 TensorRT measurement compiles the complete ConvNeXt-XL backbone; the
OneFormer head remains native PyTorch. It is not a whole-model TensorRT
engine.

## Trn2 result

| Component | Mean latency |
| --- | ---: |
| ConvNeXt-XL backbone | 56.48 ms |
| Pixel decoder | 69.95 ms |
| Task encoder | 0.39 ms |
| Transformer decoder | 23.18 ms |
| Complete pipeline | **144.53 ms** |

The stable 20-run measurement has a 144.48 ms p50, 145.50 ms p90, and a
143.42–145.99 ms range.

The backend is `torch_neuronx.trace`, not NxDI. The tested configuration uses
BF16 auto-casting, LNC2, device-resident direct-HBM chaining, 33 unique
artifacts, and 39 runtime invocations.

ConvNeXt stages 0, 1, and 2 use custom fused NKI blocks covering depthwise
convolution, LayerNorm, pointwise projections, GELU, LayerScale, and the
residual connection. The pixel decoder uses the NKI Library
MSDeformableAttention kernel. A six-layer pixel-decoder stack artifact was
also tested but was slower, so it is not used by the final path.

## Model identity

- Model: OneFormer ConvNeXt-XL, ADE20K
- Parameters: 372,007,256
- Input: 1 x 3 x 640 x 640
- Official metrics: PQ 50.1, AP 36.3, single-scale mIoU 57.4,
  multi-scale + flip mIoU 58.8
- Checkpoint SHA-256:
  `a022437a6cc16fd1485230670f2f7a3ed5e08ef9f08d3f67a42948e5a6a4d7ca`

## Repository layout

- `benchmarks/oneformer_convnext_xl_ade20k_640.json`: machine-readable results
- `STATUS.md`: validation and environment details
- `neuron_port/convnext_nki.py`: stage-0 NKI kernels
- `neuron_port/convnext_stage1_nki.py`: stage-1 NKI kernels
- `neuron_port/convnext_stage2_nki.py`: stage-2 NKI kernels
- `neuron_port/nki_ops.py`: pixel-decoder NKI wrappers
- `scripts/compile_convnext_stage_pipeline.py`: backbone compiler
- `scripts/compile_pixel_decoder_nki_pipeline.py`: pixel-decoder compiler
- `scripts/run_full_oneformer_pipeline.py`: end-to-end benchmark

Compiled NEFF artifacts, model weights, and test inputs are intentionally not
stored in Git.

## Measurement scope

- Static batch-1, 640 x 640 input
- Semantic segmentation path
- Image preprocessing and final semantic postprocessing are excluded
- Results describe this port and software stack, not a general hardware
  comparison
