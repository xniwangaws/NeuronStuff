# OneFormer Neuron port status

Last updated: 2026-09-03 Asia/Singapore

## Outcome

`PASS`: the official OneFormer ADE20K ConvNeXt-XL 640 x 640 checkpoint runs
end to end as a fixed-shape BF16 Neuron pipeline with fused NKI
multi-scale deformable attention and direct-HBM model chaining on
`trn2.3xlarge`.

## Target

- Model: OneFormer ConvNeXt-XL, ADE20K
- Parameters: 372,007,256
- Batch and input: 1 x 3 x 640 x 640
- Class logits: 1 x 250 x 151
- Mask logits: 1 x 250 x 160 x 160
- Backend: `torch_neuronx.trace`, NKI Library MSDeformableAttention, and
  direct-HBM chaining
- Logical NeuronCore configuration: LNC2
- Compiler flags:
  `-O1 --auto-cast=all --auto-cast-type=bf16`

## Final optimized Trn2 validation

- Unique artifact count: 63
- Runtime invocation count: 69
- Class-logit cosine similarity: 0.99723279
- Mask-logit cosine similarity: 0.98865259
- Semantic pixel agreement: 0.99980223
- Warmups / measured runs: 3 / 10
- Full mean: 553.462 ms
- Full p50: 553.322 ms
- Full p90: 554.362 ms
- Full min / max: 552.638 / 554.380 ms

Component means:

- Backbone: 104.540 ms
- Pixel decoder: 310.315 ms
- Task encoder: 0.472 ms
- Transformer: 142.954 ms

Compared with the validated explicit-gather baseline:

- Pixel decoder: 4190.425 -> 310.315 ms, 13.50x faster
- Complete pipeline: 4527.947 -> 553.462 ms, 8.18x faster
- Unique artifacts: 78 -> 63
- Runtime invocations: 99 -> 69

Direct-HBM chaining alone improves the fused NKI pipeline from 670.600 to
553.462 ms, or 1.21x.

## GPU comparison

The NVIDIA L4 reference used the original Detectron2 implementation.

- Strict FP32 native core: 215.964 ms
- Strict FP32 TensorRT backbone + native head: 170.738 ms
- PyTorch AMP BF16 backbone: 54.569 ms
- PyTorch AMP BF16 core: 114.158 ms
- TensorRT BF16 backbone: 24.782 ms
- TensorRT BF16 backbone + native head: 89.809 ms
- TensorRT BF16 semantic pixel agreement: 0.99956787

The TensorRT measurements compile only the full backbone; they do not claim a
fully TensorRT OneFormer graph.

## Port decisions

- The ConvNeXt backbone is split into embeddings and 36 individual blocks.
- The pixel decoder is split into input, six fused encoder layers, and output:
  8 artifacts and 8 runtime calls.
- Each fused encoder layer includes projection, sampling-coordinate
  generation, NKI deformable attention, output projection, residual/LN, and
  FFN operations.
- The transformer uses 17 unique artifacts but 23 runtime calls because mask
  builders are reused.
- Far-outside sampling points with exactly zero bilinear contribution are
  zero-weighted and moved to safe coordinates before the NKI indirect DMA.
- Integrated native `grid_sample` was rejected after numerical validation.
- Explicit bilinear gather remains the validated baseline.

## Main limitation

The final FPN output remains the largest pixel-decoder component. The complete
model also retains 69 sequential invocations from the fine-grained backbone,
pixel-decoder, and transformer pipelines. Even after direct-HBM chaining, the
batch-1 result is substantially slower than the L4 TensorRT path.

## Historical result

An earlier 512 x 512 Swin-Tiny target also compiled successfully with exact
semantic agreement. It is not the model used in the ConvNeXt-XL comparison
above.
