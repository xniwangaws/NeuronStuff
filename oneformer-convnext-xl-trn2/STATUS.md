# OneFormer Neuron port status

Last updated: 2026-09-03 Asia/Singapore

## Outcome

`PASS`: the official OneFormer ADE20K ConvNeXt-XL 640 x 640 checkpoint runs
end to end as a fixed-shape BF16 Neuron pipeline with fused NKI
multi-scale deformable attention on `trn2.3xlarge`.

## Target

- Model: OneFormer ConvNeXt-XL, ADE20K
- Parameters: 372,007,256
- Batch and input: 1 x 3 x 640 x 640
- Class logits: 1 x 250 x 151
- Mask logits: 1 x 250 x 160 x 160
- Backend: `torch_neuronx.trace` plus NKI Library MSDeformableAttention
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
- Full mean: 670.600 ms
- Full p50: 670.401 ms
- Full p90: 671.146 ms
- Full min / max: 669.965 / 671.535 ms

Component means:

- Backbone: 151.369 ms
- Pixel decoder: 326.066 ms
- Task encoder: 0.348 ms
- Transformer: 176.207 ms

Compared with the validated explicit-gather baseline:

- Pixel decoder: 4190.425 -> 326.066 ms, 12.85x faster
- Complete pipeline: 4527.947 -> 670.600 ms, 6.75x faster
- Unique artifacts: 78 -> 63
- Runtime invocations: 99 -> 69

## GPU comparison

The NVIDIA L4 reference used the original Detectron2 implementation.

- Strict FP32 native core: 215.964 ms
- Strict FP32 TensorRT backbone + native head: 170.738 ms
- PyTorch AMP BF16 core: 114.158 ms
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

The six fused encoder layers now run in roughly 12.5-12.7 ms each and
75.86 ms combined. Component microbenchmarks put the input projection at
7.44 ms and the final FPN output at 255.18 ms. The FPN upsample and
convolution path is therefore the next decoder target. The complete model
also retains 69 host-dispatched calls, mainly from the fine-grained backbone
and transformer pipelines.

## Historical result

An earlier 512 x 512 Swin-Tiny target also compiled successfully with exact
semantic agreement. It is not the model used in the ConvNeXt-XL comparison
above.
