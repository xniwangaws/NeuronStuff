# OneFormer Neuron port status

Last updated: 2026-09-03 Asia/Singapore

## Outcome

`PASS`: the official OneFormer ADE20K ConvNeXt-XL 640 x 640 checkpoint runs
end to end as a fixed-shape BF16 Neuron pipeline on `trn2.3xlarge`.

## Target

- Model: OneFormer ConvNeXt-XL, ADE20K
- Parameters: 372,007,256
- Batch and input: 1 x 3 x 640 x 640
- Class logits: 1 x 250 x 151
- Mask logits: 1 x 250 x 160 x 160
- Backend: `torch_neuronx.trace`
- Logical NeuronCore configuration: LNC2
- Compiler flags:
  `-O1 --auto-cast=all --auto-cast-type=bf16`

## Final Trn2 validation

- Component count: 78
- Class-logit cosine similarity: 0.99679095
- Mask-logit cosine similarity: 0.98793650
- Semantic pixel agreement: 0.99977297
- Warmups / measured runs: 3 / 10
- Full mean: 4527.947 ms
- Full p50: 4526.376 ms
- Full p90: 4531.472 ms
- Full min / max: 4524.419 / 4533.818 ms

Component means:

- Backbone: 152.582 ms
- Pixel decoder: 4190.425 ms
- Task encoder: 0.224 ms
- Transformer: 175.309 ms

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
- The pixel decoder is split into 23 micro-components.
- The transformer is split into 17 components.
- Integrated native `grid_sample` was rejected after numerical validation.
- Explicit bilinear gather is the current valid sampler.

## Main limitation

The explicit sampler makes the pixel decoder 92.5% of complete latency.
The current result proves functional compatibility and numerical quality but
is not performance optimized. A fused NKI deformable-attention kernel is the
recommended next step.

## Historical result

An earlier 512 x 512 Swin-Tiny target also compiled successfully with exact
semantic agreement. It is not the model used in the ConvNeXt-XL comparison
above.
