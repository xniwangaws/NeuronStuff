# OneFormer Neuron port status

Last updated: 2026-09-03 Asia/Singapore

## Outcome

`PASS`: the unchanged OneFormer ADE20K ConvNeXt-XL 640 x 640 checkpoint runs
end to end as a BF16 Neuron pipeline on `trn2.3xlarge`.

## Configuration

- Backend: `torch_neuronx.trace` + custom NKI + NKI Library MSDA
- Logical NeuronCore configuration: LNC2
- Compiler: `-O1 --auto-cast=all --auto-cast-type=bf16`
- Execution: device-resident direct-HBM chaining
- Unique artifacts / runtime invocations: 33 / 39
- Warmups / measured runs: 3 / 20

Software:

- `torch-neuronx`: `2.9.0.2.15.32035+de43f57c`
- `neuronx-cc`: `2.27.5334.0+f702b353`
- NKI: `0.6.0+31049202112.g85070674`
- Neuron runtime: `2.34.10`
- NKI Library commit: `92d11f63a9a8ec1ade34e6e1a3b8db66ef31307e`

## Final validation

| Measurement | Result |
| --- | ---: |
| Full mean | 144.533 ms |
| Full p50 | 144.476 ms |
| Full p90 | 145.501 ms |
| Full range | 143.416–145.989 ms |
| Class-logit cosine similarity | 0.996972 |
| Mask-logit cosine similarity | 0.988508 |
| Semantic pixel agreement | 0.99980468 |

Component means:

- Backbone: 56.475 ms
- Pixel decoder: 69.953 ms
- Task encoder: 0.386 ms
- Transformer: 23.183 ms

## Implemented path

- Channels-first ConvNeXt pipeline with block chunk size 9
- Fused custom NKI blocks for ConvNeXt stages 0, 1, and 2
- Fixed-shape resize operations
- Six NKI MSDeformableAttention pixel-decoder encoder layers
- Device-resident intermediate tensors
- Reusable transformer mask artifacts

The six-layer pixel-decoder stack artifact was measured at 72.39 ms versus
69.95 ms for the separate-layer path and was rejected.

## GPU comparison

- L4 PyTorch AMP BF16: 54.569 ms backbone, 114.158 ms core
- L4 TensorRT BF16 backbone + PyTorch head: 24.782 ms backbone,
  89.809 ms core
- TensorRT speedup over AMP: 2.20x backbone, 1.27x core
- Trn2 versus L4 TensorRT: 2.28x backbone latency, 1.61x core latency

## Remaining bottleneck

The 69.95 ms pixel decoder is now the largest component. Further
single-LNC kernel fusion is expected to provide incremental rather than
tens-of-milliseconds improvement. A larger same-model reduction would require
explicit cross-LNC sequence/query parallelism, which is not implemented here.
