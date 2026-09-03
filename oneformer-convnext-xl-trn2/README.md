# OneFormer ConvNeXt-XL on AWS Neuron

This workspace contains a functional AWS Neuron port of the official
OneFormer ADE20K ConvNeXt-XL 640 x 640 checkpoint. The target is static
batch-1 semantic-segmentation inference on `trn2.3xlarge`.

The Neuron backend is `torch_neuronx.trace`, not NxDI. Compilation uses
BF16 auto-casting and LNC2:

```text
--model-type=<component-type> -O1 --auto-cast=all --auto-cast-type=bf16
```

## Result

The complete model runs successfully with an NKI-fused pixel decoder. The
optimized path uses 63 unique Neuron artifacts and makes 69 runtime
invocations. The final stable benchmark used three warmups and ten measured
runs.

| Platform and path | Backbone | OneFormer core | Semantic agreement |
| --- | ---: | ---: | ---: |
| NVIDIA L4, PyTorch strict FP32 | 149.90 ms | 215.96 ms | reference |
| NVIDIA L4, TensorRT strict FP32 backbone + PyTorch head | 106.21 ms | 170.74 ms | 100.0000% |
| NVIDIA L4, PyTorch AMP BF16 | 54.57 ms | 114.16 ms | 99.9558% |
| NVIDIA L4, TensorRT BF16 backbone + PyTorch head | 24.78 ms | 89.81 ms | 99.9568% |
| Trn2, explicit-gather baseline, BF16/LNC2 | 152.58 ms | 4527.95 ms | 99.9773% |
| Trn2, fused NKI MSDA, BF16/LNC2 | 151.37 ms | 670.60 ms | 99.9802% |

The GPU TensorRT result compiles the complete ConvNeXt-XL backbone with
`require_full_compilation=True`; the OneFormer head remains native PyTorch.
Consequently, 89.81 ms is a hybrid core measurement, not a whole-model
TensorRT engine.

The optimized Trn2 core latency breaks down as follows:

| Component | Unique artifacts | Runtime calls | Mean latency |
| --- | ---: | ---: | ---: |
| ConvNeXt-XL backbone | 37 | 37 | 151.37 ms |
| Pixel decoder | 8 | 8 | 326.07 ms |
| Task encoder | 1 | 1 | 0.35 ms |
| Transformer decoder | 17 | 23 | 176.21 ms |
| Complete pipeline | 63 | 69 | 670.60 ms |

The complete-pipeline p50 is 670.40 ms. Relative to the validated
explicit-gather baseline, the pixel decoder is 12.85x faster and the complete
pipeline is 6.75x faster. The distinction between unique artifacts and
runtime calls matters: reusable transformer mask artifacts are invoked more
than once.

Machine-readable measurements are in
`benchmarks/oneformer_convnext_xl_ade20k_640.json`.

## Model identity

- Original method: OneFormer, ConvNeXt-XL, ADE20K, 640 x 640
- Parameter count: 372,007,256
- Official table metrics: PQ 50.1, AP 36.3, mIoU single-scale 57.4,
  mIoU multi-scale + flip 58.8
- Checkpoint SHA-256:
  `a022437a6cc16fd1485230670f2f7a3ed5e08ef9f08d3f67a42948e5a6a4d7ca`
- Original Detectron2 checkpoint to Hugging Face conversion:
  99.9998% semantic-pixel agreement

The conversion and validation utilities are under `convnext_xl/scripts/`.

## Port structure

Large monolithic graphs exceeded Neuron compiler instruction or state-buffer
limits. The validated path is split into:

1. ConvNeXt embeddings plus each individual block: 37 components.
2. Pixel decoder input, six fused encoder layers, and output: 8 components.
   Each encoder layer combines projections, sampling-coordinate generation,
   an NKI multi-scale deformable-attention kernel, output projection,
   residual normalization, and the feed-forward network.
3. Task encoder: one component.
4. Query preparation, query layers, attention-mask builders, decoder layers,
   and final prediction: 17 unique components and 23 runtime calls.

The NKI kernel receives BF16 values and attention weights while retaining
FP32 sampling locations. Samples that are far enough outside a feature map
to have exactly zero bilinear contribution are assigned zero weight and a
safe in-bounds coordinate before indirect DMA. This preserves zero-padding
semantics and avoids invalid far-outside addresses. The old explicit
bilinear gather remains available as a validated baseline in
`neuron_port/ops.py`.

## Compile and run

The commands below assume a Neuron PyTorch environment, a converted local
model directory, and prepared `inputs.pt`.

```bash
MODEL_DIR=agent_artifacts/data/oneformer_ade20k_convnext_xl_hf
INPUTS=agent_artifacts/data/reference/inputs.pt
CACHE=agent_artifacts/data/hf_cache

python scripts/compile_convnext_stage_pipeline.py \
  --model-id "$MODEL_DIR" \
  --inputs "$INPUTS" \
  --cache-dir "$CACHE" \
  --output-dir agent_artifacts/traces/convnext_stage_bf16_all \
  --chunk-size 1 \
  --precision bf16-all \
  --custom-grid-sample

python scripts/compile_oneformer_remaining_components.py \
  --model-id "$MODEL_DIR" \
  --inputs "$INPUTS" \
  --cache-dir "$CACHE" \
  --output-dir agent_artifacts/traces/oneformer_remaining_bf16_all \
  --precision bf16-all \
  --custom-grid-sample

python scripts/compile_pixel_decoder_micro_pipeline.py \
  --model-id "$MODEL_DIR" \
  --inputs "$INPUTS" \
  --cache-dir "$CACHE" \
  --output-dir agent_artifacts/traces/pixel_decoder_micro_bf16_all \
  --precision bf16-all \
  --sampler-implementation custom \
  --custom-grid-sample

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

python scripts/compile_transformer_pipeline.py \
  --model-id "$MODEL_DIR" \
  --inputs "$INPUTS" \
  --cache-dir "$CACHE" \
  --output-dir agent_artifacts/traces/transformer_pipeline_bf16_all \
  --precision bf16-all \
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

`compile_oneformer_remaining_components.py` also records the expected
monolithic pixel-decoder and transformer compile failures; the task encoder
artifact from that directory is used by the complete pipeline.

The NKI path requires a Neuron SDK generation that contains the NKI Library
experimental MSDeformableAttention kernel. The validated environment used
LNC2, `torch-neuronx` 2.9.0.2.15.32035, `neuronx-cc` 2.27.5334.0, NKI 0.6,
Neuron runtime 2.34.10, and NKI Library commit
`92d11f63a9a8ec1ade34e6e1a3b8db66ef31307e`.

## Optimization direction

The deformable-attention bottleneck is removed: the six fused encoder layers
run in about 12.5-12.7 ms each, or 75.86 ms combined. A component
microbenchmark measures the input projection at 7.44 ms and the final FPN
output at 255.18 ms, so the next target is specifically the FPN upsample and
convolution graph. Further safe fusion of ConvNeXt and transformer components
can also reduce the remaining 69 host invocations.

## Limitations

- Static input shape: batch 1, RGB 640 x 640.
- Semantic segmentation only.
- Latencies exclude image preprocessing and final semantic postprocessing.
- The Trn2 run uses shape-derived positional embeddings prepared outside the
  timed path.
- The NKI MSDeformableAttention API is experimental.
- Compiled artifacts and model weights are intentionally not stored in Git.
- Results describe this port and software stack, not a general hardware
  comparison between Trainium2 and L4.
