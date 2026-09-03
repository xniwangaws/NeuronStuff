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

The complete model runs successfully as 78 sequential Neuron components.
The final stable benchmark used three warmups and ten measured runs.

| Platform and path | Backbone | OneFormer core | Semantic agreement |
| --- | ---: | ---: | ---: |
| NVIDIA L4, PyTorch strict FP32 | 149.90 ms | 215.96 ms | reference |
| NVIDIA L4, TensorRT strict FP32 backbone + PyTorch head | 106.21 ms | 170.74 ms | 100.0000% |
| NVIDIA L4, PyTorch AMP BF16 | 54.57 ms | 114.16 ms | 99.9558% |
| NVIDIA L4, TensorRT BF16 backbone + PyTorch head | 24.78 ms | 89.81 ms | 99.9568% |
| Trn2, Neuron BF16/LNC2 | 152.58 ms | 4527.95 ms | 99.9773% |

The GPU TensorRT result compiles the complete ConvNeXt-XL backbone with
`require_full_compilation=True`; the OneFormer head remains native PyTorch.
Consequently, 89.81 ms is a hybrid core measurement, not a whole-model
TensorRT engine.

The Trn2 core latency breaks down as follows:

| Component | Count | Mean latency |
| --- | ---: | ---: |
| ConvNeXt-XL backbone | 37 | 152.58 ms |
| Pixel decoder | 23 | 4190.42 ms |
| Task encoder | 1 | 0.22 ms |
| Transformer decoder | 17 | 175.31 ms |
| Complete pipeline | 78 | 4527.95 ms |

The complete-pipeline p50 is 4526.38 ms. The pixel decoder accounts for
92.5% of total latency, so this is a correctness-first port rather than an
optimized representation of Trn2 peak performance.

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
2. Pixel decoder input, three reusable bilinear samplers, six
   projection/combine/post groups, and output: 23 components.
3. Task encoder: one component.
4. Query preparation, query layers, attention-mask builders, decoder layers,
   and final prediction: 17 components.

The integrated deformable-attention `grid_sample` path did not preserve
dynamic inputs correctly when traced. The valid path uses an explicit
bilinear gather implemented in `neuron_port/ops.py`. An isolated native
`grid_sample` experiment compiled but failed numerical validation and must
not be used.

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
  --pixel-decoder-dir agent_artifacts/traces/pixel_decoder_micro_bf16_all \
  --remaining-dir agent_artifacts/traces/oneformer_remaining_bf16_all \
  --transformer-dir agent_artifacts/traces/transformer_pipeline_bf16_all \
  --output agent_artifacts/results/trn2_full_oneformer_bf16_all.json \
  --warmup 3 \
  --runs 10 \
  --custom-grid-sample
```

`compile_oneformer_remaining_components.py` also records the expected
monolithic pixel-decoder and transformer compile failures; the task encoder
artifact from that directory is used by the complete pipeline.

## Optimization direction

The next meaningful optimization is a fused NKI implementation of the
multi-scale deformable-attention sampler. Merely changing compiler
optimization level is unlikely to remove the current gather/DMA bottleneck.
Reducing the 78 host-dispatched components through safe graph fusion is the
second priority.

## Limitations

- Static input shape: batch 1, RGB 640 x 640.
- Semantic segmentation only.
- Latencies exclude image preprocessing and final semantic postprocessing.
- The Trn2 run uses shape-derived positional embeddings prepared outside the
  timed path.
- Compiled artifacts and model weights are intentionally not stored in Git.
- Results describe this port and software stack, not a general hardware
  comparison between Trainium2 and L4.
