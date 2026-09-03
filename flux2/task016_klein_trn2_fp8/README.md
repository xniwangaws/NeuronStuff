# FLUX.2 Klein 9B at 1024px on `trn2.3xlarge`: BF16 and mixed FP8

This directory contains a reproducible 1024x1024 inference benchmark for
`black-forest-labs/FLUX.2-klein-base-9B` on one AWS Trainium2
`trn2.3xlarge`.

The FP8 weights are **not** from a separately downloaded checkpoint. The
starting checkpoint is the original Hugging Face BF16 checkpoint. The included
exporter converts selected scopes into sharded NxDI checkpoints containing
E4M3 weights and FP32 per-output-row scales.

## Precision scope

| Scope | E4M3 per-row weights | Dynamic E4M3 activations |
|---|---|---|
| `mlp` | All block MLP projections | MLP Linear inputs |
| `mlp_attention` | MLP plus Q/K/V and attention output projections | Corresponding Linear inputs |
| `all_linear` | All 289 Transformer Linear/GEMM weights | Every Transformer Linear input |

`all_linear` is the closest precise meaning of model W8A8 in this experiment:
all Transformer Linear/GEMM projections use FP8 weights and FP8 activations.
Normalization, RoPE, softmax, residual arithmetic, and the NKI attention kernel
remain BF16/FP32. It is therefore not an “every operator is FP8” graph.

Weight-only mode stores selected weights in E4M3 but dequantizes them before a
BF16 GEMM; it is a W8A16 storage path and showed no speedup.

## Prequantized per-row checkpoint

`src/export_fp8_checkpoint.py` converts the Hugging Face BF16 transformer into
an NxDI-format checkpoint containing E4M3 weights and FP32 per-output-row
scales:

| Scope | FP8 weights | Checkpoint size |
|---|---:|---:|
| `mlp` | 120 | 12.12 GB |
| `mlp_attention` | 280 | 9.44 GB |
| `all_linear` | 289 | 9.09 GB |

The exported checkpoint was loaded through the normal NxDI sharding path and
tested against the original BF16 checkpoint quantized during loading. With the
same 1024x1024, four-step, seed-42 run, both paths produced byte-identical PNG
files (`MSE=0`, maximum absolute pixel difference `0`). This validates the
offline-quantize-on-GPU/CPU, deploy-on-Trainium workflow; a checkpoint from
another quantizer still needs a key, matrix-orientation, dtype, scale-shape,
and tensor-parallel adapter.

## Benchmark setup

- Instance: `trn2.3xlarge`
- Tensor parallel degree: 4
- Logical NeuronCore configuration: LNC=2
- Visible logical cores: `0-3`
- Resolution: 1024x1024
- Denoising steps: 50
- Guidance scale: 4.0
- Seeds: 42 through 51
- Prompt: `A cat holding a sign that says hello world`
- Model: `black-forest-labs/FLUX.2-klein-base-9B`

Software versions used:

- Neuron SDK AMI release: 2.29.1
- `neuronx-cc`: `2.24.8799.0+6f62ff7c`
- `torch-neuronx`: `2.9.0.2.13.26312+8e870898`
- `neuronx-distributed`: `0.18.27753+1cafd54f`
- `neuronx-distributed-inference`: `0.9.17334+ced6ae4e`
- PyTorch: `2.9.1+cu128`
- Diffusers: `0.37.1`

## Results

| Mode | Samples | Valid | Mean latency | Speedup vs BF16 |
|---|---:|---:|---:|---:|
| BF16 | 10 | 10/10 | 41.785 s | 1.000x |
| FP8 MLP weight-only | 1 | 1/1 | 41.989 s | 0.995x |
| FP8 MLP dynamic activation | 10 | 10/10 | 38.562 s | 1.084x |
| FP8 all Transformer Linear dynamic | 10 | 10/10 | **37.893 s** | **1.103x** |

Dynamic MLP FP8 reduced mean latency by 7.71%. All-linear W8A8 reduced it by
9.31%. At the Capacity Block effective rate used for this run ($2.235/hour),
measured generation cost was approximately:

- BF16: $0.02594/image
- MLP W8A8: $0.02394/image
- all-Transformer-Linear W8A8: $0.02353/image

These figures exclude one-time compilation and model-loading time. The
intermediate `mlp_attention` scope produced 37.569 s for seed 42, but was not
run as a ten-seed benchmark and is not used for the aggregate comparison.

Ten-seed pixel comparison against BF16:

| Scope | Mean SSIM | Mean PSNR | Mean MAE | Mean MSE |
|---|---:|---:|---:|---:|
| MLP W8A8 | 0.9564 | 22.87 dB | 0.03615 | 0.008716 |
| all Linear W8A8 | 0.9282 | 19.96 dB | 0.05015 | 0.015233 |

These are diagnostic pixel metrics, not a perceptual-quality benchmark.
Diffusion trajectories can amplify small numeric differences. Inspect
`results/comparison_grid_bf16_mlp_all_linear.png` alongside the per-seed
metrics. The all-linear scope is faster but has more numeric drift than the
MLP-only scope.

## Important implementation details

1. FP8 weights are quantized to E4M3 per output row (`axis=0`) with FP32
   scales shaped `[out_features, 1]`.
2. Trainium2 requires PyTorch `float8_e4m3fn` HLO values to be compiled using
   the HLO-to-tensorizer compatibility option
   `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3`.
3. The compatibility option must be combined with NxDI's
   `--verify-hlo=true` in a single
   `--internal-hlo2tensorizer-options` argument.
4. NxD's dynamic activation path produces a broadcast-ready
   `[batch, sequence, 1]` scale. The included compatibility shim avoids adding
   a fourth dimension before multiplying that scale into a 3D linear output.
5. Timestep and modulation projections normally receive 2D tensors. The
   all-linear path adds a length-one sequence dimension so NxD's dynamic FP8
   activation scale remains broadcast-compatible, then removes it afterward.
6. Diffusers 0.37.1 `Flux2KleinPipeline` does not accept `negative_prompt`; the
   wrapper uses Klein Base's built-in empty unconditional prompt.
7. The timestep sinusoidal embedding denominator is `half_dim`, matching the
   reference implementation.

## Run

The scripts assume:

- model files at `/mnt/nvme/flux2-klein/weights`
- source files at `/mnt/nvme/flux2-klein/src`
- the AWS Neuron PyTorch 2.9 NxDI venv at
  `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`

BF16:

```bash
scripts/run_bf16_remote.sh \
  --steps 50 \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --warmups 1
```

FP8 MLP weight-only:

```bash
FLUX2_FP8_ACTIVATION=none scripts/run_fp8_remote.sh \
  --steps 50 \
  --seeds 42 \
  --warmups 1
```

FP8 MLP with dynamic activation quantization:

```bash
FLUX2_FP8_SCOPE=mlp FLUX2_FP8_ACTIVATION=dynamic \
  scripts/run_fp8_remote.sh \
  --steps 50 \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --warmups 1
```

Export and use a prequantized MLP checkpoint:

```bash
python src/export_fp8_checkpoint.py \
  --scope mlp \
  --output /mnt/nvme/flux2-klein/checkpoints/fp8_mlp_per_row

FLUX2_FP8_SCOPE=mlp FLUX2_FP8_ACTIVATION=dynamic \
  scripts/run_fp8_remote.sh \
  --transformer-checkpoint \
    /mnt/nvme/flux2-klein/checkpoints/fp8_mlp_per_row \
  --steps 50 \
  --seeds 42
```

Export and run the all-Transformer-Linear W8A8 checkpoint:

```bash
python src/export_fp8_checkpoint.py \
  --scope all_linear \
  --output /mnt/nvme/flux2-klein/checkpoints/fp8_all_linear_per_row

FLUX2_FP8_SCOPE=all_linear FLUX2_FP8_ACTIVATION=dynamic \
  scripts/run_fp8_remote.sh \
  --transformer-checkpoint \
    /mnt/nvme/flux2-klein/checkpoints/fp8_all_linear_per_row \
  --steps 50 \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --warmups 1
```

The first invocation for each scope/activation pair compiles the transformer.
The launcher keeps scope-specific compiled and output directories so an MLP
graph cannot be accidentally reused for an all-linear checkpoint.

## Directory contents

- `src/`: NxDI FLUX.2 Klein model, application wrapper, and benchmark driver
- `scripts/`: launch wrappers and result comparison tools
- `results/bf16/`: BF16 JSON and ten generated images
- `results/fp8_weight_only/`: weight-only JSON and representative image
- `results/fp8_dynamic/`: dynamic FP8 JSON and ten generated images
- `results/fp8_mlp_attention/`: intermediate 50-step seed-42 result
- `results/fp8_all_linear/`: all-linear manifest, JSON, and ten images
- `results/comparison_grid.png`: original BF16/MLP-FP8 grid
- `results/comparison_grid_bf16_mlp_all_linear.png`: final three-column grid
- `results/comparison_vs_bf16.json`: original MLP per-seed metrics
- `results/comparison_all_linear_vs_bf16.json`: all-linear per-seed metrics
- `results/checkpoint_equivalence.json`: offline checkpoint/load equivalence
- `results/experiment_summary_v2.json`: final latency, cost, and quality summary
- `logs/`: final benchmark logs
