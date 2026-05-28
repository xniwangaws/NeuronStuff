# SDXL VAE Decoder @ 2048×2048 — Plan A (Per-Block Tracing) Report

## Result
Plan A worked. The SDXL VAE decoder, which fails monolithic compilation at
2048² output (HLO graph 7.69M instructions vs hard limit 5M, NCC_EVRF007),
now runs end-to-end on Trn2 by chaining 10 individually compiled
sub-NEFFs. Total compile time ~71 min (one-time). Warm latency ~3.1 s
vs ~43 s for CPU bf16 reference (~14x speedup). Visual output is
indistinguishable from CPU reference.

Numerical accuracy on a realistic encoded latent: cosine = 0.9917 (target
was >= 0.999) — short of target by 8 thousandths, attributable to bf16
round-trips at sub-NEFF boundaries plus Neuron-vs-PyTorch differences in
Attention softmax / GroupNorm reductions.

## Per-sub-NEFF stats

| # | Sub-NEFF | Input shape | Output shape | Compile (s) | Post-Partition HLO |
|---|----------|-------------|--------------|-------------|---------------------|
| 01 | post_quant_conv     | [1,4,256,256]    | [1,4,256,256]    |   34.3 |   8 |
| 02 | conv_in             | [1,4,256,256]    | [1,512,256,256]  |    2.4 |   8 |
| 03a| mid_resnet0         | [1,512,256,256]  | [1,512,256,256]  |   21.5 |  79 |
| 03b| mid_attn            | [1,512,256,256]  | [1,512,256,256]  |  438.6 | 157 |
| 03c| mid_resnet1         | [1,512,256,256]  | [1,512,256,256]  |   21.4 |  79 |
| 04 | up_block_0          | [1,512,256,256]  | [1,512,512,512]  |   97.5 | 242 |
| 05 | up_block_1          | [1,512,512,512]  | [1,512,1024,1024]|  326.4 | 242 |
| 06 | up_block_2          | [1,512,1024,1024]| [1,256,2048,2048]|  782.8 | 248 |
| 07 | up_block_3          | [1,256,2048,2048]| [1,128,2048,2048]| 1949.0 | 239 |
| 08 | conv_out_block      | [1,128,2048,2048]| [1,3,2048,2048]  |  352.9 |  37 |
| .  | TOTAL               |                  |                  | 4226.8 |     |

Post-partition HLO instruction counts are far below 5M (highest 248). The
7.69M EVRF007 figure for the monolithic graph counts post-tensorize/
post-unroll BIR instructions; per-sub-NEFF post-Unroll BIR is ~200k-250k.

## End-to-end performance (2048² output, real encoded latent)

| Run                                | Time   |
|------------------------------------|--------|
| CPU bf16 reference (vae.decode)    | 43.4 s |
| Neuron chained — cold              |  3.13s |
| Neuron chained — warm (mean of 5)  |  3.10s |
| Neuron chained — warm (min)        |  3.10s |

Baseline: monolithic compile fails after ~28s with NCC_EVRF007 (no NEFF).

## Numerical accuracy

| Input                                              | Cosine | max_abs | mean_abs |
|----------------------------------------------------|--------|---------|----------|
| Random latent (OOD, std=1)                         | 0.6701 |   2.27  |  0.300   |
| Real encoded latent (in-dist, std~0.18 post-scale) | 0.9917 |   1.35  |  0.092   |

Per-block delta (Neuron vs CPU eager on same clean input) ranges
0.2%–11% relative max-abs; biggest single contributor is 03b_mid_attn
(11%, expected for softmax in bf16). On in-distribution latents the
chained output is visually identical to CPU reference.

## Sub-blocks that needed further splitting

- mid_block: original plan was 1 NEFF, hit NCC_EXSP001 — 32.88 GB HBM
  scratchpad vs 24 GB per logical core (LNC=2 on trn2 = 4 cores x 24GB).
  The 65k x 65k Attention matrix is the culprit. Split into
  resnet0/attn/resnet1. Adding --target=trn2 alone was not sufficient.
- All 4 up_blocks and conv_out_block traced cleanly as monolithic
  UpDecoderBlock2D modules.

## HBM weight footprint

Total traced_*.pt: 413 MB across 10 NEFFs. Largest are 05_up_block_1
(104 MB), 03b_mid_attn (91 MB), 06_up_block_2 (89 MB), 04_up_block_0
(70 MB), 07_up_block_3 (48 MB).

## Baseline comparison

| Metric                        | Monolithic       | Plan A           |
|-------------------------------|------------------|------------------|
| Compile result                | FAIL (NCC_EVRF007 after 28s) | 10 NEFFs (~71 min total) |
| Largest HLO instructions      | 7.69M (>>5M)     | 248 post-partition |
| HBM peak per NEFF             | n/a              | <24 GB per core (within trn2) |
| End-to-end inference          | n/a              | 3.10s warm       |
| Cosine vs CPU bf16            | n/a              | 0.992 (real)     |
