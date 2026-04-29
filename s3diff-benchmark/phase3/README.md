# Phase 3: full v3 benchmark with cat image on all 3 devices

Re-ran the full 3×3 (device × resolution) matrix with a single natural photo
(`cat_LQ_256.png`, 256×256 orange tabby cat) to compare H100 / L4 / Trn2 fairly.
Phase 1/2 used the `london2.jpg` bus-grid composite image, which was already a
"pre-tiled" image and made it impossible to distinguish content from artifacts.

## Environment

| Component | Value |
|---|---|
| Phase 2 instance (trn2.3xlarge sa-east-1b) | expired; re-ran on trn2.48xlarge |
| Phase 3 Trn2 instance | `i-0d543ca2558e58c37` us-east-2b, trn2.48xlarge |
| H100 instance | `i-0e99206276aeb2822` ap-northeast-1c, p5.4xlarge (sa/us regions had no capacity) |
| L4 instance | `i-0995d0f8ab57c17f4` us-east-2b, g6.4xlarge |
| Neuron SDK | 2.29.0, DLAMI 20260410, `vllm_0_16` venv |
| diffusers | 0.25.1 (GPU), 0.34.0 (Neuron) |
| Input | `cat_LQ_256.png` resized to bucket LQ (256/512/1024) |

## Results (BF16, 10 runs, seed 123)

| Device | Res | Load | Cold | **Steady** | P50 | P95 | Peak mem | Seam luma | PSNR vs CPU |
|---|---|---|---|---|---|---|---|---|---|
| H100 80GB | 1K | 4.80s | 9.64s | **1.26s** | 1.26 | 1.26 | 9.0 GB | 3.4 | 45.10 dB |
| H100 80GB | 2K | 4.02s | 24.67s | **24.26s** | 24.19 | 24.54 | 15.7 GB | 2.3 | — |
| H100 80GB | 4K | 4.03s | 109.71s | **107.54s** | 107.33 | 108.67 | 42.2 GB | 1.6 | — |
| L4 24GB | 1K | 4.52s | 6.41s | **2.34s** | 2.34 | 2.35 | 7.9 GB | 3.3 | 45.15 dB |
| L4 24GB | 2K | 4.07s | 29.11s | **28.45s** | 28.47 | 28.64 | 15.2 GB | 2.3 | — |
| L4 24GB | 4K | 4.08s | 132.42s | **130.63s** | 130.63 | 130.97 | 16.5 GB | 1.6 | — |
| **Trn2 (v3)** | 1K | 55.90s | 24.94s | **24.91s** | 24.91 | 24.92 | — | 39.0 | 24.55 dB |
| **Trn2 (v3)** | 2K | 56.01s | 82.71s | **82.67s** | 82.68 | 82.68 | — | 33.3 | — |
| **Trn2 (v3)** | 4K | 55.71s | 360.11s | **360.10s** | 360.11 | 360.13 | — | 18.6 | — |

`Seam luma` = max row-luma discontinuity (an empirical tile-seam visibility proxy).
Higher = more visible tile boundary. GPU values 1-3 are below human-perception
threshold. Trn2 values 18-39 are still visible in the output PNG.

`Peak mem` on Neuron: `neuron-ls` sampler sometimes returns 0; excluded where unreliable.

## Trn2/GPU speed ratio (lower is better)

| Resolution | Trn2/H100 | Trn2/L4 |
|---|---|---|
| 1K | 19.8× slower | 10.6× slower |
| 2K | 3.4× slower | 2.9× slower |
| 4K | 3.3× slower | 2.8× slower |

1K is an outlier because VAE decoder tiling overhead dominates at that
resolution (9 tiles × ~2.5s/tile ≈ 22s just for VAE decode, plus 2s for single
UNet call). 2K/4K converge to a more reasonable 3× behind H100.

## Accuracy findings

- **GPU-to-GPU** (H100 vs L4, same input, same seed): PSNR **45.1 dB** — essentially identical.
- **Neuron v3 vs CPU eager** (same input, same seed, 1K): PSNR **24.55 dB**.
- **Neuron v3 vs H100**: PSNR 24.56 dB.
- Content preservation confirmed (mean RGB within 0.1 of cat LQ's (175.9, 148.7, 120.7)).

### Remaining quality gap

The 20 dB PSNR gap Neuron-to-GPU is dominated by **tile-seam visibility** at
the VAE decoder latent 64 → pixel 512 boundary. Two mitigations tried:

1. **Overlap + gaussian blend in Python tile loop**: seam 73 → 39 (overlap=16),
   plateaus at 37.8 (overlap=32). PSNR gain only +0.4 dB. Root cause is
   per-tile BF16 numerical drift interacting with overlapping tile blends —
   not the blend algorithm itself.
2. **Retrace VAE decoder at larger latent**: failed at latent 80, 96, 128 —
   all exceed neuronx-cc's 5M instruction limit (`NCC_EVRF007`, 8-26M generated
   vs 5M typical limit).

The only remaining path to PSNR parity is **reducing tile count further** via
NxDI TP or a smaller-but-more-tiles decoder (e.g. latent 48), neither explored
due to capacity block expiry.

## Phase 3 code

- `scripts/neuron_unet_trace_v3b.py` — Plan B: attribute-routing trace
  (writes `unet_de_mods[i]` to each LoRA module's `.de_mod` inside the traced
  forward). **CTX-CHECK passed**: changing `de_mods` after trace changes output
  (sensitivity 0.172), proving de_mods flows through HLO as a real input.
  Same NEFF works for any LQ image's degradation score.
- `scripts/neuron_e2e_v3.py` — e2e pipeline using traced UNet (full-latent at
  1K, tiled at 2K/4K via the same NEFF) + traced VAE encoder/decoder.
- `scripts/patch_s3diff_v3.sh` — idempotent patches to S3Diff repo
  (`.cuda()→.cpu()`, torchvision shim, layer-name tagging).

## Per-run data

See `*.json` files in this directory — each contains 10 `per_run_s` samples
plus load / cold / steady / p50 / p95 derived metrics.

## Known issues

- L4 4K requires `--vae_decoder_tiled_size 128` (default 224 OOMs on 22GB). Documented in `scripts/` via comment in Phase 2's `bench.py`.
- Trn2 4K took 360s steady, well above Phase 2's 609s — actually faster despite
  attribute-routing overhead, because VAE encoder is now traced (not CPU eager
  like Phase 1).
- Phase 2 data (`trn2v2_{1K,2K}.json` in parent `results/`) is still the v2
  baseline — left there for A/B comparison in the matrix.

## Generated images

All SR outputs from the cat benchmark are in `phase3/images/`:

| File | Device | Resolution | Notes |
|---|---|---|---|
| `cat_LQ_256_input.png` | — | 256×256 | Source LQ (resized by bench to each bucket's LQ side) |
| `CPU_eager_cat_1K.png` | CPU eager | 1024×1024 | Reference (official S3Diff pipeline, no Neuron) |
| `H100_cat_{1K,2K,4K}.png` | H100 80GB | 1K/2K/4K | p5.4xlarge, diffusers 0.25.1 + `--mixed_precision bf16` |
| `L4_cat_{1K,2K,4K}.png` | L4 24GB | 1K/2K/4K | g6.4xlarge, same config (4K used `--vae_decoder_tiled_size 128` to avoid OOM) |
| `Trn2v3_cat_{1K,2K,4K}.png` | Trainium2 | 1K/2K/4K | v3 pipeline (attribute-routed UNet + traced VAE) |
| `Trn2v3_cat_1K_stitch_no_overlap.png` | Trainium2 | 1024×1024 | v3 baseline: VAE decoder tile stitch, no overlap (seam luma 73) |
| `Trn2v3_cat_1K_overlap16.png` | Trainium2 | 1024×1024 | v3 + overlap 16 gaussian blend (seam luma 39) |
| `Trn2v3_cat_1K_overlap32.png` | Trainium2 | 1024×1024 | v3 + overlap 32 gaussian blend (seam luma 38, plateau) |

Open the 1K images side-by-side to see:
- **GPU outputs**: no visible tile seam (luma discontinuity 1-3)
- **Trn2 stitch**: horizontal band at row 511 (seam 73)
- **Trn2 overlap 16**: softened but still visible band (seam 39)
- **Trn2 overlap 32**: marginal further improvement (seam 38)

## Why VAE decoder is slow (next-phase target)

From the AWS Neuron agentic-dev knowledge base `TEXT_TO_VIDEO_MODEL_PORTING.md`
and `Category2_Sharding_Memory_Issues.md`:

- **Root cause**: SD-Turbo's VAE decoder is Conv2d-dominant. Neuron's Tensor
  Engine is matmul-optimized (systolic array); small spatial Conv2d patches
  underutilize the hardware. On H100, tensor cores vectorize conv well; on
  Neuron, each patch becomes a small matmul (<128×128) with poor utilization.
- **Compile limit** (`NCC_EVRF007`): whole-decoder at latent 80/96/128 generates
  8-26M instructions vs 5M NEFF limit. This blocks the "trace at larger latent
  to eliminate tile" approach.
- **No TP benefit**: CNN decoder can't benefit from tensor parallelism like
  attention-heavy DiT — all-reduce overhead exceeds compute savings for
  sub-128 matmuls.

**Next-phase optimization ideas** (unvalidated):

1. **Block-by-block NEFF**: split decoder into ~6 NEFFs (conv_in, mid_resnets,
   mid_attn, each up_resnet, norm_out). Chain at runtime (~1-2ms boundary
   overhead). Each block small enough that compile at latent 128 succeeds.
   This would eliminate tiling altogether at 1K. Expected: 2× speedup + no
   seam. `TEXT_TO_VIDEO §4 Type C`.
2. **NKI Conv2d kernel**: `neuron-nki-agent/examples/conv2d_scaling_min/` has a
   fused Conv2d + bias + scaling kernel template. Expected 2-3× on Conv-heavy
   blocks if targeted correctly.
3. **Compiler flag tuning**: try `--vectorize-strided-dma` (conv-friendly).
   Expected 10-20%.
4. **TP/CP**: ruled out for VAE.
