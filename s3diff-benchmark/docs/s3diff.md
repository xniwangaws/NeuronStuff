# S3Diff

## Overview

**Status**: Phase 1 complete (GPU baselines + Neuron UNet port + 9-cell benchmark matrix)
**Created**: 2026-04-24
**Location**: `/Users/xniwang/oppo-opencode/projects/s3diff/`
**Code Directory**: `/Users/xniwang/oppo-opencode/working/s3diff/` (create this directory for your project)
**Related Repos**:
- Source: https://github.com/ArcticHare105/S3Diff
- Weights: https://huggingface.co/zhangap/S3Diff
- Base: `stabilityai/sd-turbo`
**Remote Testing**: required
**Testing Instance** (all terminated after benchmark):
- GPU L4: was `i-08c7a92287ca9cb9b` / 56.124.87.197 (g6.4xlarge, sa-east-1a)
- GPU H100: was `i-0a1e85a8e078bd9ca` / 13.158.6.113 (p5.4xlarge, ap-northeast-1c)
- Neuron: was `i-0590be734bb61d071` / 54.233.193.151 (trn2.3xlarge, sa-east-1b, SDK 2.29 DLAMI 20260410)
- **Capacity block**: cr-022854d06bad71ad1 ($17.32, 7h45m, sa-east-1b)
- **Capacity note**: p5.4xlarge was InsufficientCapacity across sa-east-1 / us-east-1 / us-east-2 / us-west-2 / ap-southeast-2 / ap-south-1 on 2026-04-24. Tokyo (ap-northeast-1c) had capacity — launched there with fresh `neuron-bench-ap-northeast-1` key pair.
**Trainium Generation**: trn2
**Region**: sa-east-1 (Sao Paulo)

## Description

Port S3Diff — a 1-step degradation-guided image super-resolution model built on SD-Turbo — to AWS Neuron (Trainium2), and benchmark it end-to-end against H100 and L4 GPUs at 1K / 2K / 4K output resolutions in BF16.

S3Diff is a UNet pipeline (not a DiT), x4 upscale. Inference is single-step (DDPMScheduler t=999, CFG 1.07) with degradation-aware LoRA adapters (VAE r=16, UNet r=32) and a small DEResNet degradation estimator. High resolutions use latent-space tiling (96×96 latent tiles, 32 overlap) plus tiled VAE encode/decode (256×256 pixel tiles) — 4K is mandatory-tiled.

Weights on HF: `s3diff.pkl` (~75 MB, LoRA + degradation MLPs) and `de_net.pth` (~65 MB, DEResNet). Base SD-Turbo (~2.6 B params) downloaded from HF at load time. BF16 total weight footprint ≈ 2.2 GB.

## Goals

- [x] Run official S3Diff pipeline on H100 (`p5.4xlarge`) and L4 (`g6.4xlarge`) in BF16 — reference images + perf captured.
- [x] Port to Neuron via `torch_neuronx.trace()` — single NEFF for UNet at tile 96x96 (de_mod baked for a specific LQ image); VAE/text-encoder/DEResNet kept on CPU.
- [x] Benchmark harness reporting per (device × resolution): model load time, cold-start, steady-state mean, P50/P95, peak device memory (via nvidia-smi sampler on GPU).
- [x] **9-cell matrix populated** (3 devices × 3 resolutions, BF16) — see `working/s3diff/results/summary.csv`.
- [ ] Full accuracy package (10 images × 3 resolutions) — **1 image done** (Golden Gate Bridge), PSNR 19.7 dB for Neuron vs CPU eager.
- [x] Final report at `working/s3diff/results/summary.md` with tables + comparison grids.

## Non-goals

- **No FP8.** S3Diff has no official FP8 implementation (no torchao, transformer-engine, or quant code in repo). Out of scope.
- **No FP32 axis.** Official default is FP32, but benchmark focuses on deployment-realistic BF16 only (per user direction).
- **No training.** Inference only.

## Active Tasks

See `/Users/xniwang/oppo-opencode/projects/s3diff/tasks/README.md` for current task list.

Initial task sequence (follows `steering/model-onboarding-tasks.md`, adapted for vision — skip token-level validation, add image-quality validation):

1. Research + reference images on `g6.4xlarge` (L4) in BF16
2. H100 reference on `p5.4xlarge` in BF16
3. Neuron env setup (deviation: isolated venv for `diffusers==0.25.1` + `peft==0.10.0`, not the pre-installed `vllm_0_16` / `pytorch_2_9_nxd_inference`)
4. Trace DEResNet + VAE encoder at 1K resolution
5. Trace UNet (tiled, fixed shape `[1, 4, 96, 96]`) at 1K
6. Trace VAE decoder (tiled) at 1K — flag if HBM overflows
7. End-to-end 1K Neuron run + latent cosine + image PSNR vs. GPU
8. Extend to 2K and 4K (separate NEFF sets, or dynamic tile count at 4K)
9. 4K HBM triage — LNC=1 fallback, then NxDI-Flux port if needed
10. Benchmark harness (shared across all 3 devices)
11. Run 9-cell matrix (3 devices × 3 resolutions)
12. Accuracy comparison grids + PSNR/LPIPS
13. Final report

## Implementation notes

- **Framework**: `torch_neuronx.trace()` primary. NxDI (Flux reference pattern) held in reserve for 4K fallback if UNet or VAE decoder overflows single-core HBM. See `steering/nxdi-model-porting.md:27-31,42-49`.
- **LNC**: start LNC=2 (default, 4 cores × 24 GB). Switch to LNC=1 via `NEURON_LOGICAL_NC_CONFIG=1` if single-core HBM is tight.
- **Compiler flag**: `--auto-cast=none` (explicitly, per user direction — keeps declared BF16 dtypes, no FP8 matmul intermediates).
- **xformers**: disabled on all devices for apples-to-apples attention (Neuron doesn't have it; GPU runs without it to match).
- **VAE fallback**: if tiled VAE decoder at 4K exceeds HBM budget, run decode on CPU (LTX-2 precedent).
- **Env**: S3Diff pins `diffusers==0.25.1`, `peft==0.10.0`, `torch==2.1.0`. The Neuron-compatible `torch==2.9.x` in `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/` (SDK 2.29) is incompatible — create a dedicated venv.

## Steering

- **Project-specific**: `/Users/xniwang/oppo-opencode/projects/s3diff/steering/`
- **Global**: `/Users/xniwang/Neuron-steering-docs/steering/`
  - `model-onboarding-tasks.md` — 18-task template
  - `nxdi-model-porting.md` — NxDI decision, Flux reference, RollingForcing lesson
  - `neuron-sdk.md` — `--auto-cast` options, BF16 guidance
  - `aws-ec2-guide.md` — capacity block purchase (trn2 only), AMI, `OC-` naming

## Related projects

- `flash-vsr` — Wan 2.1 super-resolution, used NxDI + NKI flash (transformer-heavy; different from S3Diff UNet)
- `LTX-2.3` — DiT video, VAE-on-CPU precedent for wide decoders
- `qwen-image-edit` — sequential component lifecycle (encode → free → iterate → decode)
- `SIGLIP-384` — LNC tuning for vision models on trn2

## Phase 1 results (2026-04-27)

9-cell BF16 matrix (steady-state latency / peak device memory):

| Device | 1K | 2K | 4K |
|---|---|---|---|
| NVIDIA H100 80GB | **1.24s** / 9.0 GB | 24.4s / 15.7 GB | 107s / 42.2 GB |
| NVIDIA L4 24GB | 2.34s / 7.9 GB | 27.9s / 15.2 GB | 126s / 16.5 GB (tile=128) |
| Trainium2 (LNC=2) | 24.7s | 140.3s | 609s |

Full data: `/Users/xniwang/oppo-opencode/working/s3diff/results/summary.csv`
Side-by-side grids: `results/comparison_{1K,2K,4K}.png`

### Neuron port findings (blockers / lessons)

1. **de_mod runtime injection** — S3Diff's LoRA layers receive `de_mod` as a dynamically-set attribute in `my_lora_fwd`. `torch_neuronx.trace()` can't capture attribute writes, so `de_mod` must be baked as a **per-layer nn buffer before trace**. The traced UNet is then valid only for that specific LQ image's degradation score. Production use would require either (a) re-trace per image (32 min cost), or (b) rewrite `my_lora_fwd` to accept `de_mod` as an explicit tensor input.
2. **Compiler OOM at B=2** — trn2.3xlarge has 124 GB system RAM; neuronx-cc was killed compiling UNet at CFG B=2. Compiled at B=1 instead and do CFG pos/neg as two separate Python-level calls to the traced UNet. Workaround: added 64 GB swap.
3. **VAE on CPU is the bottleneck** — Neuron UNet tile 0.535s × (4-81 tiles × 2 CFG) is fast; but VAE encode/decode on CPU at 2K/4K takes minutes. This is why trn2 end-to-end is 5-20× slower than GPU. **Next iteration: trace VAE.**
4. **SD-Turbo, diffusers 0.25 → 0.34 compat** — S3Diff pins `torch==2.1` + `diffusers==0.25.1`; Neuron SDK 2.29 pins `torch==2.9`. Successfully ported to `diffusers==0.34.0` + `peft==0.19.1`. Required `torchvision.transforms.functional_tensor` shim (module removed in tv 0.17+) and `.cuda()` → `.cpu()` patches in `s3diff_tile.py`, `model.py`, `my_utils/devices.py`.
5. **Tile blending seam** — S3Diff's default Gaussian weight `var=0.01` is very narrow (edge weight → 0). With Neuron UNet's cosine ≈ 0.9955 drift, this magnifies into a visible horizontal seam at tile overlap boundaries. Mitigated by widening to `var=0.1` (seam luma discontinuity: 83 → 71). Remaining seam is Neuron numerical drift — not fully fixable at this abstraction layer.
6. **pkg_resources / setuptools 81+** — S3Diff's `clip` dep uses `from pkg_resources import packaging` which was removed in setuptools 81. Pinned `setuptools<81` in both GPU and trn2 venvs.

### Accuracy

Neuron output vs CPU eager (same LQ, same seed):
- Trn2 UNet tile vs CPU UNet: **cosine 0.9955**, max |diff| 1.8, std preserved
- End-to-end PSNR on 1K bridge output: **19.7 dB** (dominated by tile seam + VAE-CPU vs Neuron-UNet interaction, not raw UNet drift)
- Content preservation confirmed (mean RGB matches CPU eager output within ~0.3%)

### Next work

- Re-architect `my_lora_fwd` to accept `de_mod` tensor input → UNet trace becomes input-independent.
- Trace VAE encoder + decoder on Neuron → remove CPU bottleneck (biggest latency win expected).
- Run full 10-image accuracy set with ground-truth HQ, report PSNR/LPIPS averages.

## Notes

- S3Diff is a **UNet (SD-Turbo)**, not a DiT. The user's benchmark spec asks for "DiT-based model loading + cold-start breakdown" — we report the same metrics (load time, first-inference, mean-excluding-first) but label the model correctly in the final report.
- Official S3Diff default is FP32; BF16 is an officially supported switch (`--mixed_precision bf16` → accelerate autocast). We use BF16 exclusively.

## Completion Criteria

- [ ] Neuron port runs end-to-end at 1K/2K/4K with correct images (PSNR > 35 dB vs. GPU reference on a sample)
- [ ] `results/summary.csv` has all 9 cells populated
- [ ] Accuracy comparison grids saved for 10 images × 3 resolutions
- [ ] Final report published at `results/README.md`
- [ ] Project registry updated to Completed
