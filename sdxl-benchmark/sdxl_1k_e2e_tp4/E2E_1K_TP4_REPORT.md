# SDXL 1024×1024 E2E on Trn2 — TP=4 UNet

## Pipeline composition

| Component | Implementation | Source |
|---|---|---|
| UNet | TP=4 single NEFF (4 logical cores), batch=1 | `compile_tp4_1k/unet_tp4/2026-05-29T05-21-41.412016/` (re-compiled via cache key, recompile took 822 s on this run; previously 12.9 min) |
| Text encoder (CLIP-L) | Single-core TorchScript NEFF | `/home/ubuntu/sdxl/compile_dir_1024/text_encoder/model.pt` (Jim's 1K trace) |
| Text encoder 2 (CLIP-G) | Single-core TorchScript NEFF | `/home/ubuntu/sdxl/compile_dir_1024/text_encoder_2/model.pt` |
| VAE decoder | Single-core TorchScript NEFF | `/home/ubuntu/sdxl/compile_dir_1024/vae_decoder/model.pt` |
| VAE post_quant_conv | Single-core TorchScript NEFF | `/home/ubuntu/sdxl/compile_dir_1024/vae_post_quant_conv/model.pt` |
| Scheduler / tokenizers | Diffusers default (CPU) | — |

Precision: bf16 throughout. Hardware: trn2.3xlarge, LNC=2, 4 logical cores, sa-east-1.

## CFG=7.5 strategy

The TP=4 UNet NEFF was traced for **batch=1**. Diffusers SDXL with CFG=7.5 wants
to call UNet with batch=2 (cond + uncond concatenated). The wrapper splits the
batch=2 input into two batch=1 NEFF calls and concatenates the result — no
recompile, ~2× per-step latency vs single-core CFG-disabled.

## Latency (1024² × 50 steps × CFG=7.5, prompt="An astronaut riding a green horse")

| Run | Total (s) | UNet total (s) | UNet/step (ms, batch=2) |
|---|---|---|---|
| Cold (seed 42) | 32.24 | 31.89 | 637.7 |
| Warm 1 | 32.01 | 31.69 | 633.7 |
| Warm 2 | 32.00 | 31.69 | 633.8 |
| Warm 3 | 32.19 | 31.88 | 637.6 |
| Seed 43 (warm) | 31.99 | 31.68 | 633.6 |

**Cold = 32.24 s, warm mean (3 runs) = 32.07 s.**

### Per-stage breakdown (warm)

- UNet: 50 steps × 2 NEFF calls × ~317 ms = **31.69 s** (98.8 % of total)
- Text encoders + VAE decoder + scheduler + Python overhead: **~0.38 s** (1.2 %)

Per single batch=1 NEFF call: **~317 ms** (matches the standalone benchmark
warm of 322 ms; minor speedup likely from steady-state thermal/cache state).

## Cost per image

trn2.3xlarge full chip on-demand: **$2.235 / hr** (we use all 4 logical cores
via TP=4, so the entire instance cost applies).

- Compute cost / image (warm): $2.235 × 32.07 / 3600 = **$0.01991**
- Cold-path cost: $0.02002

## Comparison with reference points

| Platform | Latency / image | $/hr | $/image |
|---|---|---|---|
| **Trn2 TP=4 UNet (this work)** | **32.07 s** | **$2.235 (full chip)** | **$0.01991** |
| Trn2 DP=2 (whn09 fork, half chip) | 46 s | $1.118 | $0.0143 |
| H100 BF16 | 3.84 s | $4.33 | $0.00462 |
| H100 FP8 | 1.84 s | $4.33 | $0.00221 |
| L4 BF16 | 19.75 s | $1.323 | $0.00726 |
| L4 FP8 + compile | 12.68 s | $1.323 | $0.00466 |

**Findings:**
- TP=4 sharding makes the UNet ~36 % faster than the whn09 fork's DP=2 baseline
  (32.07 s vs 46 s) but the per-image price is **39 % higher** because we
  consume the whole chip rather than half. TP=4 wins latency, DP=2 wins $/image.
- Versus GPUs we are 3–17× slower in wall-clock and 4–9× more expensive per
  image. Closing the gap requires either FP8 (not yet available for SDXL
  on trn2), faster UNet (e.g. fold CFG into a single batch=2 NEFF), or
  batched generation amortising fixed costs.

## Composition notes / issues hit

- **Compile cache miss.** Re-running `builder.trace + builder.compile` from a
  fresh process did not hit the previous artifact (cache key changed across
  trace re-run); it recompiled in 13.7 min. Caching ModelBuilder NEFFs across
  processes is a known wart. End-to-end script is otherwise self-contained.
- **CFG batch handling.** Plain `pipe.unet(latents, ...)` is called with
  `latent_model_input` of shape `[2,4,128,128]` for CFG. Wrapper splits into
  two batch=1 NEFF calls — clean, no recompile needed.
- **Timestep shape.** NEFF expects `timestep` shape `[1]`; diffusers passes
  scalar tensors. Wrapper coerces with `timestep.long().reshape(-1)[:1]`.
- **No issues with text encoders / VAE.** Jim's `NeuronTextEncoder` and
  traced VAE decoder dropped in unmodified.
- **EFA warnings benign.** `nccl_net_ofi_create_plugin` errors are harmless
  on a single-instance TP run and do not affect correctness.

## Outputs (on remote)

- `/home/ubuntu/work_unet_tp4/sdxl_1k_e2e_tp4.py` — driver script
- `/home/ubuntu/work_unet_tp4/E2E_1K_TP4_METRICS.json` — full metrics
- `/home/ubuntu/work_unet_tp4/sdxl_1k_tp4_seed42.png` — primary output (1.83 MB)
- `/home/ubuntu/work_unet_tp4/sdxl_1k_tp4_seed43.png` — bonus seed (1.62 MB)
- `/home/ubuntu/work_unet_tp4/e2e_1k_run.log` — full run log
