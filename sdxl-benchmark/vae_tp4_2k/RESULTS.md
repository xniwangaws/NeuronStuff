# SDXL VAE decoder TP=4 single-NEFF compile at 2K — result

## TL;DR

TP=4 sharding **structurally works** — graph reduces below the `--tiled-inst-limit 10M` ceiling cleanly — but the **compile-host CPU memory** runs out on `trn2.3xlarge` (124 GB RAM). Two separate compile attempts were SIGKILL'd by the Linux OOM-killer at the same `walrus_driver` stage (exit -9, F137). 64 GB of swap was insufficient. **The bottleneck is no longer Trainium HBM or NCC_EVRF007; it is the compile host.**

## Did TP=4 compile succeed at 2K?

**No** — `walrus_driver` was forcibly killed (Linux OOM-killer) on the trn2.3xlarge compile host:
- Run 1 (no swap): RSS reached 128 GB / 124 GB → SIGKILL after ~17 min wall-clock
- Run 2 (64 GB swap added): peak RSS 125 GB + 52 GB swap = ~178 GB working set, then `walrus_driver` killed again at ~40 min wall-clock during partition `sg03`

Per-partition peak walrus RSS:
- `sg01` (104k inst): 44 GB
- `sg00` (1.73M inst): 72 GB
- `sg03` (897k inst): **125 GB** ← OOM

## Structural validation passed

The TP=4 wrapper compiled cleanly through `Hlo2Tensorizer`, `OOMChecker`, partitioning, and most backend passes:

- Sharded `4` Linears (mid_block.attentions[0] q/k/v/out) and `32` Conv2d (all resnet conv1/conv2 + upsamplers + conv_shortcut). 4 small/non-divisible convs (post_quant_conv, conv_in, conv_out, conv_norm_out) correctly stayed replicated.
- Trace via `ModelBuilder` finished in `0.2 s`. State-dict alignment: 0 missing, 0 extra (master state dict slots 1:1 onto sharded module).
- After TP=4 partition+sharding the **largest partition graph has 1.73M instructions** (sg00) — well below the `--tiled-inst-limit 10000000` ceiling. So Luo's flag was no longer the limiter; the structural goal of escaping NCC_EVRF007 / NCC_EXSP001 succeeded.

This says: at the IR level, TP=4 *would* fit Trainium HBM at 2K (the 38 GB graph activation memory predicted by kaena-30652 has been split). The blocker is purely compile-host RAM during the walrus scheduling stage on the largest partition.

## Why the compile host runs out

`walrus_driver` builds in-memory IntervalTrees + neighbor partner graphs to do SB allocator coloring. Per-partition graph size is what drives this. The original (un-sharded) VAE decoder 2K graph had ~7.69M HLO instructions; TP=4 splits per-rank instructions but ModelBuilder still has to compile **the whole partition graph in one process** — and partition `sg03` post-sharding still has 897k instructions tiled across the 4 ranks, which translates to a peak of ~125 GB working set in the SB allocator pass.

This is the same class of compile-host RAM blowup that the previous `vae_perblock_workaround` (kaena-30652) tried to dodge by chunking the decoder. TP=4 reduces the *device-side* memory but not the *compile-side* memory.

## What would unblock this

| Approach | Comment |
|---|---|
| **Compile on a larger host** (e.g., trn1.32xlarge with 512 GB, then transfer NEFF) | Most likely fix. NEFF is portable across same-target instances. |
| TP=8 instead of TP=4 | Would cut per-partition graph further (~half), might fit in 124 GB. Need LNC=1 with 8 logical cores. Worth trying next. |
| Per-block VAE chaining (Plan A in `vae_perblock_workaround/`) | Already at 3.10 s warm; functional baseline. |
| Mixed: VAE on CPU | Eliminates compile question, costs latency. |
| File a SIM follow-up to kaena-30652 | The HLO 5M and `--tiled-inst-limit` 10M flags are now successfully bypassed. The new wall is walrus host RSS. |

## Compare vs Plan A baseline

| Variant | NEFF count | Latency at 2K | Status |
|---|---|---|---|
| Plan A: per-block VAE chained on Neuron | many small NEFFs | **3.10 s warm** | working |
| Plan B: TP=4 single-NEFF (this attempt) | would be 1 | **N/A — compile-host OOM** | blocked |

## Recommendation for OPPO POC

**Stay on Plan A (per-block VAE chained)** for the 2K demo path. Plan A delivers 3.10 s VAE at 2K today and does not depend on host RAM.

For Plan B, file a SIM follow-up against kaena-30652 (or a new ticket) noting that with the `--tiled-inst-limit 10M` workaround applied + customer-side TP=4 sharding, the failure mode has moved from `NCC_EXSP001` (instruction limit) to `walrus_driver` host-RAM OOM during SB allocation on `trn2.3xlarge`. Ask AWS whether (a) a higher-memory NEFF compile host can be used for trn2 deployments, or (b) the walrus scheduling pass can be made more memory-efficient on multi-million-instruction partitions.

## Artifacts

- `neuron_vae_decoder_tp4.py` — TP=4 sharding wrapper (in-place module replacement). Walks `vae.decoder` + `vae.post_quant_conv`, swaps `nn.Conv2d` → `OutputChannelParallelConv2d(gather_output=True)` and `nn.Linear` → `ColumnParallelLinear(gather_output=True)`. Skips small + non-divisible (out_channels=3 conv_out, etc.).
- `trace_vae_tp4.py` — single-process compile + bench using `ModelBuilder` + `NxDParallelState(world_size=4, tp=4)` + `shard_checkpoint` (same pattern as `unet_tp4/trace_unet_tp4_mb.py`).
- `bench.json` — structured failure record + diagnostic numbers (peak RSS per partition, instruction counts, env).
- `compile_log_excerpt.txt` — head + memory milestones + tail of the second compile attempt (`walrus_driver` stage with `sg03` SB allocator pushing RSS to 125 GB).

## Reproduce

```bash
# trn2.3xlarge, sa-east-1, Neuron PyTorch 2.9 DLAMI 2026-05-22 (SDK 2.29 / neuronx-cc 2.25.3371)
sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile && \
  sudo mkswap /swapfile && sudo swapon /swapfile  # adds 64GB swap (still insufficient)
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
pip install diffusers==0.34.0 accelerate
HF_TOKEN=$(cat ~/.hf_token) hf download stabilityai/stable-diffusion-xl-base-1.0 \
  --local-dir ~/sdxl-base --include 'vae/*'
python trace_vae_tp4.py  # OOM at walrus_driver SB allocator pass on sg03
```

## SIM follow-up text (in case kaena-30652 update is wanted)

> FYI follow-up. Tried Plan B (TP=4 + your `--tiled-inst-limit 10000000` flag) on trn2.3xlarge sa-east-1. Wrapper monkey-patches diffusers VAE decoder convs/linears to NxD `OutputChannelParallelConv2d` / `ColumnParallelLinear` (gather_output=True), then compiles via `ModelBuilder` + `NxDParallelState(world_size=4, tp=4)`.
>
> Structural shard works: 32 convs + 4 linears sharded, state-dict aligned, post-shard largest partition graph drops to **1.73M instructions** — comfortably below your 10M limit, so the original `NCC_EVRF007/NCC_EXSP001` ceiling is no longer the wall. **However compile fails with F137 (`walrus_driver` SIGKILL) — host CPU RAM blew past 125 GB on partition sg03 SB allocator.** trn2.3xl only has 124 GB; adding 64 GB swap was not enough either. Per-partition peak walrus RSS: sg00=72 GB, sg03=125 GB.
>
> So the new wall for VAE 2K on a single trn2.3xl is **compile-host memory**, not Trainium HBM or instruction count. Posting in case useful for other customers hitting this — recommended workaround for them is to compile on a larger host and ship the NEFF, or TP=8 to shrink per-partition further.
