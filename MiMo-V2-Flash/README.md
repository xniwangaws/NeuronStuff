# MiMo-V2-Flash on Trn2 — Profile Findings & Optimization Directions

Work resulting from profiling [NxDI PR #137](https://github.com/aws-neuron/neuronx-distributed-inference/pull/137) (Xiaomi MiMo-V2-Flash MoE on Trainium 2) across **~15 hours** of instance time on 2× trn2.48xlarge capacity blocks (us-east-2b, SDK 2.29). Total spend: ~$824.

## What's in here

| Path | Purpose |
|---|---|
| `README.md` (this file) | Executive summary |
| `findings.md` | What we observed — measurements, not speculation |
| `optimization_directions.md` | Ranked optimization proposals with evidence + expected gain + cost |
| `sop_capacity_block_survival.md` | Three-daemon S3 sync setup so `/opt/dlami/nvme` never gets lost |
| `profile_data/` | Raw `neuron-profile` summary outputs (FP8 + BF16, BS=1 and BS=32) |
| `scripts/` | The patched preprocess + experiment scripts used |
| `analysis/` | Processed comparison tables |

## TL;DR of the measurements

| Config | Our tok/s | PR tok/s | Delta | TPOT (our / PR) |
|---|---:|---:|---:|---:|
| BF16 BS=32 vLLM c=1 | **26.61** | 27.98 | **-5%** ✅ | **33.72 / 33.65 ms** (0.2%) |
| BF16 BS=32 vLLM c=16 | 159.94 | 224.57 | -29% | 89.00 / 64.95 ms |
| BF16 BS=32 vLLM c=32 | 196.58 | 302.61 | -35% | 135.35 / 90.23 ms |
| FP8 BS=32 vLLM c=32 | 147.21 | not reported | — | 188 ms |
| FP8 BS=32 standalone `shard_on_intermediate` | 146.29 | — | +6% vs `shard_on_block` | — |

**Key finding**: c=1 matches PR perfectly (device-bound). c>1 is ~30% slower than PR — the gap is 100ms of host-side overhead per decoded token at c=32 (device NEFF takes only 35ms).

## The headline profile numbers

### BF16 BS=32 TGEN (per-step, device-measured via `neuron-profile`)
```
total_time:                              34.7 ms
tensor_engine_active_time_percent:        47%
dma_active_time_percent:                  86%   ← memory-bound
mfu_estimated_percent:                    5.8%
mfu_hlo_max_achievable_estimated_percent: 12.5% ← 2× headroom
mbu_estimated_percent:                    46%
hbm_read_bytes:                           11 GB
```

### BF16 BS=32 CTE (prefill, per-call)
```
total_time:                              348.8 ms
tensor_engine_active_time_percent:        69%   ← good compute utilization
dma_active_time_percent:                  82%
mfu_estimated_percent:                    42%
mfu_hlo_max_achievable_estimated_percent: 73%
hbm_read_bytes:                           119 GB (~BF16 weight stream)
```

See `findings.md` for FP8 + BS=1 tables and full interpretation.

## If you want to reproduce

1. Follow `sop_capacity_block_survival.md` Phase 0 before any work (else lose everything when block expires).
2. Source checkpoints are at `s3://xniwang-neuron-models-us-east-2/mimo-v2-flash/{MiMo-V2-Flash-FP8,MiMo-V2-Flash-Neuron-FP8}/` — skip HF download + preprocess (27min → 15min).
3. Scripts in `scripts/` are drop-in for `~/neuronx-distributed-inference/contrib/models/MiMo-V2-Flash/`.
4. Raw summary outputs in `profile_data/` are the actual `neuron-profile view --output-format summary-text` dumps.

## Artifacts not in this repo

Too large for git:
- `s3://xniwang-neuron-models-us-east-2/mimo-v2-flash/profile/mimo_bf16_results_20260427T1004Z.tgz` (2.3 GB) — full BF16 bench logs + rank_0 .ntff traces
- `s3://.../profile/mimo_v2_flash_profile_20260426T2048Z.tgz` (54 MB) — FP8 BS=1 smoke profile
- Preprocessed Neuron-FP8 and Neuron-BF16 checkpoints (311 GB + 617 GB)
