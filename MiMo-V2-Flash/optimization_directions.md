# Optimization Directions for MiMo-V2-Flash on Trn2

Ranked by evidence strength and expected impact. Each entry lists: **evidence** (from profile data, not speculation), **hypothesis**, **expected gain**, **cost to validate**.

---

## Tier 1 — Strong profile evidence

### 1. Close the 100 ms host-side gap at c>1

- **Evidence**: BF16 BS=32 device time = 34.7 ms/token; measured vLLM TPOT at c=32 = 135 ms. 100 ms of per-token time is host-side.
- **Hypothesis**: Python MoE dispatch + vLLM V1 scheduler + collective setup together cost 100 ms per decode step at c=32, but only ~0 ms at c=1 (because c=1 uses fast-path scheduling and static routing).
- **Expected gain**: 50–60% throughput at c=32 if we can halve the host gap (135 ms → 85 ms would match PR).
- **Cost to validate**: 1–2 h with `py-spy record` while serving at c=32. No new capacity block needed if we can re-use the existing compiled NEFFs from S3.
- **Action**: Run `py-spy record --pid <vllm-engine-pid> --duration 60` during a bench, inspect the flame graph, identify top 3 host functions.

### 2. Switch the MoE blockwise kernel flag to `shard_on_intermediate` (FP8 standalone verified; vLLM unverified)

- **Scope where verified**: **FP8 standalone NxDI only** (`adapter.generate()` path). Not tested on:
  - **BF16 path** — uses `use_torch_block_wise=true` (Python fallback), so this flag does not apply.
  - **FP8 vLLM serving path** — same kernel underneath, but vLLM adds ~100 ms/token host overhead at c=32 which may dominate the 6 ms kernel gain. Needs empirical verification.
- **Evidence (FP8 standalone, MOE_EP=64, BS=32)**:
  - `use_shard_on_block_dynamic_while=true`: 137.87 tok/s
  - `use_shard_on_intermediate_dynamic_while=true`: **146.29 tok/s (+6%)**
- **Hypothesis**: `shard_on_intermediate` has a lower-overhead MoE kernel path on SDK 2.29. Not verified mechanistically.
- **Expected gain**:
  - FP8 standalone: +6% (proven)
  - FP8 vLLM: unknown — may be dominated by host overhead
  - BF16: not applicable
- **Cost to validate on vLLM**: One line change in `COMMON_MIMO_CONFIG` in `bench_mimo_v2_flash.sh`, recompile (~30 min), rerun the three-concurrency bench (~15 min). Can confirm or rule out the hypothesis.
- **Action**: (1) Update `bench_mimo_v2_flash.sh` (FP8 recipe) to use `use_shard_on_intermediate_dynamic_while: true`. (2) Re-bench c=1/16/32, compare to current 147 tok/s (c=32).

### 3. Re-enable `qkv_nki_kernel` and `flash_decoding`

- **Evidence**: PR README lists six NKI kernels explicitly disabled (`qkv_kernel_enabled`, `qkv_nki_kernel_enabled`, `qkv_cte_nki_kernel_fuse_rope`, `attn_kernel_enabled`, `strided_context_parallel_kernel_enabled`, `flash_decoding_enabled`). Disabling is a compatibility workaround for Q/K=192 / V=128 asymmetry.
- **Hypothesis**: Some of these six kernels actually tolerate the asymmetry and were disabled defensively. Each kernel typically gives 10–30% on its op.
- **Expected gain**: Unknown per kernel; cumulative 20–40% if 2–3 of them work.
- **Cost to validate**: One capacity block, 6 compiles (~40 min each) + 6 benches (~5 min each) = ~4.5 h of instance time (~$120 on a 7h block).
- **Action**: Flip one flag at a time, verify accuracy (generate a known prompt, cosine-sim against BF16 baseline), measure throughput.

---

## Tier 2 — Compiler-side (depends on AWS Neuron team)

### 4. Wait for `blockwise_mm_baseline_shard_hidden` NKI kernel in SDK 2.30

- **Evidence**: PR source comment says this kernel is the designed default for MoE blockwise but is missing from SDK 2.29's public surface. `_call_shard_hidden_kernel` is the default code path but fails to link.
- **Hypothesis**: When the kernel ships, MoE FP8 path can hit compiler-predicted mfu_max = 100% (vs current 38%).
- **Expected gain**: PR author's own estimate is "FP8 ≈ 2× BF16 path". Current: FP8 is ~same as BF16. Potential 2× uplift.
- **Cost to validate**: Zero — just wait for Neuron release, re-run same bench.
- **Action**: Watch `neuronxcc` release notes. Re-bench when the symbol appears in `neuronxcc.nki._private.blockwise_mm`.

### 5. Close the CTE mfu gap (38% actual vs 100% achievable)

- **Evidence**: `neuron-profile view` on FP8 BS=32 CTE NEFF reports `mfu_hlo_max_achievable_estimated_percent = 100%`, actual `mfu_estimated_percent = 38%`.
- **Hypothesis**: Intra-NEFF pipeline / dependency gaps. Not a config issue — a compiler scheduling issue.
- **Expected gain**: Up to 2.6× on CTE. CTE is only ~10% of TPOT at BS=32, so end-to-end gain ~10–15%.
- **Cost to validate**: File compiler ticket with NEFF hash + profile data. Out of our hands.
- **Action**: Include as evidence in a ticket to AWS Neuron.

---

## Tier 3 — Architecture / config sweeps

### 6. Try `enable_prefix_caching`

- **Evidence**: PR disables it. vLLM prefix cache amortizes CTE for repeated prefixes.
- **Hypothesis**: If workload has repeated system prompts, TTFT drops 3–5×.
- **Expected gain**: Workload-dependent; 0% if prompts are random, 50%+ if there's a shared system prompt.
- **Cost to validate**: One flag, recompile, rerun bench. ~40 min.
- **Action**: Test with a realistic chat-style dataset, not the `random` dataset PR uses.

### 7. Reduce `max_model_len` from 1024 to 512

- **Evidence**: PR uses 1024; KV cache scales linearly with seq_len.
- **Hypothesis**: Smaller KV cache → more active batch slots → more throughput (not latency).
- **Expected gain**: Up to 30% throughput if KV-cache-limited.
- **Cost to validate**: One flag, recompile, rerun.
- **Action**: Only if the target workload tolerates shorter context.

### 8. BS=128 throughput config

- **Evidence**: PR's `bench_mimo_v2_flash.sh` includes `bs128_tp1_ep64_opt` config we skipped.
- **Hypothesis**: BS=128 amortizes per-token overhead more; PR claims throughput grows ~3× from BS=32 to BS=128 elsewhere.
- **Expected gain**: 2–3× aggregate tok/s at c=32+.
- **Cost to validate**: One capacity block, long (~2 h) BS=128 compile.
- **Action**: Run after optimizations in Tier 1 land.

---

## Tier 4 — Only if nothing else helps

### 9. Write our own NKI blockwise MoE kernel

- **Evidence**: PR can't use `shard_hidden`, our FP8 runs all fall back to `shard_on_block` or `shard_on_intermediate`, both suboptimal per profile.
- **Hypothesis**: A hand-written NKI kernel matching `shard_hidden`'s intended layout could unlock the 2.6× CTE headroom and the MoE torch-fallback penalty.
- **Expected gain**: Potentially 2× FP8 throughput.
- **Cost**: 1–2 weeks of NKI engineering + testing + NxDI integration.
- **Action**: Only if #4 (wait for SDK 2.30) is blocked >2 months.

### 10. SDK + driver sweep

- **Evidence**: We ran driver 2.27.4 + neuronx-cc 2.24.5133. PR author may run a different patch level.
- **Hypothesis**: c>1 gap is sensitive to driver/runtime version (scheduling changes between patches).
- **Expected gain**: Could be 0, could be 30%. Unknown.
- **Cost**: Each SDK refresh needs one capacity block ($250–500).
- **Action**: Re-bench on next DLAMI release only if we're investigating c>1 gap and other options exhausted.

### 11. AZ / physical-machine variance investigation

- **Evidence**: c=16/c=32 deltas (29% and 35%) vs PR are suspicious — could reflect hardware heterogeneity in us-east-2b capacity pool.
- **Hypothesis**: Different physical machines serve different capacity blocks; network topology and HBM binning vary.
- **Expected gain**: Unknown; finding would be "variance is X%", not a fix.
- **Cost**: 3–4 additional capacity blocks ($1000–1500 total) to build a distribution.
- **Action**: Only justified if Tier 1+2 close most of the gap but a residual remains.

---

## Priority recommendation

| Order | Action | When |
|---|---|---|
| 1 | `py-spy` trace on c=32 serving to locate host gap | Next capacity block, within first hour |
| 2 | Flip to `shard_on_intermediate` — FP8 path only (free +6%) | Same capacity block, hour 2 |
| 3 | Re-enable NKI kernels one by one | Same capacity block, hours 3–6 |
| 4 | File compiler ticket for CTE mfu gap | No instance needed |
| 5 | Watch SDK 2.30 release notes for `shard_hidden` | No instance needed, months out |

If all of #1–#3 land, expect end-to-end throughput to move from our current 196 tok/s (c=32) toward PR's 302 tok/s, and possibly beyond.
